#!/usr/bin/env python3
"""Offline Isaac Sim: two hollow cups, top RGB, measured O6/wrist, ACT and ServoP outer loop.

No robot SDK, WebSocket client or CAN is instantiated. PhysX drives plus local
IK approximate the inner ServoP plant; the existing replay outer loop is reused.
Run with env_isaaclab Python. Images are rendered current simulation observations.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSET = Path('/home/zmy/Dexverse_workspace/DexVerse/model/tron2-robot-description/tron2a/DACH_TRON2A/usd/robot_with_o6_hands.usd')
BASELINE = ROOT/'artifacts/tron2_act/cup_stack_baseline_20261117_01'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=BASELINE/'train/policy_best.ckpt')
    parser.add_argument('--foreground', type=Path, default=ROOT/'artifacts/top_foreground_pipeline_20261116/segmenter/foreground.pt')
    parser.add_argument('--seconds', type=float, default=15.)
    parser.add_argument('--scene-config',type=Path,help='Task scene appearance/mount/preview configuration')
    parser.add_argument('--task-validation',action='store_true',help='Score contact/grasp/lift/release/stability using physics only')
    parser.add_argument('--mode', choices=['act', 'hold', 'probe'], default='act')
    parser.add_argument('--no-feedback', action='store_true')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--raw-reference', action='store_true', help='Reproduce the initial unconditioned chunk experiment')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, obj):
        (args.output/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    save('status.json', {'state':'starting', 'hardware_connections':0})
    sources = [Path(__file__),ROOT/'tron2_act/simulation.py',ROOT/'tron2_pose_reference.py',ROOT/'tron2_pose_compensation.py',ROOT/'tron2_act/task_validation.py',ROOT/'tools/tron2_act_scene.py',ROOT/'tron2_act/simulation_ik.py']
    save('source_sha256.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    from isaacsim import SimulationApp
    app = SimulationApp({'headless':not args.gui, 'width':960, 'height':720,
                         'extra_args':['--portable-root='+str(args.output/'kit')]})
    try:
        run(app, args, save)
    except BaseException as exc:
        save('status.json', {'state':'failed', 'error':str(exc), 'traceback':traceback.format_exc(), 'hardware_connections':0})
        raise
    finally:
        app.close()


def run(app, args, save):
    import cv2
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from pxr import Gf, UsdGeom, UsdPhysics, UsdLux, UsdShade, PhysxSchema
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, RigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.utils.xforms import get_world_pose
    from isaacsim.sensors.camera import Camera
    from tools.simulate_tron2_z_feedback import NAMES, TCP, KP, KD
    from tron2_act.config import CONTRACT, Config
    from tron2_act.model import ACT
    from tron2_act.simulation import (SimServoP, ActionChunk, PoseHistory, HAND_JOINTS,
                                     hand_to_angles, angles_to_hand, HandCommandClock)
    from tron2_vision.model import ForegroundPredictor

    from tools.tron2_act_scene import rotate_hand_mount, camera_rotation
    from tron2_act.task_validation import CupTaskMonitor, mask_diagnostics
    scene_cfg = {} if args.scene_config is None else json.loads(args.scene_config.read_text())
    initial_hand = np.array(scene_cfg.get('initial_o6_sdk',[255.]*6))
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    dt = 1/300
    base_height = 1.25
    world = World(physics_dt=dt, rendering_dt=1/20, stage_units_in_meters=1.)
    stage = omni.usd.get_context().get_stage()
    world.scene.add_default_ground_plane()
    add_reference_to_stage(str(ASSET), '/World/Robot')
    robot = world.scene.add(SingleArticulation('/World/Robot', name='robot', position=np.array([0.,0.,base_height])))
    if 'o6_mount_z_rotation_deg' in scene_cfg:
        rotate_hand_mount(stage,scene_cfg['o6_mount_z_rotation_deg'])
    fixed = UsdPhysics.FixedJoint.Define(stage, '/World/Robot/joints/fixed_base')
    fixed.CreateBody1Rel().SetTargets(['/World/Robot/base_Link'])
    fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(0,0,base_height))
    fixed.CreateLocalRot0Attr().Set(Gf.Quatf(1))
    fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(0))
    fixed.CreateLocalRot1Attr().Set(Gf.Quatf(1))
    for i, name in enumerate(NAMES):
        prim = stage.GetPrimAtPath('/World/Robot/joints/'+name)
        drive = UsdPhysics.DriveAPI.Apply(prim, 'angular')
        drive.CreateTypeAttr().Set('force')
        # USD angular drive gains are per degree. The reference model uses radians.
        # Isaac's runtime get_gains()/set_gains() tensor API uses per-radian gains.
        drive.CreateStiffnessAttr().Set(float(KP[i])*np.pi/180)
        drive.CreateDampingAttr().Set(float(KD[i])*np.pi/180)
        drive.CreateMaxForceAttr().Set(100.)
    material = UsdShade.Material.Define(stage, '/World/ContactMaterial')
    mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mat.CreateStaticFrictionAttr(.8); mat.CreateDynamicFrictionAttr(.6); mat.CreateRestitutionAttr(0.)
    def collider(geom):
        prim = geom.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        PhysxSchema.PhysxCollisionAPI.Apply(prim).CreateContactOffsetAttr(.001)
        PhysxSchema.PhysxCollisionAPI(prim).CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')
    table = UsdGeom.Cube.Define(stage, '/World/Table')
    table.CreateSizeAttr(1.)
    table.AddTranslateOp().Set(Gf.Vec3d(.63,.15,.83))
    table.AddScaleOp().Set(Gf.Vec3d(.56,.85,.04))
    table.CreateDisplayColorAttr([tuple(scene_cfg.get('table_color',[.34,.30,.26]))])
    collider(table)
    cups = []
    cup_positions = scene_cfg.get('cup_positions',[[.55,.27],[.55,.10]])
    cup_colors = scene_cfg.get('cup_colors',[[.85,.86,.89],[.22,.42,.75]])
    for k, (xy, color) in enumerate(zip(cup_positions,cup_colors)):
        path = '/World/Cup'+str(k)
        cup = UsdGeom.Xform.Define(stage, path)
        cup.AddTranslateOp().Set(Gf.Vec3d(xy[0],xy[1],scene_cfg.get('cup_bottom_z',[.851,.851])[k]))
        UsdPhysics.RigidBodyAPI.Apply(cup.GetPrim())
        UsdPhysics.MassAPI.Apply(cup.GetPrim()).CreateMassAttr(.045)
        bodyapi = PhysxSchema.PhysxRigidBodyAPI.Apply(cup.GetPrim())
        bodyapi.CreateSolverPositionIterationCountAttr(16)
        bodyapi.CreateSolverVelocityIterationCountAttr(4)
        base = UsdGeom.Cylinder.Define(stage, path+'/Bottom')
        base.CreateRadiusAttr(.03); base.CreateHeightAttr(.003)
        base.AddTranslateOp().Set(Gf.Vec3d(0,0,.0015))
        base.CreateDisplayColorAttr([color]); collider(base)
        # Separate convex wall sectors leave the mouth and interior open.
        for j in range(32):
            a, b = 2*np.pi*j/32, 2*np.pi*(j+1)/32
            points = []
            for z, radius in [(0.,.03),(.09,.04)]:
                for radius1, angle in [(radius,a),(radius,b),(radius-.002,b),(radius-.002,a)]:
                    points.append((radius1*np.cos(angle),radius1*np.sin(angle),z))
            mesh = UsdGeom.Mesh.Define(stage, path+f'/Wall{j:02}')
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr([4]*6)
            mesh.CreateFaceVertexIndicesAttr([0,3,2,1,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7])
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateDoubleSidedAttr(True)
            inner = scene_cfg.get('cup_inner_colors',cup_colors)[k]
            mesh.CreateDisplayColorAttr([tuple(inner),tuple(color),tuple(color),tuple(color),tuple(inner),tuple(color)])
            mesh.GetDisplayColorPrimvar().SetInterpolation('uniform')
            collider(mesh)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        cups.append(world.scene.add(SingleRigidPrim(path, name='cup'+str(k))))
    contact_paths = [f'/World/Robot/o6_L_{name}' for name in ['thumb_distal','thumb_metacarpals','index_distal','index_proximal','middle_distal','middle_proximal','ring_distal','ring_proximal','pinky_distal','pinky_proximal','hand_base_link']]
    contact_view = None
    if args.task_validation:
        from isaacsim.core.utils.semantics import add_update_semantics
        for prim in stage.GetPrimAtPath('/World/Robot').GetChildren():
            if prim.GetName().startswith('o6_L_'):
                add_update_semantics(prim,'hand')
        for i in range(2):add_update_semantics(stage.GetPrimAtPath('/World/Cup'+str(i)),'cup'+str(i))
        contact_view = world.scene.add(RigidPrim('/World/Cup0',name='task_contacts',reset_xform_properties=False,contact_filter_prim_paths_expr=contact_paths+['/World/Cup1'],max_contact_count=32768))
    dome = UsdLux.DomeLight.Define(stage, '/World/Light')
    dome.CreateIntensityAttr(900.)
    camera = Camera('/World/Top', position=np.array([.52,.20,1.85]),
                    orientation=np.array([1.,0.,0.,0.]), resolution=(640,480), frequency=20)
    world.scene.add(camera)
    wrist_body = world.scene.add(SingleRigidPrim('/World/Robot/wrist_roll_L_Link', name='wrist_feedback', reset_xform_properties=False))
    base_body = world.scene.add(SingleRigidPrim('/World/Robot/base_Link', name='base_feedback', reset_xform_properties=False))
    world.reset()
    camera.initialize()
    cam_pos = np.array(scene_cfg.get('camera_position',[.52,.20,1.85]))
    cam_rot = np.array([1.,0.,0.,0.]) if 'camera_target' not in scene_cfg else camera_rotation(cam_pos,scene_cfg['camera_target'])
    camera.set_world_pose(position=cam_pos,orientation=cam_rot,camera_axes='usd')
    if args.task_validation:camera.add_semantic_segmentation_to_frame()
    camera.set_focal_length(scene_cfg.get('camera_focal_length',20.))
    camera.set_horizontal_aperture(20.955)
    camera.set_clipping_range(.01,10.)
    names = list(robot.dof_names)
    arm_indices = np.array([names.index(x) for x in NAMES])
    read_kp,read_kd = robot._articulation_view.get_gains(joint_indices=arm_indices)
    gain_readback = {'units':'Nm/rad and Nm*s/rad', 'requested_kp':KP.tolist(),
                     'requested_kd':KD.tolist(), 'runtime_kp':np.asarray(read_kp)[0].tolist(),
                     'runtime_kd':np.asarray(read_kd)[0].tolist(),
                     'usd_angular_gain_scale':float(np.pi/180)}
    save('drive_gain_readback.json',gain_readback)
    if not (np.allclose(read_kp[0],KP,rtol=1e-5,atol=1e-6) and
            np.allclose(read_kd[0],KD,rtol=1e-5,atol=1e-6)):
        raise RuntimeError('PhysX drive gains do not match the radian reference model')
    hand_indices = np.array([names.index('o6_L_'+x) for x in HAND_JOINTS])
    q0 = np.zeros(16)
    q0[:7] = [0,0,.18322281083347552,-np.pi/2,-np.pi/2,0,0]
    q0[9] = -1.4
    all_q = np.zeros(len(names)); all_q[arm_indices] = q0
    all_q[hand_indices] = hand_to_angles(initial_hand)
    # Initialize mimic joints consistently, avoiding an artificial startup finger impulse.
    for suffix,ratio,reference in [('thumb_ip',1.83,'thumb_cmc_pitch'),('index_dip',.89,'index_mcp_pitch'),('middle_dip',.89,'middle_mcp_pitch'),('ring_dip',.89,'ring_mcp_pitch'),('pinky_dip',.89,'pinky_mcp_pitch')]:
        all_q[names.index('o6_L_'+suffix)] = ratio*all_q[names.index('o6_L_'+reference)]
    # O6 measured state is read from the articulated physics joints thereafter.
    robot.set_joint_positions(all_q)
    robot.set_joint_velocities(np.zeros(len(names)))
    robot.apply_action(ArticulationAction(joint_positions=all_q))
    save('before_step.json', {'root_pose':[np.asarray(x).tolist() for x in robot.get_world_pose()], 'base_pose':[np.asarray(x).tolist() for x in base_body.get_world_pose()], 'fixed_joints':[{"path":str(pr.GetPath()),"b0":[str(x) for x in UsdPhysics.Joint(pr).GetBody0Rel().GetTargets()],"b1":[str(x) for x in UsdPhysics.Joint(pr).GetBody1Rel().GetTargets()],"pos0":str(UsdPhysics.Joint(pr).GetLocalPos0Attr().Get())} for pr in stage.Traverse() if pr.IsA(UsdPhysics.FixedJoint) and not UsdPhysics.Joint(pr).GetBody0Rel().GetTargets()]})
    world.step(render=True)
    engine = SimServoP(q0, feedback=not args.no_feedback, continuous_reference=not args.raw_reference,preview_s=scene_cfg.get('servo_preview_s'),ik_solver=scene_cfg.get('simulator_ik_solver','legacy'))
    def pose():
        wp, wq = wrist_body.get_world_pose()
        bp, _ = base_body.get_world_pose()
        r = Rotation.from_quat(np.asarray(wq)[[1,2,3,0]]).as_matrix()
        return np.asarray(wp)+r@TCP-np.asarray(bp), r
    actual_initial_p,actual_initial_r = pose()
    modelp,modelr,_ = engine.engine.forward(q0)
    p0,r0 = modelp.copy(),modelr.copy()
    measured_model_p,measured_model_r,_ = engine.engine.forward(np.asarray(robot.get_joint_positions())[arm_indices])
    mismatch = {'position_mm':float(np.linalg.norm(actual_initial_p-measured_model_p)*1000),
                'rotation_deg':float(np.rad2deg(Rotation.from_matrix(actual_initial_r@measured_model_r.T).magnitude()))}
    save('initial_diagnostic.json', {'q0':q0.tolist(),'q_actual':robot.get_joint_positions()[arm_indices].tolist(),'actual_p':actual_initial_p.tolist(),'model_p':modelp.tolist(),'actual_r':actual_initial_r.tolist(),'model_r':modelr.tolist(),'base':base_body.get_world_pose()[0].tolist(),'wrist':wrist_body.get_world_pose()[0].tolist(),'mismatch':mismatch})
    if mismatch['position_mm'] > 2 or mismatch['rotation_deg'] > .5:
        raise RuntimeError('USD / ServoP reference model mismatch: '+str(mismatch))
    contract = {'schema':'tron2-act-isaac-validation-v1', 'mode':args.mode,
        'hardware_connections':0, 'asset':str(ASSET), 'physics_hz':300, 'act_hz':20,
        'feedback_hz':20, 'feedback_enabled':not args.no_feedback,
        'controller':'existing pose-reference-v1 + PoseCorrection; local IK/PhysX inner plant',
        'vendor_servop_runtime':False, 'controller_profile':engine.profile,
        'pose_feedback_config':asdict(engine.controller.config), 'anchor_model_mismatch':mismatch,
        'o6_sdk_to_angle':'(255-command)/255 * URDF independent-joint limit; approximate, not identified',
        'cup_geometry':{'height_m':.09,'bottom_radius_m':.03,'top_radius_m':.04,'wall_m':.002,'mass_kg':.045,'wall_convex_parts':32},
        'scene_calibrated_to_real':False, 'foreground':'same frozen student used for training, no privileged masks',
        'limitations':['Cup sizes/positions, camera and material are assumed, not calibrated.',
                       'Outer ServoP reference is shared; PhysX IK/PD is not vendor firmware.',
                       'SDK wrist TCP is used as the Tracker proxy, matching the current execution contract.',
                       'O6 SDK units to simulated joint angles are approximate; no tactile input.']}
    contract.update(scene_config=scene_cfg,task_validation=args.task_validation,servo_preview_s=engine.preview_s,privileged_state_used_for_policy=False,drive_gains=gain_readback)
    save('contract.json', contract)
    stage.GetRootLayer().Export(str(args.output/'scene.usda'))
    print('SCENE_READY '+json.dumps(mismatch), flush=True)
    fixed_target = lambda t: (p0.copy(),r0.copy(),initial_hand.copy())
    nominal = fixed_target
    hand_clock = HandCommandClock(initial_hand)
    history = PoseHistory()
    records = []
    warmup_records = []
    writer = cv2.VideoWriter(str(args.output/'top.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1280,480))
    trace = (args.output/'trace.jsonl').open('w')
    model = predictor = None
    inference_ms = []
    mask_counts = []
    chunks = []
    conditioning_errors = []
    if args.mode == 'act':
        bundle = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if bundle['contract'] != CONTRACT:
            raise ValueError('ACT checkpoint contract mismatch')
        config = Config(**bundle['config'])
        if config.action_hz != 20 or config.history_steps != 2:
            raise ValueError('This adapter requires 20Hz ACT with two wrist history steps')
        model = ACT(config,load_pretrained=False).cuda().eval()
        model.load_state_dict(bundle['model'])
        stats = {k:np.array(bundle['stats'][k],np.float32) for k in ['state_mean','state_std','action_mean','action_std']}
        predictor = ForegroundPredictor(args.foreground,device='cuda')
        for path,key in [(args.checkpoint,'checkpoint_sha256'),(args.foreground,'foreground_sha256')]:
            contract[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        contract['training_time_scale'] = config.time_scale
        contract['execution_time_scale'] = 1.
        contract['chunk_size'] = config.chunk_size
        contract['policy_execution'] = 'replan every 50ms from measured current pose; spatial rotation left-multiplies anchor; 124ms preview within predicted chunk'
        save('contract.json',contract)
        # Warm kernels before the physics rollout, never feed these images as observations.
        with torch.inference_mode():
            model(torch.zeros(1,3,*config.image_size,device='cuda'),torch.zeros(1,config.state_dim,device='cuda'))
        torch.cuda.synchronize()
    task_start = None
    stop_reason = None
    stable_stack_s = 0.
    success = False
    max_lift = 0.
    task_monitor = None
    task_events = []
    perception_diagnostics = []
    # Five-second model ramp and a hold check before enabling the policy.
    warmup_s = 8.
    total_s = warmup_s+(1. if args.mode == 'probe' else args.seconds)
    started = time.perf_counter()
    try:
        for step in range(round(total_s*300)):
            t = step*dt
            p,r = pose()
            hand = angles_to_hand(np.asarray(robot.get_joint_positions())[hand_indices])
            history.append(t,p,r)
            np_,nr,_ = nominal(t)
            hand_command = hand_clock.advance(t)
            if step % 15 == 0:
                engine.observe(t,np_,nr,p,r)
            ramp = min(1.,t/5)
            ramp = ramp*ramp*(3-2*ramp)
            if t >= warmup_s and task_start is None:
                ep = np.linalg.norm(p-p0)
                er = np.rad2deg(Rotation.from_matrix(r@r0.T).magnitude())
                if ep > .003 or er > 1.:
                    raise RuntimeError(f'Initial hold did not settle: {ep*1000:.2f}mm / {er:.2f}deg')
                task_start = t
                if args.task_validation:
                    task_monitor = CupTaskMonitor(cups[0].get_world_pose()[0],cups[1].get_world_pose()[0])
                print('POLICY_START',flush=True)
            if step % 15 == 0:
                # Camera frames are refreshed at the same simulation timestamps as proprioception.
                world.render()
                rgba = camera.get_rgba()
                for _ in range(8):
                    if rgba is not None and rgba.shape[:2] == (480,640):
                        break
                    world.render()
                    rgba = camera.get_rgba()
                if rgba is None or rgba.shape[:2] != (480,640):
                    raise RuntimeError('Top camera has no valid RGB frame')
                rgb = np.ascontiguousarray(rgba[:,:,:3]).astype(np.uint8)
                fg = rgb
                inference = None
                counts = None
                if task_start is not None and model is not None:
                    tick = time.perf_counter()
                    segmented = predictor.predict(rgb)
                    fg = segmented['rgb']
                    if args.task_validation:
                        semantic_packet = camera._custom_annotators['semantic_segmentation'].get_data()
                        if isinstance(semantic_packet,dict) and np.asarray(semantic_packet.get('data')).shape[:2]==(480,640):
                            diagnostic = mask_diagnostics(segmented['labels'],np.asarray(semantic_packet['data']).squeeze(),semantic_packet['info']['idToLabels'])
                            perception_diagnostics.append({'t':t,**diagnostic})
                    counts = [int(segmented[x].sum()) for x in ['hand','objects']]
                    mask_counts.append(counts)
                    if min(counts) == 0:
                        cv2.imwrite(str(args.output/'empty_mask_rgb.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                        cv2.imwrite(str(args.output/'empty_mask_foreground.png'),cv2.cvtColor(fg,cv2.COLOR_RGB2BGR))
                        raise RuntimeError(f'Sim domain foreground empty: hand={counts[0]}, objects={counts[1]}')
                    state = history.state(t,p,r,hand,config.history_steps,config.action_hz)
                    state = (state-stats['state_mean'])/stats['state_std']
                    image = torch.from_numpy(fg.transpose(2,0,1).copy()).cuda().float()[None]/255
                    with torch.inference_mode():
                        output,_,_ = model(image,torch.from_numpy(state).cuda()[None])
                    action = output[0].cpu().numpy()*stats['action_std']+stats['action_mean']
                    inference = (time.perf_counter()-tick)*1000
                    inference_ms.append(inference)
                    chunks.append({'t':t,'anchor_p':p.copy(),'anchor_r':r.copy(),'action':action.copy()})
                    action_chunk = ActionChunk(t,p,r,hand,action,config.action_hz)
                    hand_clock.submit(action_chunk)
                    nominal = action_chunk.target
                panel = cv2.cvtColor(np.concatenate([rgb,fg],axis=1),cv2.COLOR_RGB2BGR)
                cv2.putText(panel,f'{args.mode} t={max(0.,t-warmup_s):.2f}s', (15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,180,255),2)
                writer.write(panel)
                if step == 0 or (task_start is not None and abs(t-task_start) < 1e-8):
                    cv2.imwrite(str(args.output/('initial_top.png' if step == 0 else 'policy_start.png')),panel)
                cup_poses = [np.asarray(c.get_world_pose()[0]).tolist() for c in cups]
                ep = float(np.linalg.norm(p-np_)*1000)
                er = float(np.rad2deg(Rotation.from_matrix(r@nr.T).magnitude()))
                row = {'t':t,'task_t':None if task_start is None else t-task_start,
                    'actual_p':p.tolist(),'nominal_p':np_.tolist(),'position_error_mm':ep,
                    'rotation_error_deg':er,'o6_q_pos':hand.tolist(),'o6_action_command':hand_command.tolist(),'cups':cup_poses,
                    'inference_ms':inference,'mask_pixels':counts,'joint_q':np.asarray(robot.get_joint_positions())[arm_indices].tolist(), 'feedback_correction_m':engine.controller.translation.tolist()}
                trace.write(json.dumps(row)+'\n'); trace.flush()
                (records if task_start is not None else warmup_records).append(row)
                if step % 300 == 0:
                    print(f't={t:.1f}s error={ep:.2f}mm/{er:.2f}deg',flush=True)
            qtarget, correction = engine.step(t,nominal,ramp)
            conditioning_errors.append(correction['reference_conditioning_error_m'])
            # Apply command events from the previous chunk before replacing it.
            # A newly queried first action is due 50ms later, not measured q_pos.
            command = hand_command
            ids = np.r_[arm_indices,hand_indices]
            targets = np.r_[qtarget,hand_to_angles(command)]
            robot.apply_action(ArticulationAction(joint_positions=targets,joint_indices=ids))
            world.step(render=False)
            if not np.isfinite(robot.get_joint_positions()).all():
                raise RuntimeError('Nonfinite articulation state')
            if task_start is not None:
                p_a,qa = cups[0].get_world_pose(); p_b,qb = cups[1].get_world_pose()
                max_lift = max(max_lift,float(p_a[2]-.851))
                upright = all(Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()[2,2]>.95 for q in [qa,qb])
                stacked = (np.linalg.norm(p_a[:2]-p_b[:2])<.008 and .015<p_a[2]-p_b[2]<.085
                           and abs(p_b[2]-.851)<.01 and upright and max_lift>.03)
                stable_stack_s = stable_stack_s+dt if stacked else 0.
                success |= stable_stack_s >= 1.
                if task_monitor is not None:
                    matrix = np.asarray(contact_view.get_contact_force_matrix(dt=dt))[0]
                    forces = np.linalg.norm(matrix,axis=1)
                    grouped = [forces[0:2].sum(),forces[2:4].sum(),forces[4:6].sum(),forces[6:8].sum(),forces[8:10].sum(),forces[10]]
                    palm_p,_ = get_world_pose('/World/Robot/o6_L_hand_base_link')
                    state = task_monitor.update(t-task_start,dt,p_a,p_b,
                        [Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()[2,2] for q in [qa,qb]],
                        [float(np.linalg.norm(c.get_linear_velocity())) for c in cups],
                        [float(np.linalg.norm(c.get_angular_velocity())) for c in cups],grouped,float(forces[-1]),float(np.linalg.norm(np.asarray(palm_p)-(p_a+np.array([0,0,.045])))))
                    if step%15==0:task_events.append(state)
                    success = task_monitor.success
    except RuntimeError as exc:
        stop_reason = str(exc)
        save('stop_diagnostic.json',{'sim_t':t,'measured_joint_q':np.asarray(robot.get_joint_positions())[arm_indices].tolist(),'nominal_p':nominal(t)[0].tolist(),'nominal_r':nominal(t)[1].tolist(),'conditioned_preview_r':engine.filtered_r.tolist(),'conditioned_preview_p':engine.filtered_p.tolist() if not args.raw_reference else None,'model_reference_joint_q':engine.engine.qref.tolist(),'model_issued_joint_q':engine.engine.qissued.tolist(),'model_lower_limits':engine.engine.lower.tolist(),'model_upper_limits':engine.engine.upper.tolist()})
        print('ROLLOUT_STOP '+stop_reason,flush=True)
    finally:
        writer.release();trace.close()
    def error_report(rows):
        if not rows:
            return None
        errors = np.array([x['position_error_mm'] for x in rows])
        return {'samples':len(rows),'position_rmse_mm':float(np.sqrt(np.mean(errors**2))),
            'position_p95_mm':float(np.percentile(errors,95)),
            'rotation_p95_deg':float(np.percentile([x['rotation_error_deg'] for x in rows],95)),
            'final_position_error_mm':rows[-1]['position_error_mm'],
            'final_rotation_error_deg':rows[-1]['rotation_error_deg']}
    if chunks:
        np.savez_compressed(args.output/'policy_chunks.npz', times=np.array([x['t'] for x in chunks]), anchor_p=np.array([x['anchor_p'] for x in chunks]), anchor_r=np.array([x['anchor_r'] for x in chunks]), action=np.array([x['action'] for x in chunks]))
    contract['reference_conditioning'] = None if args.raw_reference else {'translation_speed_m_s':.08,'translation_acceleration_m_s2':.3,'rotation_speed_deg_s':30,'rotation_acceleration_deg_s2':90}
    save('contract.json',contract)
    if args.task_validation:
        save('task_metrics.json',None if task_monitor is None else task_monitor.report())
        save('perception_diagnostics.json',perception_diagnostics)
        (args.output/'task_trace.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in task_events))
    report = {'hardware_connections':0,'mode':args.mode,'completed':stop_reason is None,
        'stop_reason':stop_reason,'policy_started':task_start is not None,
        'policy_queries':len(inference_ms),'task_simulated_s':0. if task_start is None else t-task_start,
        'o6_max_measured_change_sdk':None if not records else np.max(np.abs(np.array([x['o6_q_pos'] for x in records])-np.array(records[0]['o6_q_pos'])),axis=0).tolist(),
        'wall_elapsed_s':time.perf_counter()-started,'warmup':error_report(warmup_records),
        'tracking':error_report(records),'task_success':bool(success),'task_stages':None if task_monitor is None else task_monitor.report(),'maximum_cup0_lift_m':max_lift,
        'inference_p95_ms':None if not inference_ms else float(np.percentile(inference_ms,95)),
        'mask_pixels_min':None if not mask_counts else np.min(mask_counts,axis=0).tolist(),
        'reference_conditioning_max_mm':None if not conditioning_errors else float(max(conditioning_errors)*1000),
        'real_time_deadline_measured_only':True,'vendor_servop_runtime':False,
        'limitations':contract['limitations']}
    save('report.json',report)
    save('status.json',{'state':'completed' if stop_reason is None else 'stopped_by_guard', 'hardware_connections':0,'stop_reason':stop_reason})
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
