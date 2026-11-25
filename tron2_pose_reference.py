"""Offline model reference for an explicit experimental ServoP deployment.

No robot clients. Same g/Kp + Kd*dq/Kp Cartesian construction as simulation;
IK checks retain SDK/model intersections and the existing strict wrist bounds.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

VERSION='pose-reference-v1-experimental-20261105'
ROOT=Path(__file__).resolve().parent
DEFAULT_PROFILE=ROOT/'tracker_calibrations/pose_reference_experimental_20261105/profile.json'


class PoseIKUnavailable(RuntimeError):
    """A requested Cartesian pose could not be represented by the bounded model."""


def limit_correction(previous_p,previous_r,requested_p,requested_r,dt,linear_rate,angular_rate_deg):
    """Slew only the correction, preserving the nominal trajectory and clock."""
    if not np.isfinite([dt,linear_rate,angular_rate_deg]).all() or min(dt,linear_rate,angular_rate_deg)<=0:
        raise ValueError('Invalid correction slew parameters')
    dp=requested_p-previous_p
    rv=Rotation.from_matrix(requested_r@previous_r.T).as_rotvec()
    a=min(1.,linear_rate*dt/max(float(np.linalg.norm(dp)),1e-30))
    b=min(1.,np.deg2rad(angular_rate_deg)*dt/max(float(np.linalg.norm(rv)),1e-30))
    return previous_p+a*dp,Rotation.from_rotvec(b*rv).as_matrix()@previous_r,bool(a<1 or b<1)


def clamp_correction(translation,rotation,max_translation,max_rotation_deg):
    """Clamp a Cartesian correction without changing its direction."""
    translation=np.asarray(translation,float)
    rv=Rotation.from_matrix(rotation).as_rotvec()
    a=min(1.,max_translation/max(float(np.linalg.norm(translation)),1e-30))
    b=min(1.,np.deg2rad(max_rotation_deg)/max(float(np.linalg.norm(rv)),1e-30))
    return a*translation,Rotation.from_rotvec(b*rv).as_matrix(),bool(a<1 or b<1)


def load_profile(path=DEFAULT_PROFILE):
    path=Path(path)
    raw=path.read_bytes();p=json.loads(raw)
    if p.get('schema')!='tron2-experimental-real-pose-reference-v1' or p.get('controller_version')!=VERSION:
        raise ValueError('Expected explicit experimental real pose profile; simulation profiles are not deployment profiles')
    for name in ['model','vendor_urdf']:
        source=Path(p[name]['path'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=p[name]['sha256']:
            raise ValueError(f'Pose profile {name} hash mismatch')
    if p.get('hardware_validated') is not False or p.get('timing_hardware_calibrated') is not False:
        raise ValueError('This version requires its explicit unvalidated deployment provenance')
    for key in ['kp','kd']:
        v=np.asarray(p[key],float)
        if v.shape!=(7,) or not np.isfinite(v).all() or np.any(v<=0):raise ValueError('Invalid model gains')
    for key,maximum in [('maximum_total_translation_m',.04),('maximum_total_rotation_deg',5.),
                        ('maximum_correction_rate_m_s',.04),('maximum_correction_rate_deg_s',5.)]:
        value=p.get(key)
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not np.isfinite(value) or not 0<value<=maximum:
            raise ValueError(f'Invalid or widened {key}')
    if not np.isfinite(p.get('handover_model_ramp_s',0)) or p['handover_model_ramp_s']<5:
        raise ValueError('Model handover ramp must be at least 5 seconds')
    p['profile_sha256']=hashlib.sha256(raw).hexdigest();p['profile_path']=str(path.resolve())
    return p


class PoseTrajectory:
    def __init__(self,times,positions,quaternions):
        self.times=np.asarray(times,float);self.positions=np.asarray(positions,float)
        q=np.asarray(quaternions,float)
        n=len(self.times)
        if (n<2 or self.positions.shape!=(n,3) or q.shape!=(n,4) or self.times[0]!=0
                or not np.isfinite(self.times).all() or not np.isfinite(self.positions).all()
                or not np.isfinite(q).all() or np.any(np.diff(self.times)<=0)
                or np.any(np.linalg.norm(q,axis=1)<1e-9)):
            raise ValueError('Invalid pose trajectory')
        self.rotations=Slerp(self.times,Rotation.from_quat(q[:,[1,2,3,0]]))
        self.duration=float(self.times[-1])

    def target(self,t):
        t=float(np.clip(t,0,self.duration))
        return np.array([np.interp(t,self.times,self.positions[:,j]) for j in range(3)]),self.rotations(t).as_matrix()


class PoseReferenceModel:
    def __init__(self,profile,q_start):
        import mujoco
        from tools.simulate_tron2_z_feedback import modified_xml,NAMES,TCP
        from tron2_left_ik import SDK_LOWER,SDK_UPPER
        self.mj=mujoco;self.profile=profile;self.tcp=TCP.copy()
        payload=profile['payload_model']
        xml=modified_xml(Path(profile['model']['path']),payload['mass_kg'],payload['COM_wrist_m'],payload)
        self.model=mujoco.MjModel.from_xml_string(xml)
        self.data=mujoco.MjData(self.model)
        self.ids=np.array([self.model.joint(n).id for n in NAMES])
        self.qa=self.model.jnt_qposadr[self.ids];self.da=self.model.jnt_dofadr[self.ids]
        self.lower=np.maximum(self.model.jnt_range[self.ids,0],SDK_LOWER)
        self.upper=np.minimum(self.model.jnt_range[self.ids,1],SDK_UPPER)
        self.lower[[5,6,12,13]]=np.maximum(self.lower[[5,6,12,13]],np.deg2rad(-43))
        self.upper[[5,6,12,13]]=np.minimum(self.upper[[5,6,12,13]],np.deg2rad(43))
        self.qref=np.asarray(q_start,float).copy();self.check_joints(self.qref)
        self.qissued=self.qref.copy();self.previous=self.qref.copy();self.velocity=np.zeros(7)
        self.body=self.model.body('wrist_roll_L_Link').id;self.base=self.model.body('base_Link').id
        self.jp=np.zeros((3,self.model.nv));self.jr=np.zeros((3,self.model.nv))
        self.kp=np.asarray(profile['kp']);self.kd=np.asarray(profile['kd'])

    def snapshot_state(self):
        return tuple(getattr(self,name).copy() for name in ('qref','qissued','previous','velocity'))

    def restore_state(self,state):
        self.qref,self.qissued,self.previous,self.velocity=(value.copy() for value in state)

    def check_joints(self,q):
        q=np.asarray(q,float)
        if q.shape!=(16,) or not np.isfinite(q).all():raise ValueError('Expected 16 finite joint values')
        if np.any(q<self.lower) or np.any(q>self.upper):raise RuntimeError('Pose model joint limit violation')

    def forward(self,q):
        self.check_joints(q);self.data.qpos[self.qa]=q;self.data.qvel[:]=0
        self.mj.mj_forward(self.model,self.data)
        r=self.data.xmat[self.body].reshape(3,3).copy()
        point=self.data.xpos[self.body]+r@self.tcp
        return point-self.data.xpos[self.base],r,point

    def check_anchor(self,q,p,r):
        actual,rotation,_=self.forward(q)
        distance=float(np.linalg.norm(actual-p));angle=float(np.rad2deg(Rotation.from_matrix(rotation@r.T).magnitude()))
        if distance>.002 or angle>.5:
            raise RuntimeError(f'Pose model / SDK anchor mismatch: {distance*1000:.3f}mm / {angle:.3f}deg')
        return dict(position_mm=distance*1000,rotation_deg=angle)

    def solve(self,p,r,seed):
        q=seed.copy()
        for _ in range(12):
            pi,ri,point=self.forward(q)
            ep=p-pi;er=Rotation.from_matrix(r@ri.T).as_rotvec()
            if np.linalg.norm(ep)<1e-6 and np.linalg.norm(er)<1e-5:break
            self.mj.mj_jac(self.model,self.data,self.jp,self.jr,point,self.body)
            j=np.vstack((self.jp[:,self.da[:7]],.25*self.jr[:,self.da[:7]]))
            dq=j.T@np.linalg.solve(j@j.T+1e-6*np.eye(6),np.r_[ep,.25*er])
            dq*=min(1.,.02/max(np.max(abs(dq)),1e-12))
            q[:7]=np.clip(q[:7]+dq,self.lower[:7]+1e-4,self.upper[:7]-1e-4)
        pi,ri,_=self.forward(q)
        if np.linalg.norm(p-pi)>.001 or Rotation.from_matrix(r@ri.T).magnitude()>np.deg2rad(.5):
            raise PoseIKUnavailable('Pose reference IK residual exceeds 1mm / 0.5deg')
        return q

    @staticmethod
    def _interpolate_pose(p0,r0,p1,r1,fraction):
        p=p0+fraction*(p1-p0)
        rv=Rotation.from_matrix(r1@r0.T).as_rotvec()
        r=Rotation.from_rotvec(fraction*rv).as_matrix()@r0
        return p,r

    def solve_toward(self,p,r,seed,iterations=10):
        """Return the furthest feasible pose toward a request without widening bounds."""
        start_p,start_r,_=self.forward(seed)
        try:
            return np.asarray(p,float),np.asarray(r,float),self.solve(p,r,seed),1.
        except PoseIKUnavailable:
            low,high=0.,1.;best_p,best_r,best_q=start_p,start_r,seed.copy()
            for _ in range(iterations):
                fraction=(low+high)/2
                candidate_p,candidate_r=self._interpolate_pose(
                    start_p,start_r,np.asarray(p,float),np.asarray(r,float),fraction)
                try:
                    candidate_q=self.solve(candidate_p,candidate_r,best_q)
                except PoseIKUnavailable:
                    high=fraction
                else:
                    low=fraction;best_p,best_r,best_q=candidate_p,candidate_r,candidate_q
            return best_p,best_r,best_q,low

    def constrain_issued(self,p,r):
        p,r,q,fraction=self.solve_toward(p,r,self.qissued)
        self.qissued=q
        return p,r,fraction

    def step(self,p,r,dt,weight=1.):
        if not 0<dt<=.1 or not 0<=weight<=1:raise ValueError('Invalid pose model timestep/ramp')
        requested_p=np.asarray(p,float);requested_r=np.asarray(r,float)
        p,r,self.qref,reference_fraction=self.solve_toward(requested_p,requested_r,self.qref)
        raw=(self.qref[:7]-self.previous[:7])/dt
        alpha=-np.expm1(-dt/.03);self.velocity+=alpha*(raw-self.velocity)
        self.previous=self.qref.copy()
        velocity_limited=bool(np.max(abs(self.velocity))>2)
        self.velocity=np.clip(self.velocity,-2.,2.)
        _,_,point=self.forward(self.qref)
        support=self.data.qfrc_bias[self.da[:7]]+self.kd*self.velocity
        delta=support/self.kp
        self.mj.mj_jac(self.model,self.data,self.jp,self.jr,point,self.body)
        raw_shift=self.jp[:,self.da[:7]]@delta
        raw_rv=self.jr[:,self.da[:7]]@delta
        translation_scale=min(1.,self.profile['maximum_total_translation_m']/max(np.linalg.norm(raw_shift),1e-30))
        rotation_scale=min(1.,np.deg2rad(self.profile['maximum_total_rotation_deg'])/max(np.linalg.norm(raw_rv),1e-30))
        model_scale=min(translation_scale,rotation_scale)
        shift=model_scale*raw_shift;rv=model_scale*raw_rv
        issued_p=p+weight*shift;issued_r=Rotation.from_rotvec(weight*rv).as_matrix()@r
        issued_p,issued_r,issued_fraction=self.constrain_issued(issued_p,issued_r)
        applied_shift=issued_p-p
        applied_rv=Rotation.from_matrix(issued_r@r.T).as_rotvec()
        return issued_p,issued_r,dict(model_shift_m=applied_shift.tolist(),
            model_rotation_vector_rad=applied_rv.tolist(),
            raw_model_shift_m=raw_shift.tolist(),raw_model_rotation_vector_rad=raw_rv.tolist(),
            model_shift_limited=bool(model_scale<1.),model_scale=float(model_scale),
            reference_velocity_limited=velocity_limited,
            reference_ik_limited=bool(reference_fraction<1.),reference_fraction=float(reference_fraction),
            issued_ik_limited=bool(issued_fraction<1.),issued_fraction=float(issued_fraction),
            filtered_reference_velocity_rad_s=self.velocity.tolist())


def preflight(trajectory,q_start,profile,preview_s=0.):
    from replay_tron2_tracker import check_workspace
    if not np.isfinite(preview_s) or not 0<=preview_s<=.2:raise ValueError('Preview must be within 0..200ms')
    engine=PoseReferenceModel(profile,q_start)
    p0,r0=trajectory.target(0);anchor=engine.check_anchor(q_start,p0,r0)
    check_workspace('left',trajectory.positions)
    issued=[];rotations=[];cost=[];shifts=[];offset_rotations=[];last=0.;maximum_shift=maximum_angle=0.;limited_count=0
    for i,t in enumerate(trajectory.times):
        dt=float(t-last) if i else 1/300;last=t
        p,r=trajectory.target(t+preview_s)
        started=time.perf_counter();pi,ri,_=engine.step(p,r,dt)
        if i:
            shift,rotation,limited=limit_correction(shifts[-1],offset_rotations[-1],pi-p,ri@r.T,dt,
                profile['maximum_correction_rate_m_s'],profile['maximum_correction_rate_deg_s'])
            if limited:
                pi=p+shift;ri=rotation@r;engine.qissued=engine.solve(pi,ri,engine.qissued);limited_count+=1
        cost.append(time.perf_counter()-started);issued.append(pi);rotations.append(ri)
        shifts.append(pi-p);offset_rotations.append(ri@r.T)
        maximum_shift=max(maximum_shift,float(np.linalg.norm(pi-p)))
        maximum_angle=max(maximum_angle,float(np.rad2deg(Rotation.from_matrix(ri@r.T).magnitude())))
    issued=np.asarray(issued);check_workspace('left',issued)
    rate=float((np.linalg.norm(np.diff(shifts,axis=0),axis=1)/np.diff(trajectory.times)).max())
    rr=Rotation.from_matrix(np.array(offset_rotations))
    angular_rate=float((np.rad2deg((rr[1:]*rr[:-1].inv()).magnitude())/np.diff(trajectory.times)).max())
    if rate>profile['maximum_correction_rate_m_s']+1e-10 or angular_rate>profile['maximum_correction_rate_deg_s']+1e-8:
        raise RuntimeError(f'Model correction rate exceeds deployment bound: {rate*1000:.2f}mm/s / {angular_rate:.2f}deg/s')
    return dict(controller_version=VERSION,hardware_motion_commands=0,trajectory_frames=len(issued),
        duration_s=trajectory.duration,anchor=anchor,profile_sha256=profile['profile_sha256'],
        preview_s=preview_s,hardware_timing_calibrated=False,
        nominal_min_m=trajectory.positions.min(0).tolist(),nominal_max_m=trajectory.positions.max(0).tolist(),
        issued_min_m=issued.min(0).tolist(),issued_max_m=issued.max(0).tolist(),
        maximum_model_shift_mm=maximum_shift*1000,maximum_model_rotation_deg=maximum_angle,
        correction_slew_limited_frames=limited_count,
        maximum_model_translation_correction_rate_mm_s=rate*1000,maximum_model_rotation_correction_rate_deg_s=angular_rate,
        computation_P95_ms=float(np.percentile(cost,95)*1000),computation_maximum_ms=float(max(cost)*1000),
        residual_envelope_fully_proven=False,
        limitations=['Online residuals separately checked for IK/workspace; no contact or collision proof.',
                     'ServoP may use a different joint branch; local IK is only a reference model.'])
