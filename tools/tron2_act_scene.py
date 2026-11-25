"""Isaac scene corrections and diagnostic helpers. Import after SimulationApp."""
import numpy as np
from scipy.spatial.transform import Rotation


def rotate_hand_mount(stage, degrees):
    """Rotate the instance's O6 mounting joint and all authored hand link poses.

    Imported O6 has palm normal X and finger length Z. A -90deg mounting
    rotation makes the palm face down at the existing neutral wrist start.
    The source USD files are never modified.
    """
    from pxr import Gf, UsdGeom, UsdPhysics
    from isaacsim.core.utils.xforms import get_world_pose
    path = '/World/Robot'
    root = stage.GetPrimAtPath(path)
    hand_p,hand_q = get_world_pose(path+'/o6_L_hand_base_link')
    hand_r = Rotation.from_quat(np.asarray(hand_q)[[1,2,3,0]]).as_matrix()
    local = Rotation.from_euler('z',degrees,degrees=True).as_matrix()
    delta = hand_r@local@hand_r.T
    parent = UsdGeom.XformCache().GetLocalToWorldTransform(root)
    parent_inv = parent.GetInverse()
    for prim in root.GetChildren():
        if not prim.GetName().startswith('o6_L_') or not prim.IsA(UsdGeom.Xform):
            continue
        p,q = get_world_pose(str(prim.GetPath()))
        new_p = np.asarray(hand_p)+delta@(np.asarray(p)-hand_p)
        new_r = delta@Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
        quat = Rotation.from_matrix(new_r).as_quat()
        matrix = Gf.Matrix4d(1.)
        matrix.SetRotate(Gf.Quatd(float(quat[3]),Gf.Vec3d(*quat[:3])))
        matrix.SetTranslateOnly(Gf.Vec3d(*new_p))
        x = UsdGeom.Xformable(prim)
        x.ClearXformOpOrder()
        x.AddTransformOp().Set(matrix*parent_inv)
    joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(path+'/joints/o6_L_mount_Joint'))
    old = joint.GetLocalRot0Attr().Get()
    q = Rotation.from_euler('z',degrees,degrees=True).as_quat()
    dq = Gf.Quatf(float(q[3]),Gf.Vec3f(*q[:3]))
    joint.CreateLocalRot0Attr().Set(old*dq)


def camera_rotation(position,target):
    """USD camera: -Z forward, +Y up; image right approximately world -Y."""
    z = np.asarray(position)-np.asarray(target)
    z = z/np.linalg.norm(z)
    x = np.array([0.,-1.,0.])
    x -= z*np.dot(x,z);x /= np.linalg.norm(x)
    y = np.cross(z,x)
    q = Rotation.from_matrix(np.column_stack([x,y,z])).as_quat()
    return q[[3,0,1,2]]
