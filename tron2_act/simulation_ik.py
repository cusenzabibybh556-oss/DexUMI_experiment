"""Simulation inner-plant IK with joint bounds in the optimization itself.

The hardware ServoP path is unchanged. This replaces only the simulated plant's
12-iteration local IK, which can get stuck by clipping redundant wrist joints.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from tron2_pose_reference import PoseReferenceModel


class BoundedPoseReferenceModel(PoseReferenceModel):
    def solve(self,p,r,seed):
        q = np.array(seed,float)
        p0,r0,_ = self.forward(q)
        if np.linalg.norm(p-p0)<1e-6 and Rotation.from_matrix(r@r0.T).magnitude()<1e-5:
            return q
        # Bound each solve's joint step as well as the original SDK/model limits.
        lower = np.maximum(self.lower[:7]+1e-4,q[:7]-.02)
        upper = np.minimum(self.upper[:7]-1e-4,q[:7]+.02)
        initial = np.clip(q[:7],lower+1e-10,upper-1e-10)
        def residual(x):
            candidate = q.copy();candidate[:7] = x
            pp,rr,_ = self.forward(candidate)
            return np.r_[pp-p,.25*Rotation.from_matrix(rr@r.T).as_rotvec(),1e-5*(x-initial)]
        fit = least_squares(residual,initial,bounds=(lower,upper),max_nfev=35,
                            ftol=1e-9,xtol=1e-9,gtol=1e-9)
        q[:7] = fit.x
        pp,rr,_ = self.forward(q)
        if np.linalg.norm(pp-p)>.001 or Rotation.from_matrix(rr@r.T).magnitude()>np.deg2rad(.5):
            raise RuntimeError('Bounded simulator IK residual exceeds 1mm / 0.5deg')
        return q
