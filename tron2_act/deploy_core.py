"""Hardware-independent observation alignment and absolute ACT target scheduling.

Only physical monotonic nanoseconds are used. There is no robot, ROS or CAN I/O.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .geometry import matrix_to_rotation6d, rotation6d_to_matrix


class PendingObservation(RuntimeError):
    pass


class Timeline:
    def __init__(self, rotations=False, max_gap_s=.15):
        self.rows = deque()
        self.rotations = rotations
        self.max_gap_ns = round(max_gap_s*1e9)

    def add(self, stamp, values):
        value = np.asarray(values,float)
        if not isinstance(stamp,int) or not np.isfinite(value).all():
            raise ValueError('Invalid monotonic sample')
        if self.rows and stamp < self.rows[-1][0]:
            raise ValueError('Feedback clock moved backwards')
        if self.rows and stamp == self.rows[-1][0]:
            if not np.array_equal(value,self.rows[-1][1]):
                raise ValueError('Conflicting feedback at one timestamp')
            return
        self.rows.append((stamp,value.copy()))
        while len(self.rows)>2 and self.rows[1][0] < stamp-4_000_000_000:
            self.rows.popleft()

    def at(self, stamp):
        if len(self.rows)<2 or stamp<self.rows[0][0] or stamp>self.rows[-1][0]:
            raise PendingObservation('Need measured feedback bracketing image time/history')
        times=np.array([x[0] for x in self.rows],np.int64)
        right=int(np.clip(np.searchsorted(times,stamp),1,len(times)-1));left=right-1
        if stamp not in (times[left],times[right]) and times[right]-times[left]>self.max_gap_ns:
            raise RuntimeError('Feedback gap too large for observation interpolation')
        alpha=(stamp-int(times[left]))/(int(times[right])-int(times[left]))
        if self.rotations:
            return Slerp([0.,1.],Rotation.from_matrix([self.rows[left][1],self.rows[right][1]]))(alpha).as_matrix()
        return (1-alpha)*self.rows[left][1]+alpha*self.rows[right][1]


@dataclass
class ReferencePoint:
    """T_E,T: optional fixed tracker proxy in the SDK TCP frame."""
    translation: np.ndarray
    rotation: np.ndarray

    @classmethod
    def identity(cls):
        return cls(np.zeros(3),np.eye(3))

    @classmethod
    def from_matrix(cls,value):
        t=np.asarray(value,float)
        if (t.shape!=(4,4) or not np.isfinite(t).all() or
            not np.allclose(t[3],[0,0,0,1]) or
            not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-6) or
            not np.isclose(np.linalg.det(t[:3,:3]),1,atol=1e-6)):
            raise ValueError('Expected rigid T_ET, not an axis-map/world transform')
        return cls(t[:3,3].copy(),t[:3,:3].copy())

    def tracker_pose(self,p,r):
        return p+r@self.translation,r@self.rotation

    def sdk_targets(self,p,r,actions):
        delta_r=rotation6d_to_matrix(actions[:,3:9])
        future_r=delta_r@r
        # A fixed orientation cancels from spatial relative rotations; the lever
        # arm does not cancel from translations when the wrist turns.
        future_p=p+r@self.translation+actions[:,:3]-future_r@self.translation
        return future_p,future_r


def observation_state(stamp,positions,rotations,hands,reference,history_steps=2,hz=20):
    p,r=positions.at(stamp),rotations.at(stamp)
    current_p,current_r=reference.tracker_pose(p,r)
    old=[]
    for offset in range(history_steps,0,-1):
        at=stamp-round(offset/hz*1e9)
        hp,hr=reference.tracker_pose(positions.at(at),rotations.at(at))
        old.extend(np.r_[hp-current_p,matrix_to_rotation6d(hr@current_r.T)])
    hand=np.asarray(hands.at(stamp),float)
    # Raw O6 samples are validated before entering the timeline. Linear
    # interpolation can nevertheless produce 255 + a few ulps when both
    # bracketing byte-valued samples are 255. Accept only that numerical
    # round-off here and clamp it back to the measured SDK domain.
    epsilon=1e-6
    if (hand.shape!=(6,) or not np.isfinite(hand).all() or
            np.any((hand < -epsilon)|(hand > 255+epsilon))):
        extent='invalid shape/value'
        if hand.shape==(6,) and np.isfinite(hand).all():
            extent=f'range [{hand.min():.9g}, {hand.max():.9g}]'
        raise ValueError(f'O6 q_pos must be six measured SDK feedback values; got {extent}')
    hand=np.clip(hand,0.,255.)
    return np.r_[old,hand].astype(np.float32),p,r


class SpatialChunk:
    def __init__(self,stamp,p,r,actions,reference,hz=20,execution_speed=1.):
        if not np.isfinite(execution_speed) or not .1<=execution_speed<=1.:
            raise ValueError('ACT execution_speed must be within 0.1..1.0')
        effective_hz=hz*execution_speed
        self.stamp=int(stamp);self.step=round(1e9/effective_hz)
        a=np.asarray(actions,float)
        if a.ndim!=2 or a.shape[0]<2 or a.shape[1]!=15 or not np.isfinite(a).all():
            raise ValueError('Expected a finite N x 15 ACT action chunk with N >= 2')
        self.length=a.shape[0]
        self.positions,self.rotations=reference.sdk_targets(np.asarray(p),np.asarray(r),a)
        self.hands=np.clip(a[:,9:], [80,0,0,0,0,0],255)
        self.end=self.stamp+self.length*self.step
        self.indices=np.arange(1,self.length+1)
        self.slerp=Slerp(self.indices/effective_hz,Rotation.from_matrix(self.rotations))

    def target(self,stamp):
        at=float(np.clip((stamp-self.stamp)/self.step,1,self.length))
        p=np.array([np.interp(at,self.indices,self.positions[:,j]) for j in range(3)])
        r=self.slerp(at*self.step/1e9).as_matrix()
        hand=self.hands[min(self.length-1,max(0,int(np.floor(at+1e-8))-1))]
        return p,r,hand


class ChunkPool:
    """Combine overlapping predictions after decoding them into one SDK/base frame."""
    def __init__(self,reference=None,decay=.1,execution_speed=1.,max_result_age_ms=150.):
        if not np.isfinite(decay) or decay<0:
            raise ValueError('Temporal aggregation decay must be nonnegative')
        if not np.isfinite(execution_speed) or not .1<=execution_speed<=1.:
            raise ValueError('ACT execution_speed must be within 0.1..1.0')
        if not np.isfinite(max_result_age_ms) or not 100<=max_result_age_ms<=1000:
            raise ValueError('ACT max_result_age_ms must be within 100..1000')
        self.reference=reference or ReferencePoint.identity()
        self.decay=decay;self.execution_speed=float(execution_speed)
        self.max_result_age_ns=round(max_result_age_ms*1e6)
        self.chunks=deque();self.last_observation_ns=None

    def add(self,stamp,p,r,action,now):
        if stamp>now or now-stamp>self.max_result_age_ns:
            raise RuntimeError(
                f'ACT result is future-dated or older than {self.max_result_age_ns/1e6:g}ms'
            )
        if self.last_observation_ns is not None and stamp<=self.last_observation_ns:
            raise RuntimeError('ACT results arrived out of observation order')
        chunk=SpatialChunk(stamp,p,r,action,self.reference,execution_speed=self.execution_speed)
        self.chunks.append(chunk);self.last_observation_ns=stamp
        # A prediction remains useful until the end of its action horizon.  A
        # fixed, short history window silently turns ACT into a first-few-step
        # controller and prevents later actions (for example, a grasp near the
        # middle of a long chunk) from ever becoming due.
        self.chunks=deque(c for c in self.chunks if c.end>=now)

    def target(self,stamp,hand=False):
        eligible=[c for c in self.chunks if c.stamp+(c.step if hand else 0)<=stamp<=c.end]
        if not eligible:
            raise PendingObservation('No prediction covers this command time')
        # Match ACT temporal ensembling: eligible chunks are ordered from the
        # earliest prediction to the latest, and earlier commitments receive
        # the larger exponential weight.  Favoring the newest prediction here
        # lets every replanning cycle postpone a later action indefinitely.
        weights=np.exp(-self.decay*np.arange(len(eligible)));weights/=weights.sum()
        targets=[c.target(stamp) for c in eligible]
        return (np.average([x[0] for x in targets],axis=0,weights=weights),
                Rotation.from_matrix([x[1] for x in targets]).mean(weights=weights).as_matrix(),
                np.average([x[2] for x in targets],axis=0,weights=weights))


class LiveReference:
    def __init__(self,p,r,pool,max_displacement=.35,max_angle=90.,ramp_s=.5):
        self.p,self.r=np.array(p),np.array(r);self.pool=pool
        self.max_displacement=max_displacement;self.max_angle=max_angle
        self.ramp_s=ramp_s;self.epoch_ns=None;self.frozen=None
        self.last_target=(self.p.copy(),self.r.copy())

    def target(self,seconds):
        if self.frozen is not None:return self.frozen[0].copy(),self.frozen[1].copy()
        if self.epoch_ns is None or seconds<0:return self.p.copy(),self.r.copy()
        try:
            p,r,_=self.pool.target(self.epoch_ns+round(seconds*1e9))
        except PendingObservation:
            # An ACT rollout should not abort because one image/prediction is
            # late.  Keep publishing the last decoded target until a new chunk
            # becomes eligible, as the LimX environment keeps the prior action
            # between policy updates.
            return self.last_target[0].copy(),self.last_target[1].copy()
        if np.linalg.norm(p-self.p)>self.max_displacement or np.rad2deg(Rotation.from_matrix(r@self.r.T).magnitude())>self.max_angle:
            raise RuntimeError('ACT target exceeds startup displacement/rotation envelope')
        weight=min(1.,max(0.,seconds/self.ramp_s)) if self.ramp_s else 1.
        weight=weight*weight*(3-2*weight)
        target=(self.p+weight*(p-self.p),
                Rotation.from_rotvec(weight*Rotation.from_matrix(r@self.r.T).as_rotvec()).as_matrix()@self.r)
        self.last_target=(target[0].copy(),target[1].copy())
        return target
