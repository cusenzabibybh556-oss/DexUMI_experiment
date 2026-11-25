"""独立 ServoP 诊断使用的纯算法：有界 Z 位置修正。

不连接机器人，不读写文件，不发送力矩。nominal_z 是反馈采样时刻对应的
未补偿物理参考，不能把本模块的修正量再次加入该参数。输出为向上的位置偏置，
不是实测重力前馈。observe 与 advance 须由同一线程顺序调用。
调用者仍负责反馈超时、IK、工作空间与碰撞检查。本版有限速，无加速度/加加速度限幅。
"""

from dataclasses import dataclass, fields
import math
from numbers import Integral, Real


@dataclass(frozen=True)
class Config:
    gain_per_s: float = .5
    max_offset_m: float = .02
    max_rate_m_s: float = .001
    deadband_m: float = .0003
    feedback_max_age_s: float = .15
    feedback_max_gap_s: float = .15
    max_dt_s: float = .1
    max_rtt_s: float = .1

    def __post_init__(self):
        nonnegative = {'gain_per_s', 'max_offset_m', 'deadband_m'}
        for field in fields(self):
            value = getattr(self, field.name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value)):
                raise ValueError(f'{field.name} must be finite and numeric')
            if value < 0 or (field.name not in nonnegative and value == 0):
                raise ValueError(f'{field.name} has an invalid bound')


class ZCompensationController:
    """仅修正位置，不积累无法发出的隐藏积分目标。

    observe 对新且有效的反馈返回 True；首帧/恢复帧仅建立时间基线，不改变目标。
    拒用或禁用时返回 False。advance 返回加入名义目标的偏置，首次返回零。
    时间参数均为主机单调时钟的整数纳秒，可相等、不可倒退；帧龄按接收时刻计算，
    RTT 为接收减发送时间，采样间隔按查询中点计算，不假设机器人时钟与主机同步。
    GUID/序号在本次会话唯一，已处理 ID 保留以阻止重复积分。
    """

    def __init__(self, config=None, *, active_limit_m=None):
        self.config = Config() if config is None else config
        if not isinstance(self.config, Config):
            raise TypeError('config must be Config')
        self._applied = 0.
        self._target = 0.
        self._active_limit = min(.005, self.config.max_offset_m)
        self._status = 'initial'
        self._last_now_ns = None
        self._last_advance_ns = None
        self._last_valid_received_ns = None
        self._sample_midpoint_ns = None
        self._last_sent_ns = None
        self._last_received_ns = None
        self._last_sequence = None
        self._seen_ids = set()
        if active_limit_m is not None:
            self.set_active_limit(active_limit_m)

    @property
    def applied_offset(self):
        return self._applied

    @property
    def target_offset(self):
        return self._target

    @property
    def status(self):
        return self._status

    @property
    def active_limit(self):
        return self._active_limit

    def set_active_limit(self, limit_m):
        """改变运行上限，不扩大配置中的硬边界。

        默认上限为 min(5mm, 硬上限)。扩限不会改变当前输出或待发目标；
        拒绝把上限降到已输出偏置之下。是否允许扩限由调用者完成实测验收后决定。
        """
        if (isinstance(limit_m, bool) or not isinstance(limit_m, Real)
                or not math.isfinite(limit_m)
                or not 0. <= limit_m <= self.config.max_offset_m
                or limit_m < self._applied):
            raise ValueError('active limit must contain applied offset and stay within hard bound')
        self._active_limit = float(limit_m)
        self._target = min(self._target, self._active_limit)

    def freeze(self, now_ns=None):
        """取消待发修正，保留已输出偏置。

        可选核对/推进主机时钟。恢复后首个有效新帧只建立采样基线，
        下一帧才允许修正；本操作不伪造反馈新鲜度或采样 ID。
        """
        if now_ns is not None:
            self._clock(now_ns)
        self._freeze('frozen')

    def _freeze(self, status):
        self._target = self._applied
        self._sample_midpoint_ns = None
        self._status = status

    def _fail(self, message):
        self._freeze('invalid')
        raise ValueError(message)

    def _time(self, value, label):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            self._fail(f'{label} must be nonnegative integer nanoseconds')
        return int(value)

    def _clock(self, now_ns):
        now_ns = self._time(now_ns, 'now_ns')
        if self._last_now_ns is not None and now_ns < self._last_now_ns:
            self._fail('host monotonic time moved backward')
        self._last_now_ns = now_ns
        self._expire(now_ns)
        return now_ns

    def _expire(self, now_ns):
        if (self._last_valid_received_ns is not None
                and now_ns-self._last_valid_received_ns > self.config.feedback_max_age_s*1e9):
            self._freeze('stale')

    def observe(self, sample_id, sent_ns, received_ns, now_ns,
                nominal_z, actual_z, enabled=True):
        now_ns = self._clock(now_ns)
        sent_ns = self._time(sent_ns, 'sent_ns')
        received_ns = self._time(received_ns, 'received_ns')
        if not sent_ns <= received_ns <= now_ns:
            self._fail('expected sent_ns <= received_ns <= now_ns')
        if (isinstance(sample_id, bool)
                or not isinstance(sample_id, (str, Integral))
                or isinstance(sample_id, str) and not sample_id
                or isinstance(sample_id, Integral) and sample_id < 0):
            self._fail('sample_id must be a nonempty GUID or nonnegative sequence')
        if not isinstance(enabled, bool):
            self._fail('enabled must be bool')
        for value in (nominal_z, actual_z):
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value)):
                self._fail('Z positions must be finite numbers')
        error = float(nominal_z)-float(actual_z)
        if not math.isfinite(error):
            self._fail('Z error overflowed')
        if not enabled:
            self._seen_ids.add(sample_id)
            self._freeze('disabled')
            return False
        if sample_id in self._seen_ids:
            if self._status != 'stale':
                self._status = 'duplicate'
            return False
        self._seen_ids.add(sample_id)
        if now_ns-received_ns > self.config.feedback_max_age_s*1e9:
            self._freeze('rejected_age')
            return False
        if received_ns-sent_ns > self.config.max_rtt_s*1e9:
            self._freeze('rejected_rtt')
            return False
        if (self._last_sent_ns is not None and sent_ns <= self._last_sent_ns
                or self._last_received_ns is not None and received_ns <= self._last_received_ns
                or isinstance(sample_id, Integral) and self._last_sequence is not None
                and sample_id <= self._last_sequence):
            self._freeze('rejected_order')
            return False
        previous_midpoint = self._sample_midpoint_ns
        midpoint = (sent_ns+received_ns)//2
        self._last_sent_ns = sent_ns
        self._last_received_ns = received_ns
        if isinstance(sample_id, Integral):
            self._last_sequence = int(sample_id)
        self._last_valid_received_ns = received_ns
        self._sample_midpoint_ns = midpoint
        if previous_midpoint is None:
            self._target = self._applied
            self._status = 'primed'
            return True
        gap_ns = midpoint-previous_midpoint
        if gap_ns > self.config.feedback_max_gap_s*1e9:
            self._target = self._applied
            self._status = 'gap'
            return True
        dt = min(gap_ns*1e-9, self.config.max_dt_s)
        deadband_error = math.copysign(max(abs(error)-self.config.deadband_m, 0.), error)
        # 先裁剪速度，避免极大但有限的输入乘法溢出；没有独立的无界积分状态。
        error_rate = self.config.gain_per_s*deadband_error
        rate = max(-self.config.max_rate_m_s, min(self.config.max_rate_m_s, error_rate))
        wanted = self._applied+rate*dt
        self._target = max(0., min(self._active_limit, wanted))
        if wanted >= self._active_limit and rate > 0:
            self._status = 'saturated_high'
        elif wanted <= 0 and rate < 0:
            self._status = 'saturated_low'
        elif deadband_error == 0 or self.config.gain_per_s == 0:
            self._status = 'deadband'
        else:
            self._status = 'tracking'
        return True

    def advance(self, now_ns):
        now_ns = self._clock(now_ns)
        previous = self._last_advance_ns
        self._last_advance_ns = now_ns
        if previous is None:
            return self._applied
        maximum_step = self.config.max_rate_m_s*((now_ns-previous)*1e-9)
        difference = self._target-self._applied
        self._applied += max(-maximum_step, min(maximum_step, difference))
        self._applied = max(0., min(self._active_limit, self._applied))
        return self._applied
