"""只读保存 SDK 的电机目标及状态；不初始化机器人，也不发布控制命令。

q 的含义取决于 motor_names 和 parallel_solve_required；本模块只保存原值，
不把电机坐标自动解释成 URDF 关节坐标。缺少目标回调也不代表没有控制目标。
"""

import math
import threading
import time

import numpy as np


FIELDS = ('stamp', 'mode', 'q', 'dq', 'tau', 'Kp', 'Kd',
          'motor_names', 'parallel_solve_required')


def plain_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (np.ndarray, np.generic)):
        return plain_value(value.tolist())
    if isinstance(value, (tuple, list)):
        return [plain_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain_value(item) for key, item in value.items()}
    return repr(value)


class MotorObserver:
    def __init__(self, trace, rate_hz=50.):
        if not math.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError('观测记录频率须为有限正数')
        self.trace = trace
        self.period_ns = int(1e9/rate_hz)
        self.lock = threading.Lock()
        self.active = True
        self.streams = {name: {'received': 0, 'recorded': 0, 'errors': 0,
                               'last_record_ns': None, 'last_fields': None}
                        for name in ('command', 'state')}
        self.subscriptions = {}
        # 保留绑定方法引用，避免 native 回调生命周期短于诊断。
        self.callbacks = {name: self._callback(name) for name in self.streams}

    def _callback(self, name):
        def observe(message):
            # 所有异常均留在 Python 层，不越过 native 回调边界。
            try:
                now = time.monotonic_ns()
                with self.lock:
                    if not self.active:
                        return
                    state = self.streams[name]
                    state['received'] += 1
                    if state['last_record_ns'] is not None and now-state['last_record_ns'] < self.period_ns:
                        return
                    state['last_record_ns'] = now
                    fields = {field: plain_value(getattr(message, field))
                              for field in FIELDS if hasattr(message, field)}
                    self.trace.record('motor_'+name+'_observed',
                                      received_monotonic_ns=now, fields=fields)
                    state['recorded'] += 1
                    state['last_fields'] = fields
            except Exception as exc:
                with self.lock:
                    self.streams[name]['errors'] += 1
                try:
                    self.trace.record('motor_observer_error', stream=name, message=str(exc))
                except Exception:
                    pass
        return observe

    def start(self, robot):
        for name, method in (('command', 'subscribeRobotCmd'), ('state', 'subscribeRobotState')):
            try:
                result = getattr(robot, method)(self.callbacks[name])
                self.subscriptions[name] = {'returned': plain_value(result)}
            except Exception as exc:
                self.subscriptions[name] = {'error': str(exc)}
        self.trace.record('motor_observer_subscriptions', subscriptions=self.subscriptions)

    def stop(self):
        # SDK 不提供对应的取消订阅；停止写记录，不改变任何机器人状态。
        with self.lock:
            self.active = False

    def summary(self):
        with self.lock:
            return {'subscriptions': self.subscriptions.copy(),
                    'streams': {key: value.copy() for key, value in self.streams.items()},
                    'interpretation': 'raw SDK motor fields; coordinate domain not assumed',
                    'control_commands_published': 0}
