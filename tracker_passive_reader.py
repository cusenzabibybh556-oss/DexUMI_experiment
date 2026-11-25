"""Linux hidraw 输入报告的只读副本；不初始化 Dongle、不发配对/建图/控制报文。

现有 stream_poses 继续负责连接。这里只用 O_RDONLY + read，关闭自己的句柄
不会停止其它读者。二进制布局和 WZYX→WXYZ 顺序沿用当前 pyvut 解码。
"""
import os
import select
import struct
import threading
import time
import numpy as np


def decode_report(report, mac, tracker_index=0):
    if len(report) < 12 or report[0] != 0x28:
        return None
    _, sequence, address, kind, size = struct.unpack('<BH6sHB', report[:12])
    identity = ':'.join(f'{v:02x}' for v in address)
    if identity != mac.lower() or kind != 0x110 or size not in (37, 39) or len(report) < 12+size:
        return None
    data = report[12:12+size]
    p = struct.unpack_from('<3f', data, 2)
    raw_q = struct.unpack_from('<4e', data, 14)
    q = [raw_q[0], raw_q[3], raw_q[2], raw_q[1]]
    if not np.isfinite(p).all() or not np.isfinite(q).all() or not .9 <= np.linalg.norm(q) <= 1.1:
        return None
    return {'position_m': list(p), 'quaternion_wxyz': q, 'mac': identity,
            'tracker_index': tracker_index, 'tracking_status': data[36],
            'packet_sequence': sequence, 'raw_pose_hex': data.hex(),
            'received_monotonic_ns': time.monotonic_ns(),
            'parser_host_timestamp_ms': int(time.time()*1000)}


class PassiveTrackerReader:
    def __init__(self, path, mac, tracker_index):
        self.path, self.mac, self.tracker_index = path, mac.lower(), tracker_index
        self.stop_event = threading.Event()
        self.error = ''
        self.fd = None
        self.thread = None

    def start(self, callback):
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        self.thread = threading.Thread(target=self._run, args=(callback,), daemon=True)
        self.thread.start()

    def _run(self, callback):
        last = None
        try:
            while not self.stop_event.is_set():
                if not select.select([self.fd], [], [], .1)[0]:
                    continue
                try:
                    raw = os.read(self.fd, 1024)
                except BlockingIOError:
                    continue
                if not raw:
                    raise OSError('Tracker HID 输入已断开')
                frame = decode_report(raw, self.mac, self.tracker_index)
                if frame is not None and frame['packet_sequence'] != last:
                    last = frame['packet_sequence']
                    callback(frame)
        except Exception as exc:
            self.error = str(exc)

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
