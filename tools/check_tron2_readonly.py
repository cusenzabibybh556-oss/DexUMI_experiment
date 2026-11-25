#!/usr/bin/env python3
"""TRON2 开机只读检查；不切网络/模式，不发布运动、参数或故障清除指令。"""

import argparse
import array
from datetime import datetime
import ipaddress
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from replay_tron2_tracker import TraceRecorder
from tools.tron2_motor_observer import MotorObserver, plain_value
from tools.tron2_raw_ws_client import RawWSClient


def direct_network(robot_ip):
    """在构造原生SDK前检查直连，避免失败连接的已知析构问题。"""
    address = ipaddress.ip_address(robot_ip)
    route = json.loads(subprocess.check_output(
        ['ip', '-j', 'route', 'get', robot_ip], text=True, timeout=3))
    interfaces = json.loads(subprocess.check_output(['ip', '-j', 'address'], text=True, timeout=3))
    selected = route[0] if len(route) == 1 else {}
    dev = selected.get('dev')
    networks = [ipaddress.ip_interface(f"{a['local']}/{a['prefixlen']}").network
                for iface in interfaces if iface.get('ifname') == dev
                for a in iface.get('addr_info', []) if a.get('family') == 'inet']
    if selected.get('gateway') or not any(address in net for net in networks):
        raise RuntimeError(f'机器人没有同子网直连路由：{selected}；未初始化SDK，不自动切网')
    return {'route': route, 'direct_interface': dev}


READ_QUERIES = (
    ('request_drag_teach_manage', {'cmd': 'status', 'arg': ''}),
    ('request_get_payload_param', {}),
    ('request_get_compliant_enable', {}),
    ('request_get_compliant_param', {}),
    ('request_get_joint_state', {}),
    ('request_get_move_pose', {}),
    ('request_get_servo_pose', {}),
)


def run(args):
    out = args.output or ROOT/'validation_results'/('tron2_boot_readonly_'+datetime.now(
        ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    report = {'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'motion_commands_sent': 0, 'parameter_writes': 0,
              'mode_changes': 0, 'fault_clear_requests': 0, 'requests': []}
    client = trace = observer = robot = diagnostic = None
    raw_subscriptions = []
    raw_topics = {}
    diagnostics = []
    diagnostic_lock = threading.Lock()
    try:
        report['network'] = direct_network(args.robot_ip)
        trace = TraceRecorder(out/'observation.jsonl')
        client = RawWSClient(args.robot_ip, trace, connect_timeout=3)
        client.wait_for_identity(timeout=3)
        if client.accid != 'DACH_TRON2A_253':
            raise RuntimeError('机器人身份不符，停止查询')
        report['robot_id'] = client.accid
        with client._condition:
            report['identity_notifications'] = [m.copy() for m in client._messages
                                                 if m.get('title') == 'notify_robot_info']
        import limxsdk.robot.Robot as Robot
        import limxsdk.robot.RobotType as RobotType
        robot = Robot(RobotType.Tron2)
        if not robot.init(args.robot_ip):
            raise RuntimeError('原生SDK通信初始化失败；未发布任何控制命令')
        observer = MotorObserver(trace, rate_hz=25)
        observer.start(robot)

        def diagnostic(message):
            try:
                fields = {key: plain_value(getattr(message, key))
                          for key in ('stamp', 'level', 'name', 'code', 'message', 'frame_id')
                          if hasattr(message, key)}
                with diagnostic_lock:
                    diagnostics.append(fields)
                trace.record('diagnostic_observed', fields=fields)
            except Exception as exc:
                trace.record('diagnostic_callback_error', message=str(exc))

        report['diagnostic_subscription'] = plain_value(robot.subscribeDiagnosticValue(diagnostic))
        for title, data in READ_QUERIES:
            item = {'title': title, 'data': data}
            try:
                item['reply'] = client.request(title, data, title.replace('request_', 'response_', 1), timeout=2)
            except Exception as exc:
                item['error'] = str(exc)
            report['requests'].append(item)
        # 仅作主题发现；不知道类型/语义的主题不会猜测订阅或写入。
        try:
            report['discovered_topics'] = plain_value(robot.get_topics(duration=2))
        except Exception as exc:
            report['topic_discovery_error'] = str(exc)
        if args.extended:
            try:
                report['published_topics'] = plain_value(robot.get_published_topics(duration=2))
            except Exception as exc:
                report['publisher_discovery_error'] = str(exc)
            try:
                description = robot.getRobotDescription(timeout_ms=1000)
                if isinstance(description, str) and '<robot' in description:
                    (out/'robot_description.urdf').write_text(description)
                    report['robot_description'] = 'robot_description.urdf'
                else:
                    report['robot_description'] = '接口未返回URDF，不替换为本机模型冒充真机文件'
            except Exception as exc:
                report['robot_description_error'] = str(exc)
            from limxsdk.msg.controller_msgs import JointCmd, JointState
            from limxsdk.msg.std_msgs import Float32MultiArray, UInt16Array

            def raw_plain(value):
                if isinstance(value, array.array):
                    return value.tolist()
                if hasattr(value, '__slots__'):
                    return {k: raw_plain(getattr(value, k)) for k in value.__slots__}
                if isinstance(value, (list, tuple)):
                    return [raw_plain(v) for v in value]
                return plain_value(value)

            def raw_callback(topic):
                def receive(message):
                    try:
                        now = time.monotonic_ns()
                        state = raw_topics[topic]
                        state['received'] += 1
                        if now-state['last_record_ns'] < 100_000_000:
                            return
                        state['last_record_ns'] = now
                        state['last'] = raw_plain(message)
                        trace.record('raw_topic_observed', topic=topic, fields=state['last'])
                    except Exception as exc:
                        trace.record('raw_topic_decode_error', topic=topic, error=str(exc))
                return receive

            for cls, topic in (
                (JointCmd, '/motor/cmd'), (JointCmd, '/motor/cmd/raw'),
                (JointCmd, '/motor/cmd/calibration'),
                (JointState, '/motor/state'), (JointState, '/motor/state/raw'),
                (UInt16Array, '/device_status'), (Float32MultiArray, '/arm_pose'),
                (Float32MultiArray, '/arm_pose_des'),
                (Float32MultiArray, '/manipulation_contact/payload_grav_comp'),
                (Float32MultiArray, '/manipulation_contact/friction_comp'),
            ):
                raw_topics[topic] = {'received': 0, 'last_record_ns': 0, 'last': None}
                try:
                    raw_subscriptions.append(robot.subscribe(cls, topic, raw_callback(topic)))
                except Exception as exc:
                    raw_topics[topic]['error'] = str(exc)
        time.sleep(args.seconds)
        for title in ('request_get_joint_state', 'request_get_servo_pose'):
            try:
                report['requests'].append({'title': title, 'label': 'final_readonly',
                    'reply': client.request(title, {}, title.replace('request_', 'response_', 1), timeout=2)})
            except Exception as exc:
                report['requests'].append({'title': title, 'label': 'final_readonly', 'error': str(exc)})
        report['result'] = 'readback_completed' if all('error' not in r for r in report['requests']) else 'partial_readback'
    except Exception as exc:
        report.update(result='unavailable', error=str(exc))
    finally:
        for sub in raw_subscriptions:
            try:
                sub.close()
            except Exception as exc:
                report.setdefault('subscription_close_errors', []).append(str(exc))
        report['raw_topics'] = raw_topics
        if observer is not None:
            observer.stop()
            report['motor_observer'] = observer.summary()
        with diagnostic_lock:
            report['diagnostics'] = list(diagnostics)
        report['diagnostic_warning_or_error_frames'] = [d for d in report['diagnostics']
                                                        if isinstance(d.get('level'), (int, float)) and d['level'] >= 1]
        report['limitations'] = [
            '开机只读采样，不验证运动跟踪、碰撞、负载真实性或ServoP分支。',
            '没有诊断帧不等于故障码为0；只记录实际收到的消息。',
            'Cmd.q不是已经核实的驱动器最终内部参考；SDK未公开的电流、温度或限幅不能推测。',
            'Idle/阻尼/开机状态下的目标与实际差异不能直接当作ServoP跟踪误差。']
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                report['close_error'] = str(exc)
        if trace is not None:
            trace.close()
        (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({'output': str(out), 'result': report.get('result'),
                          'error': report.get('error'), 'diagnostic_frames': len(diagnostics),
                          'warning_error_frames': len(report['diagnostic_warning_or_error_frames'])},
                         ensure_ascii=False), flush=True)
    # 避免已知RobotNative进程析构崩溃；日志及只读连接已先关闭。
    return report, robot, observer, diagnostic


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-ip', default='10.192.1.2')
    parser.add_argument('--seconds', type=float, default=5.)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--extended', action='store_true', help='额外只读获取真机URDF及原始电机/状态主题')
    args = parser.parse_args()
    if not 2 <= args.seconds <= 30:
        parser.error('--seconds 须在2～30之间')
    # 保持原生对象及回调引用存活至os._exit。
    _result = run(args)
    os._exit(0 if _result[0].get('result') == 'readback_completed' else 1)
