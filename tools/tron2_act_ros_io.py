#!/usr/bin/env python3
"""ROS I/O sidecar. No CAN ownership; o6_bridge remains the only device owner.

Camera compressed bytes and measured O6 feedback go to stdout as JSON lines.
stdin accepts acquire/target/stop. Merely starting this sidecar sends no targets.
"""
import argparse
import base64
import json
import os
import queue
import sys
import threading
import time

OUTPUT_LOCK=threading.Lock()


def emit(event,**fields):
    with OUTPUT_LOCK:
        print(json.dumps({'event':event,'received_ns':time.monotonic_ns(),**fields}),flush=True)


def stamp_ns(message):
    return int(message.header.stamp.sec)*1_000_000_000+int(message.header.stamp.nanosec)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-topic',default='/camera/top/color/image_raw/compressed')
    parser.add_argument('--info-topic',default='/camera/top/color/camera_info')
    parser.add_argument('--expected-can',default='can0')
    parser.add_argument('--camera-fd',required=True,type=int)
    args=parser.parse_args()
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState,CompressedImage,CameraInfo
    from std_srvs.srv import SetBool
    from rcl_interfaces.srv import GetParameters
    camera_output=os.fdopen(args.camera_fd,'w',encoding='utf-8',buffering=1,closefd=True)
    camera_outgoing=queue.Queue(maxsize=1)
    rclpy.init();node=Node('tron2_act_live_io')
    service=node.create_client(SetBool,'/robot/o6/left/replay_mode')
    publisher=node.create_publisher(JointState,'/robot/o6/left/replay_cmd',1)
    incoming=queue.Queue(maxsize=4);latest={'info':None,'sequence':0,'applied_sequence':None}
    acquired=False;last_target_ns=None
    def read_stdin():
        try:
            for line in sys.stdin:
                command=json.loads(line)
                incoming.put(command,timeout=.1)
        except BaseException as exc:
            emit('error',error='stdin: '+str(exc))
        finally:
            try:incoming.put({'command':'stop'},timeout=.1)
            except queue.Full:pass
    def set_mode(enabled):
        if not service.wait_for_service(timeout_sec=3.):raise RuntimeError('O6 replay_mode service unavailable')
        request=SetBool.Request();request.data=enabled
        future=service.call_async(request);rclpy.spin_until_future_complete(node,future,timeout_sec=3.)
        if not future.done() or future.result() is None or not future.result().success:
            raise RuntimeError('O6 replay ownership request failed')
    def on_info(msg):
        latest['info']={'width':msg.width,'height':msg.height,'k':list(msg.k),'d':list(msg.d),'distortion_model':msg.distortion_model}
    def on_camera(msg):
        if latest['info'] is None:return
        source=stamp_ns(msg)
        if source<=latest.get('camera_source_ns',0):return
        latest['camera_source_ns']=source;latest['sequence']+=1
        item=dict(sequence=latest['sequence'],source_stamp_ns=source,wall_received_ns=time.time_ns(),
                  jpeg=bytes(msg.data),info=latest['info'])
        try:camera_outgoing.put_nowait(item)
        except queue.Full:
            try:camera_outgoing.get_nowait()
            except queue.Empty:pass
            try:camera_outgoing.put_nowait(item)
            except queue.Full:pass
    def on_state(msg):
        source=stamp_ns(msg)
        emit('hand_state',source_stamp_ns=source,wall_received_ns=time.time_ns(),
             names=list(msg.name),q_pos=list(msg.position),source='measured_o6_joint_states')
    def on_applied(msg):
        expected=latest.get('expected_values')
        matched=(expected is not None and len(msg.position)==6 and all(abs(float(a)-float(b))<=.5 for a,b in zip(msg.position,expected)))
        if msg.header.frame_id=='replay' and matched:
            emit('hand_applied',source_stamp_ns=stamp_ns(msg),values=list(msg.position),
                 sequence=latest['applied_sequence'])
    node.create_subscription(CameraInfo,args.info_topic,on_info,qos_profile_sensor_data)
    node.create_subscription(CompressedImage,args.image_topic,on_camera,qos_profile_sensor_data)
    node.create_subscription(JointState,'/robot/o6/left/joint_states',on_state,10)
    node.create_subscription(JointState,'/robot/o6/left/action_applied',on_applied,10)
    def write_camera():
        while True:
            item=camera_outgoing.get()
            if item is None:return
            item['jpeg']=base64.b64encode(item['jpeg']).decode('ascii')
            print(json.dumps({'event':'camera','received_ns':time.monotonic_ns(),**item}),
                  file=camera_output,flush=True)
    camera_thread=threading.Thread(target=write_camera,daemon=True);camera_thread.start()
    try:
        client=node.create_client(GetParameters,'/o6_bridge_node/get_parameters')
        if not client.wait_for_service(timeout_sec=5.):raise RuntimeError('O6 bridge parameters unavailable')
        request=GetParameters.Request();request.names=['dry_run','hand_type','can']
        future=client.call_async(request);rclpy.spin_until_future_complete(node,future,timeout_sec=5.)
        if not future.done() or future.result() is None:raise RuntimeError('O6 identity query timed out')
        values=future.result().values
        if len(values)!=3 or values[0].type!=1 or values[0].bool_value or values[1].string_value!='left' or values[2].string_value!=args.expected_can:
            raise RuntimeError('Expected real left O6 bridge on '+args.expected_can+'; dry_run feedback is rejected')
        emit('ready',dry_run=False,hand_type='left',can=args.expected_can)
        threading.Thread(target=read_stdin,daemon=True).start()
        while rclpy.ok():
            rclpy.spin_once(node,timeout_sec=.001)
            try:command=incoming.get_nowait()
            except queue.Empty:command=None
            if command:
                kind=command.get('command')
                if kind=='stop':break
                if kind=='acquire':
                    if acquired:raise RuntimeError('O6 already acquired')
                    set_mode(True);acquired=True;emit('acquired')
                elif kind=='target':
                    if not acquired:raise RuntimeError('O6 target without exclusive ownership')
                    values=command['values'];due=int(command['due_ns']);now=time.monotonic_ns()
                    if len(values)!=6 or due>now+5_000_000:
                        raise RuntimeError('Invalid O6 action')
                    if now-due>100_000_000:
                        emit('hand_target_dropped',sequence=command['sequence'],reason='expired',age_ms=(now-due)/1e6)
                        continue
                    import math
                    if any(not math.isfinite(float(v)) or not lo<=v<=255 for v,lo in zip(values,[80,0,0,0,0,0])):
                        raise RuntimeError('O6 command outside bridge limits')
                    msg=JointState();msg.name=['thumb_flexion','thumb_abduction','index_flexion','middle_flexion','ring_flexion','pinky_flexion']
                    msg.position=[float(v) for v in values]
                    wall=time.time_ns();msg.header.stamp.sec=wall//1_000_000_000;msg.header.stamp.nanosec=wall%1_000_000_000
                    latest['applied_sequence']=command['sequence'];latest['expected_values']=values;publisher.publish(msg);last_target_ns=now
                    emit('hand_target',sequence=command['sequence'],values=values,due_ns=due)
                else:raise ValueError('Unknown stdin command')
    except BaseException as exc:
        emit('error',error=str(exc));return 2
    finally:
        if acquired:
            try:set_mode(False);emit('released')
            except BaseException as exc:emit('error',error='O6 release: '+str(exc))
        try:camera_outgoing.put_nowait(None)
        except queue.Full:pass
        node.destroy_node();rclpy.shutdown();camera_output.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
