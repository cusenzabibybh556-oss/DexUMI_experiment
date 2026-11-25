"""Real TRON2 I/O adapter. Instantiation is explicit; imports never connect."""
from collections import deque
import json
import os
import queue
import shlex
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .deploy_core import Timeline,PendingObservation,observation_state

ROOT=Path(__file__).resolve().parents[1]


def ros_command(script,arguments):
    setup=ROOT/'linkerhand_telop_python-master/ros2/install/setup.bash'
    command='source /opt/ros/humble/setup.bash && source '+shlex.quote(str(setup))+' && exec '+shlex.join(['/usr/bin/python3',str(script),*arguments])
    return ['bash','-lc',command]


class ROSInputs:
    def __init__(self,config,trace):
        self.config,self.trace=config,trace
        self.camera=None;self.hand_events=queue.Queue(maxsize=256)
        self.ready=threading.Event();self.acquired=threading.Event();self.error=None
        self.last_applied_ns=None;self.process=None
        self.commands=queue.Queue(maxsize=4);self.stopping=False
        self.camera_stream=None;self.camera_reader=None

    def probe_bridge(self):
        result=subprocess.run(ros_command(ROOT/'replay_o6_hand_ros.py',['--probe']),capture_output=True,text=True,timeout=9.)
        if result.returncode or 'probe_ready' not in result.stdout:
            raise RuntimeError('O6 bridge discovery failed: '+(result.stdout+result.stderr)[-1000:])

    def start(self):
        c=self.config
        camera_read_fd,camera_write_fd=os.pipe()
        try:
            self.process=subprocess.Popen(ros_command(ROOT/'tools/tron2_act_ros_io.py',[
                '--expected-can',c['o6_can'],'--image-topic',c['image_topic'],'--info-topic',c['info_topic'],
                '--camera-fd',str(camera_write_fd)]),stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,bufsize=1,cwd=ROOT,start_new_session=True,
                pass_fds=(camera_write_fd,))
        finally:
            os.close(camera_write_fd)
        self.camera_stream=os.fdopen(camera_read_fd,'r',encoding='utf-8',buffering=1)
        def reader():
            try:
                for line in self.process.stdout:
                    try:event=json.loads(line)
                    except ValueError:continue
                    kind=event.get('event')
                    if kind=='hand_state':
                        self.hand_events.put_nowait(event)
                        self.trace.record('act_hand_feedback',**{k:v for k,v in event.items() if k!='event'})
                    elif kind=='hand_applied':
                        self.last_applied_ns=event['received_ns']
                        self.trace.record('act_hand_applied',**{k:v for k,v in event.items() if k!='event'})
                    elif kind=='ready':self.ready.set()
                    elif kind=='acquired':self.acquired.set()
                    elif kind=='error':
                        self.error=event['error'];self.trace.record('act_ros_error',message=self.error)
                    elif kind in {'hand_target','hand_target_dropped','released'}:
                        self.trace.record('act_ros_'+kind,**{k:v for k,v in event.items() if k!='event'})
                if not self.stopping:self.error=self.error or 'ROS sidecar exited'
            except BaseException as exc:self.error=str(exc)
        def read_camera():
            try:
                for line in self.camera_stream:
                    try:event=json.loads(line)
                    except ValueError:continue
                    if event.get('event')=='camera':self.camera=event
            except BaseException as exc:
                if not self.stopping:self.error='Camera pipe: '+str(exc)
        def writer():
            try:
                while True:
                    command=self.commands.get()
                    self.process.stdin.write(json.dumps(command)+'\n');self.process.stdin.flush()
                    if command['command']=='stop':return
            except BaseException as exc:
                if not self.stopping:self.error='ROS stdin: '+str(exc)
        self.reader=threading.Thread(target=reader,daemon=True);self.reader.start()
        self.camera_reader=threading.Thread(target=read_camera,daemon=True);self.camera_reader.start()
        self.writer=threading.Thread(target=writer,daemon=True);self.writer.start()

    def send(self,command):
        if self.error:raise RuntimeError(self.error)
        try:self.commands.put_nowait(command)
        except queue.Full:raise RuntimeError('O6 output queue blocked; no stale command catch-up')

    def camera_fresh(self,now):
        freshness_ns=round(float(self.config.get('io_freshness_ms',150.))*1e6)
        return bool(self.camera and now-self.camera['received_ns']<=freshness_ns)

    def healthy(self,now,need_applied=False,require_camera=True):
        freshness_ns=round(float(self.config.get('io_freshness_ms',150.))*1e6)
        if self.error:raise RuntimeError(self.error)
        if self.process and self.process.poll() is not None:raise RuntimeError('ROS I/O process exited')
        if require_camera and not self.camera_fresh(now):
            raise PendingObservation('Top camera absent/stale')
        if need_applied and (self.last_applied_ns is None or now-self.last_applied_ns>freshness_ns):
            raise RuntimeError('O6 bridge has not confirmed fresh applied commands')

    def close(self):
        self.stopping=True
        if self.process is None:return
        try:
            self.commands.put({'command':'stop'},timeout=.2)
            self.process.wait(timeout=7.)
        except (queue.Full,subprocess.TimeoutExpired):
            self.process.terminate()
            try:self.process.wait(timeout=2.)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2.)
        if self.camera_stream is not None:self.camera_stream.close()
        self.reader.join(timeout=1.);self.writer.join(timeout=1.)
        if self.camera_reader is not None:self.camera_reader.join(timeout=1.)
        if self.error:raise RuntimeError(self.error)


class Observations:
    def __init__(self,reference,camera_clock='receive',feedback_age_ms=0.,freshness_ms=150.):
        if not np.isfinite(freshness_ms) or not 100<=freshness_ms<=1000:
            raise ValueError('freshness_ms must be within 100..1000')
        gap_s=float(freshness_ms)/1000
        self.positions=Timeline(max_gap_s=gap_s)
        self.rotations=Timeline(rotations=True,max_gap_s=gap_s)
        self.hands=Timeline(max_gap_s=gap_s)
        self.reference=reference;self.clock=camera_clock;self.age_ns=round(feedback_age_ms*1e6)
        self.freshness_ns=round(freshness_ms*1e6)
        self.last_packet=None;self.last_robot_ns=None;self.last_hand_ns=None

    def pump(self,telemetry,io):
        packet=telemetry.packet
        if packet and packet.guid!=self.last_packet:
            if not packet.source_progress:raise RuntimeError('Robot feedback source clock did not advance')
            if packet.received_ns-packet.sent_ns>self.freshness_ns:
                raise RuntimeError(
                    f'Successful robot pose request RTT exceeds {self.freshness_ns/1e6:g}ms'
                )
            self.last_packet=packet.guid
            self.last_robot_ns=packet.received_ns
            stamp=(packet.sent_ns+packet.received_ns)//2-self.age_ns
            self.positions.add(stamp,packet.pose.left_position)
            self.rotations.add(stamp,Rotation.from_quat(packet.pose.left_quaternion[[1,2,3,0]]).as_matrix())
        while True:
            try:event=io.hand_events.get_nowait()
            except queue.Empty:break
            # The O6 header uses CLOCK_REALTIME and can move backwards during an
            # NTP correction. The callback receipt is captured in the same host
            # monotonic domain as camera receipt and robot feedback, so use it for
            # control alignment and retain the hardware header only in the trace.
            capture=int(event['received_ns']);age=time.monotonic_ns()-capture
            if age<0:
                raise RuntimeError('O6 measured feedback receive timestamp is in the future')
            # Ignore sidecar startup backlog and wait for a current measurement;
            # the normal freshness guard still stops if no current feedback follows.
            if age>self.freshness_ns:
                continue
            expected=['thumb_flexion','thumb_abduction','index_flexion','middle_flexion','ring_flexion','pinky_flexion']
            if event['names']!=expected:raise RuntimeError('Unexpected O6 feedback channel order')
            q_pos=np.asarray(event['q_pos'],float)
            if (q_pos.shape!=(6,) or not np.isfinite(q_pos).all() or
                    np.any((q_pos<0)|(q_pos>255))):
                raise RuntimeError('Raw O6 q_pos is not six measured SDK byte values')
            self.hands.add(int(capture),q_pos);self.last_hand_ns=capture

    def packet(self,camera,now,*,startup_cutoff=None):
        received=camera['received_ns']
        stamp=received if self.clock=='receive' else received-(camera['wall_received_ns']-camera['source_stamp_ns'])
        if startup_cutoff is not None:
            # A later measurement can supply the right interpolation endpoint,
            # but it can never supply history earlier than our first sample.
            if stamp < startup_cutoff or now-stamp > self.freshness_ns:
                return None
            for timeline, earliest in ((self.positions,stamp-100_000_000),
                                       (self.rotations,stamp-100_000_000),(self.hands,stamp)):
                if timeline.rows and earliest < timeline.rows[0][0]:
                    return None
        if now-received>self.freshness_ns:
            raise RuntimeError(f'Camera input older than {self.freshness_ns/1e6:g}ms')
        if not 0<=now-stamp<=self.freshness_ns:raise RuntimeError('Camera header clock/age invalid')
        if self.last_hand_ns is None or now-self.last_hand_ns>self.freshness_ns:
            raise PendingObservation('Measured O6 feedback absent/stale')
        state,p,r=observation_state(int(stamp),self.positions,self.rotations,self.hands,self.reference)
        return {'sequence':camera['sequence'],'stamp_ns':int(stamp),'jpeg':camera['jpeg'],
                'info':camera['info'],'state':state,'p':p,'r':r}


def make_pose_session(client,anchor,q,reference,config,trace):
    from types import SimpleNamespace
    from tron2_pose_replay import PoseReplaySession
    from tron2_pose_reference import load_profile
    class ACTPoseSession(PoseReplaySession):
        def __init__(self,*args):
            super().__init__(*args);self.nominal_p=Timeline();self.nominal_r=Timeline(rotations=True)
        def reference_at_ns(self,stamp):
            if self.start_ns is None or len(self.nominal_p.rows)<2:
                return reference.p.copy(),reference.r.copy()
            clipped=int(np.clip(stamp,self.nominal_p.rows[0][0],self.nominal_p.rows[-1][0]))
            return self.nominal_p.at(clipped),self.nominal_r.at(clipped)
        def stream(self,phase,reference_time_s=None,adaptive=True):
            now=time.monotonic_ns()
            t=reference_time_s
            if t is None:
                t=0. if self.start_ns is None else (now-self.start_ns)/1e9
            p,r=reference.target(t)
            self.nominal_p.add(now,p);self.nominal_r.add(now,r)
            super().stream(phase,t,adaptive)
    freshness_ms=float(config.get('io_freshness_ms',150.))
    args=SimpleNamespace(_pose_profile=load_profile(),_pose_trajectory=reference,_pose_q_start=q,
        pose_compensation=True,z_max_correction_mm=20.,feedback_timeout=freshness_ms/1000,
        pose_input_freshness_ms=freshness_ms,
        pose_preview_ms=config['pose_preview_ms'],pose_feedback_age_ms=config['pose_feedback_age_ms'],
        robot_ip=config['robot_ip'],rate=300.,interpolation='linear')
    session=ACTPoseSession(client,anchor,args)
    return session
