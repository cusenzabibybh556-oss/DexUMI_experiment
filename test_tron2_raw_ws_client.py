"""All transport tests use in-memory fake sockets; no network or RobotNative."""
import json
import queue
import sys
import threading
import time
import unittest
from unittest.mock import Mock, patch

from websocket import WebSocketTimeoutException

from replay_tron2_tracker import Tron2JsonClient
from tools.tron2_raw_ws_client import RawWSClient


class FakeSocket:
    def __init__(self, identity=True):
        self.incoming = queue.Queue()
        self.sent = []
        self.closed = False
        self.timeout = None
        self.close_count = 0
        self.recv_owners = set()
        self.send_error = None
        self.on_send = None
        self._sending = 0
        self.max_parallel_sends = 0
        self._lock = threading.Lock()
        if identity:
            self.feed({'title':'notify_robot_info','accid':'DACH_TRON2A_253'})

    def feed(self, message):
        self.incoming.put(json.dumps(message) if isinstance(message, dict) else message)

    def settimeout(self, timeout):
        self.timeout = timeout

    def recv(self):
        self.recv_owners.add(threading.get_ident())
        try:
            item = self.incoming.get(timeout=.005)
        except queue.Empty:
            raise WebSocketTimeoutException('fake receive timeout')
        if isinstance(item, BaseException):
            raise item
        return item

    def send(self, text):
        if self.send_error is not None:
            raise self.send_error
        with self._lock:
            self._sending += 1
            self.max_parallel_sends = max(self.max_parallel_sends, self._sending)
        try:
            # Give competing writers an opportunity to expose missing locking.
            time.sleep(.00001)
            self.sent.append(text)
            if self.on_send:
                self.on_send(json.loads(text))
            return len(text.encode('utf-8'))
        finally:
            with self._lock:
                self._sending -= 1

    def close(self):
        self.close_count += 1
        self.closed = True
        self.feed('')


def wait_until(predicate, timeout=.5):
    deadline = time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate():
            return
        time.sleep(.001)
    raise AssertionError('fake test condition did not become true')


def pose_reply(guid, value=0):
    return {'title':'response_get_servo_pose','guid':guid,
            'data':{'result':'success','left_position':[value,0.,0.],
                    'right_position':[0.,0.,0.],
                    'left_quat':[1.,0.,0.,0.],'right_quat':[1.,0.,0.,0.]}}


class RawWSClientTests(unittest.TestCase):
    def make(self, socket=None, **kwargs):
        socket = socket or FakeSocket()
        factory = Mock(return_value=socket)
        client = RawWSClient('robot.invalid',socket_factory=factory,**kwargs)
        self.addCleanup(client.close)
        return client,socket,factory

    def ready(self, **kwargs):
        client,socket,factory = self.make(**kwargs)
        client.wait_for_identity(.5)
        return client,socket,factory

    def test_protocol_methods_are_inherited_and_native_constructor_is_not_called(self):
        self.assertIs(RawWSClient.send,Tron2JsonClient.send)
        self.assertIs(RawWSClient.request,Tron2JsonClient.request)
        self.assertIs(RawWSClient._on_message,Tron2JsonClient._on_message)
        with patch.object(Tron2JsonClient,'__init__',side_effect=AssertionError('native init')):
            client,socket,factory = self.ready()
        factory.assert_called_once_with('ws://robot.invalid:5000',timeout=4.,
                                        enable_multithread=True,http_no_proxy=['robot.invalid'])
        self.assertEqual(socket.timeout,.05)
        self.assertEqual(client.accid,'DACH_TRON2A_253')

    def test_immediate_identity_arrives_after_all_callback_state_is_initialized(self):
        client,socket,_ = self.ready()
        self.assertTrue(client.receiver_alive)
        self.assertIsNone(client.failure())
        self.assertTrue(client._robot_identity_ready)
        self.assertEqual(len(client._messages),1)
        self.assertEqual(len(socket.recv_owners),1)

    def test_no_identity_refuses_send_and_times_out_without_publishing(self):
        client,socket,_ = self.make(FakeSocket(identity=False))
        with self.assertRaisesRegex(RuntimeError,'identity not ready'):
            client.send('request_get_joint_state',{})
        with self.assertRaises(TimeoutError):
            client.wait_for_identity(.015)
        self.assertEqual(socket.sent,[])

    def test_invalid_arguments_fail_before_socket_factory(self):
        factory = Mock()
        for field in ['connect_timeout','io_timeout','close_timeout']:
            for value in [0.,-1.,float('nan'),float('inf'),True]:
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                    RawWSClient('robot.invalid',socket_factory=factory,**{field:value})
        for address in ['',None,'http://robot.invalid']:
            with self.assertRaises(ValueError):
                RawWSClient(address,socket_factory=factory)
        factory.assert_not_called()

    def test_4501_frames_preserve_all_coordinates_and_trace_schema(self):
        trace = Mock()
        client,socket,_ = self.ready(trace=trace)
        for index in range(4501):
            left=[.451234567890123,.256,-.306+.02*(index/4500),.54,.55,-.45,-.46]
            right=[-.032,-.17,-.755,.999,-.006,.045,.005]
            data={'left_pos':left,'right_pos':right}
            guid=client.send('request_servop',data,guid=f'frame-{index}')
            raw=socket.sent[-1]
            self.assertNotIn(' ',raw)
            payload=json.loads(raw)
            self.assertEqual(payload['data'],data)
            self.assertEqual(set(payload),{'accid','title','timestamp','guid','data'})
            self.assertEqual(payload['guid'],guid)
            self.assertEqual(payload['title'],'request_servop')
        events=[c for c in trace.record.call_args_list if c.args[0]=='servop_command']
        self.assertEqual(len(events),4501)
        self.assertEqual(events[-1].kwargs['left_pos'],left)
        self.assertEqual(len(socket.sent),4501)

    def test_single_rx_dispatches_wrong_guid_and_unrelated_messages_without_losing_reply(self):
        client,socket,_ = self.ready()
        def reply(message):
            socket.feed(pose_reply('another-client',99))
            socket.feed({'title':'unrelated_notification','data':{'sample':1}})
            socket.feed(pose_reply(message['guid'],.42))
        socket.on_send=reply
        value=client.request('request_get_servo_pose',{},'response_get_servo_pose',.1)
        self.assertEqual(value['data']['left_position'][0],.42)
        self.assertEqual(len(socket.recv_owners),1)
        self.assertIn('unrelated_notification',[m['title'] for m in client._messages])

    def test_two_concurrent_queries_accept_out_of_order_guid_replies(self):
        client,socket,_ = self.ready()
        pending=[];lock=threading.Lock();answers={};errors=[]
        def reply(message):
            with lock:
                pending.append(message)
                if len(pending)==2:
                    for m in reversed(pending):
                        socket.feed(pose_reply(m['guid'],m['data']['id']))
        socket.on_send=reply
        def query(index):
            try:
                answers[index]=client.request('request_get_servo_pose',{'id':index},
                                              'response_get_servo_pose',.2)
            except Exception as exc:
                errors.append(exc)
        threads=[threading.Thread(target=query,args=(index,)) for index in [1,2]]
        for thread in threads:thread.start()
        for thread in threads:thread.join(.5)
        self.assertFalse(any(t.is_alive() for t in threads))
        self.assertFalse(errors)
        self.assertEqual({k:v['data']['left_position'][0] for k,v in answers.items()},{1:1,2:2})

    def test_late_timed_out_reply_does_not_satisfy_new_query(self):
        client,socket,_ = self.ready()
        with self.assertRaises(TimeoutError):
            client.request('request_get_servo_pose',{},'response_get_servo_pose',.015)
        old=json.loads(socket.sent[-1])['guid']
        def reply(message):
            socket.feed(pose_reply(old,99))
            socket.feed(pose_reply(message['guid'],.2))
        socket.on_send=reply
        answer=client.request('request_get_servo_pose',{},'response_get_servo_pose',.1)
        self.assertEqual(answer['data']['left_position'][0],.2)

    def test_controller_failure_notification_is_latched_while_query_waits(self):
        client,socket,_ = self.ready()
        def reply(message):
            socket.feed({'title':'notify_servop','data':{'result':'fail_limits'}})
            socket.feed(pose_reply(message['guid']))
        socket.on_send=reply
        client.request('request_get_servo_pose',{},'response_get_servo_pose',.1)
        self.assertEqual(client.failure(),'fail_limits')
        socket.feed({'title':'notify_servop','data':{'result':'success'}})
        wait_until(lambda:len(client._messages)>=4)
        self.assertEqual(client.failure(),'fail_limits')

    def test_concurrent_publishers_do_not_interleave_socket_writes(self):
        client,socket,_ = self.ready();errors=[]
        def send(index):
            try:
                for _ in range(15):client.send('test_query',{'writer':index})
            except Exception as exc:errors.append(exc)
        threads=[threading.Thread(target=send,args=(i,)) for i in range(4)]
        for thread in threads:thread.start()
        for thread in threads:thread.join(.5)
        self.assertFalse(errors)
        self.assertFalse(any(t.is_alive() for t in threads))
        self.assertEqual(socket.max_parallel_sends,1)
        self.assertEqual(len(socket.sent),60)

    def test_normal_receive_timeout_and_malformed_json_do_not_lose_identity(self):
        client,socket,_ = self.make(FakeSocket(identity=False))
        socket.feed(WebSocketTimeoutException('idle'))
        socket.feed('not json')
        socket.feed({'title':'notify_robot_info','accid':'DACH_TRON2A_253'})
        client.wait_for_identity(.2)
        self.assertIsNone(client.failure())

    def test_eof_latches_failure_and_never_reconnects_or_sends_again(self):
        client,socket,factory = self.ready()
        socket.feed('')
        wait_until(lambda:client.failure() is not None)
        wait_until(lambda:not client.receiver_alive)
        with self.assertRaises(RuntimeError):client.send('request_get_joint_state',{})
        self.assertEqual(socket.sent,[])
        self.assertEqual(factory.call_count,1)

    def test_transport_failure_before_identity_fails_wait_immediately(self):
        client,socket,_ = self.make(FakeSocket(identity=False))
        socket.feed(OSError('gone'))
        wait_until(lambda:client.failure() is not None)
        with self.assertRaisesRegex(RuntimeError,'gone'):client.wait_for_identity(.2)

    def test_failed_send_is_not_retried_and_later_motion_is_blocked(self):
        client,socket,factory = self.ready()
        socket.send_error=OSError('transmit broken')
        data={'left_pos':[0.]*7,'right_pos':[0.]*7}
        with self.assertRaisesRegex(RuntimeError,'transmit broken'):client.send('request_servop',data)
        self.assertIn('transmit broken',client.failure())
        socket.send_error=None
        with self.assertRaises(RuntimeError):client.send('request_servop',data)
        self.assertEqual(socket.sent,[])
        self.assertEqual(factory.call_count,1)

    def test_non_read_command_request_timeout_does_not_republish(self):
        client,socket,_ = self.ready()
        with self.assertRaises(TimeoutError):
            client.request('test_non_read',{'value':1},'test_non_read_reply',.015)
        self.assertEqual(len(socket.sent),1)

    def test_servoj_stays_blocked_and_identity_change_prevents_transmit(self):
        client,socket,_ = self.ready()
        with self.assertRaisesRegex(RuntimeError,'ServoJ'):client.send('request_servoj',{})
        socket.feed({'title':'notify_robot_info','accid':'different-robot'})
        wait_until(lambda:client._robot_identity_error is not None)
        with self.assertRaisesRegex(RuntimeError,'identity changed'):
            client.send('request_get_joint_state',{})
        self.assertEqual(socket.sent,[])

    def test_async_probe_and_snapshot_do_not_consume_each_others_guid(self):
        client,socket,_ = self.ready()
        socket.on_send=lambda message:socket.feed(pose_reply(message['guid']))
        client.probe_pose()
        wait_until(lambda:client.latest_pose_sample() is not None)
        self.assertEqual(len(client.pose_samples()),1)
        client.current_servo_pose(label='fake snapshot')
        self.assertEqual(len(client.pose_samples()),1)
        self.assertFalse(client._snapshot_pose_guids)

    def test_close_unblocks_receiver_is_idempotent_and_blocks_future_send(self):
        client,socket,_ = self.ready()
        client.close();client.close()
        self.assertFalse(client.receiver_alive)
        self.assertTrue(socket.closed)
        self.assertEqual(socket.close_count,1)
        with self.assertRaisesRegex(RuntimeError,'closed'):client.send('request_get_joint_state',{})

    def test_factory_failure_is_not_retried(self):
        factory=Mock(side_effect=OSError('offline'))
        with self.assertRaisesRegex(RuntimeError,'offline'):
            RawWSClient('robot.invalid',socket_factory=factory)
        self.assertEqual(factory.call_count,1)

    def test_no_limxsdk_import_is_needed_for_fake_connection(self):
        with patch.dict(sys.modules,{'limxsdk':None,'limxsdk.robot':None}):
            client,_,_=self.ready()
            self.assertEqual(client.accid,'DACH_TRON2A_253')

    def test_identity_before_callback_thread_and_setup_failure_cleanup(self):
        socket=FakeSocket()
        socket.settimeout=Mock(side_effect=OSError('setup failed'))
        factory=Mock(return_value=socket)
        with self.assertRaisesRegex(RuntimeError,'setup failed'):
            RawWSClient('robot.invalid',socket_factory=factory)
        self.assertTrue(socket.closed)
        self.assertEqual(socket.close_count,1)
        self.assertFalse(socket.recv_owners)

    def test_zero_or_noninteger_send_result_latches_transport_failure(self):
        for result in (0,None):
            with self.subTest(result=result):
                client,socket,_=self.ready()
                socket.send=Mock(return_value=result)
                with self.assertRaisesRegex(RuntimeError,'did not complete'):
                    client.send('request_get_joint_state',{})
                self.assertIn('did not complete',client.failure())
                self.assertEqual(socket.send.call_count,1)

    def test_close_during_query_is_bounded_and_does_not_retry_motion(self):
        client,socket,_=self.ready();errors=[]
        def pending():
            try:client.request('test_non_read',{'value':1},'test_non_read_reply',.05)
            except Exception as exc:errors.append(exc)
        thread=threading.Thread(target=pending)
        thread.start();wait_until(lambda:len(socket.sent)==1)
        client.close();thread.join(.2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)
        self.assertEqual(len(socket.sent),1)
        self.assertFalse(client.receiver_alive)

    def test_stuck_receive_thread_is_reported_and_cleanup_can_be_retried(self):
        entered=threading.Event();release=threading.Event()
        class StuckSocket(FakeSocket):
            def recv(self):
                entered.set();release.wait();return ''
            def close(self):
                self.close_count+=1;self.closed=True
        socket=StuckSocket(identity=False)
        client=RawWSClient('robot.invalid',socket_factory=lambda *a,**k:socket,
                           close_timeout=.01)
        self.assertTrue(entered.wait(.2))
        try:
            with self.assertRaisesRegex(RuntimeError,'did not stop'):
                client.close()
            self.assertIn('did not stop',client.failure())
            with self.assertRaises(RuntimeError):client.send('request_get_joint_state',{})
        finally:
            release.set();client.close()
        self.assertFalse(client.receiver_alive)


if __name__=='__main__':unittest.main()
