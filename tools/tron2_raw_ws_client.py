"""Raw WebSocket transport for the existing TRON2 JSON protocol client.

Importing this module does not connect or initialize limxsdk/RobotNative. Constructing
RawWSClient connects to the robot unless an offline socket_factory is injected.
The caller owns the existing replay process lock, motion guards and explicit close().
This bypasses the SDK's MROS ingress; it is not proof of identical server-side routing.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Callable, Optional

import websocket

from replay_tron2_tracker import PoseSample, TraceRecorder, Tron2JsonClient


class _RawWSTransport:
    """The narrow Robot facade used by inherited send(), with one receive owner."""

    def __init__(self, callback, on_failure, trace, io_timeout, close_timeout):
        self.callback = callback
        self.on_failure = on_failure
        self.trace = trace
        self.io_timeout = io_timeout
        self.close_timeout = close_timeout
        self.socket = None
        self.receiver = None
        self.stop = threading.Event()
        self.write_lock = threading.Lock()
        self.error_lock = threading.Lock()
        self.close_lock = threading.Lock()
        self.error: Optional[str] = None
        self.close_complete = False

    def _record(self, event, **fields):
        if self.trace is not None:
            self.trace.record(event, **fields)

    def _remember_error(self, message):
        # Never acquire the protocol Condition while holding write_lock. A query
        # can hold that Condition while waiting for write_lock in inherited send().
        with self.error_lock:
            first = self.error is None
            if first:
                self.error = message
        return first

    def _report_error(self, message):
        self.on_failure(message)
        self._record('raw_ws_error', message=message)

    def open(self, robot_ip, socket_factory, connect_timeout):
        url = f'ws://{robot_ip}:5000'
        try:
            self.socket = socket_factory(
                url, timeout=connect_timeout, enable_multithread=True,
                http_no_proxy=[robot_ip])
            # Set once, before either thread uses the socket. Per-request timeout
            # changes would also affect concurrent sends on the shared socket.
            self.socket.settimeout(self.io_timeout)
            self._record('raw_ws_connected', url=url, io_timeout_s=self.io_timeout,
                         automatic_reconnect=False)
            self.receiver = threading.Thread(
                target=self._receive, name='TRON2-RawWS-RX', daemon=True)
            self.receiver.start()
        except Exception as exc:
            message = f'RawWS connection failed: {exc}'
            if self._remember_error(message):
                self._report_error(message)
            self.close()
            raise RuntimeError(message) from exc

    def _receive(self):
        while not self.stop.is_set():
            try:
                payload = self.socket.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as exc:
                if not self.stop.is_set():
                    message = f'RawWS receive failed: {exc}'
                    if self._remember_error(message):
                        self._report_error(message)
                break
            if self.stop.is_set():
                break
            try:
                if payload in ('', b'', None):
                    raise EOFError('WebSocket closed')
                if isinstance(payload, bytes):
                    payload = payload.decode('utf-8')
                # No title/GUID filtering here: replies and notifications go to
                # the unchanged protocol callback. It owns all correlation.
                self.callback(payload)
            except Exception as exc:
                message = f'RawWS message handling failed: {exc}'
                if self._remember_error(message):
                    self._report_error(message)
                break

    def publishJsonMessage(self, text):
        failure = None
        first = False
        try:
            with self.write_lock:
                if self.stop.is_set():
                    raise RuntimeError('RawWS transport is closed')
                with self.error_lock:
                    existing = self.error
                if existing is not None:
                    raise RuntimeError(existing)
                try:
                    # The caller has already produced compact JSON. Do not parse,
                    # reserialize, cast coordinates or add a second envelope.
                    sent = self.socket.send(text)
                    if not isinstance(sent, int) or sent <= 0:
                        raise OSError('WebSocket send did not complete a frame')
                except Exception as exc:
                    failure = f'RawWS send failed: {exc}'
                    first = self._remember_error(failure)
                    raise
        except Exception as exc:
            # write_lock has been released before notifying protocol waiters.
            if first:
                self._report_error(failure)
            if failure is not None:
                raise RuntimeError(failure) from exc
            raise
        return True  # Transport completion only; not a controller execution ACK.

    def close(self):
        with self.close_lock:
            if self.close_complete:
                return
            self.stop.set()
            socket = self.socket
            cleanup_errors = []
            if socket is not None:
                # websocket-client's immediate shutdown avoids its default 3s
                # closing handshake and wakes a blocked recv before joining RX.
                abort = getattr(socket, 'abort', None)
                if callable(abort):
                    try:
                        abort()
                    except OSError:
                        # Shutdown below still closes an already-disconnected socket.
                        pass
                shutdown = getattr(socket, 'shutdown', None)
                try:
                    if callable(shutdown):
                        shutdown()
                    else:
                        socket.close()
                except Exception as exc:
                    cleanup_errors.append(f'socket shutdown failed: {exc}')
            if self.receiver is not None and self.receiver is not threading.current_thread():
                self.receiver.join(timeout=self.close_timeout)
                if self.receiver.is_alive():
                    cleanup_errors.append('receive thread did not stop within close_timeout')
            if cleanup_errors:
                message = 'RawWS close failed: ' + '; '.join(cleanup_errors)
                if self._remember_error(message):
                    self._report_error(message)
                # A subsequent close may retry cleanup, but sending stays disabled.
                raise RuntimeError(message)
            self.close_complete = True
            self._record('raw_ws_closed')


class RawWSClient(Tron2JsonClient):
    """Drop-in protocol client, using WS :5000 instead of a RobotNative instance.

    send/request/_on_message are inherited unchanged, including the ServoJ block,
    GUID handling, read-only retry rules and existing servop_command trace schema.
    No reconnect or motion retransmission is implemented. socket_factory exists
    for fake-socket tests and must be explicit in an offline caller.
    """

    def __init__(self, robot_ip: str, trace: Optional[TraceRecorder] = None, *,
                 socket_factory: Optional[Callable] = None,
                 connect_timeout: float = 4., io_timeout: float = .05,
                 close_timeout: float = 1.):
        for name, value in (('connect_timeout', connect_timeout),
                            ('io_timeout', io_timeout), ('close_timeout', close_timeout)):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not isinstance(robot_ip, str) or not robot_ip or any(c in robot_ip for c in '/?#'):
            raise ValueError('robot_ip must be a hostname or address, without a URL path')

        # Deliberately do not call super().__init__: it would construct RobotNative.
        # Keep protocol state in sync with Tron2JsonClient's constructor. All state
        # exists BEFORE connecting/starting RX, so an immediate identity is safe.
        self.accid = ''
        self.trace = trace
        self._condition = threading.Condition()
        self._robot_identity_ready = False
        self._robot_identity_error: Optional[str] = None
        self._messages: list[dict] = []
        self._failure: Optional[str] = None
        self._pose_probe_guid: Optional[str] = None
        self._pose_probe_sent_at = 0.
        self._pose_samples: list[PoseSample] = []
        self._pose_probe_error: Optional[str] = None
        self._snapshot_pose_guids: set[str] = set()
        self._client_closed = False
        self.robot = _RawWSTransport(
            self._on_message, self._transport_failed, trace, io_timeout, close_timeout)
        factory = websocket.create_connection if socket_factory is None else socket_factory
        self.robot.open(robot_ip, factory, connect_timeout)

    def _transport_failed(self, message):
        with self._condition:
            if self._failure is None:
                self._failure = message
            if self._robot_identity_error is None:
                self._robot_identity_error = message
            self._condition.notify_all()

    def wait_for_identity(self, timeout=5.):
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('identity timeout must be finite and positive')
        with self._condition:
            deadline = time.monotonic() + timeout
            while not self._robot_identity_ready:
                if self._robot_identity_error:
                    raise RuntimeError(self._robot_identity_error)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('no notify_robot_info/accid after connecting')
                self._condition.wait(remaining)
            if self._robot_identity_error:
                raise RuntimeError(self._robot_identity_error)

    @property
    def receiver_alive(self):
        thread = self.robot.receiver
        return thread is not None and thread.is_alive()

    @property
    def transport_error(self):
        with self.robot.error_lock:
            return self.robot.error

    def close(self):
        try:
            self.robot.close()
        finally:
            with self._condition:
                self._client_closed = True
                if self._robot_identity_error is None:
                    self._robot_identity_error = 'RawWS client closed'
                self._condition.notify_all()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
