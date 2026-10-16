"""网页真机回放的参数、两段确认和 API 隔离测试；不连接真机。"""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "TransHandR0725/TransHandR/scripts"))

from realtime_multisource_dashboard import DashboardServer
from robot_replay_web import RobotReplayManager, replay_options


class ReplayWebTests(unittest.TestCase):
    def test_browser_numeric_defaults_pass_native_constraints(self):
        from html.parser import HTMLParser

        class Inputs(HTMLParser):
            def __init__(self):
                super().__init__()
                self.fields = {}

            def handle_starttag(self, tag, attrs):
                values = dict(attrs)
                if tag == "input" and values.get("id", "").startswith("robot"):
                    self.fields[values["id"]] = values

        parser = Inputs()
        parser.feed((ROOT / "TransHandR0725/TransHandR/scripts/dashboard/index.html").read_text())
        for name in ("robotScale", "robotSpeed", "robotMaxDisplacement", "robotCountdown"):
            field = parser.fields[name]
            value = float(field["value"])
            self.assertLessEqual(float(field["min"]), value)
            self.assertLessEqual(value, float(field["max"]))
            step = field["step"]
            if step != "any":
                quotient = (value - float(field["min"])) / float(step)
                self.assertAlmostEqual(quotient, round(quotient))
            self.assertIn("required", field)
        self.assertEqual(parser.fields["robotO6Can"]["pattern"], "can[0-9]+")
        self.assertEqual(parser.fields["robotO6Can"]["value"], "can1")

    def test_parameters_are_whitelisted_and_bounded(self):
        defaults = replay_options({})
        self.assertEqual(defaults["scale"], 1.0)
        self.assertEqual(defaults["speed"], 0.2)
        self.assertEqual(defaults["axis_map"], "y,z,x")
        self.assertEqual(defaults["o6_can"], "")
        self.assertEqual(replay_options({"o6_can": "can1"})["o6_can"], "can1")
        for payload in ({"scale": 2}, {"speed": 0}, {"countdown": 1},
                        {"axis_map": "x,x,z"}, {"robot_ip": "bad"},
                        {"with_o6_hand": "false"}, {"o6_can": "/dev/ttyUSB0"},
                        {"with_o6_hand": False, "o6_can": "can1"}):
            with self.assertRaises(ValueError):
                replay_options(payload)

    def test_preview_never_uses_execute_and_start_requires_both_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_000065.hdf5"
            path.touch()
            manager = RobotReplayManager(lambda _id: path, lambda: None,
                                         python="/fake/python")
            result = Mock(returncode=0, stdout="offline plan", stderr="")
            with patch("robot_replay_web.subprocess.run", return_value=result) as run:
                preview = manager.preview({"id": path.name, "scale": 1})
            self.assertIn("offline plan", preview["summary"])
            cmd = run.call_args.args[0]
            self.assertNotIn("--execute", cmd)
            self.assertNotIn("--web-control", cmd)
            self.assertIn("--file", cmd)
            with self.assertRaisesRegex(ValueError, "PREPARE"):
                manager.start({"id": path.name})
            with self.assertRaisesRegex(RuntimeError, "重新执行离线预检"):
                manager.start({"id": path.name, "scale": 0.3,
                               "confirmation": "PREPARE"})
            with self.assertRaisesRegex(RuntimeError, "未等待回放确认"):
                manager.confirm({"confirmation": "REPLAY"})

    def test_o6_can_is_passed_to_offline_and_execute_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_000065.hdf5"
            path.touch()
            guard = Mock()
            manager = RobotReplayManager(lambda _id: path, lambda: None,
                                         python="/fake/python",
                                         can_start_o6_bridge=guard)
            payload = {"id": path.name, "with_o6_hand": True, "o6_can": "can1"}
            with patch("robot_replay_web.subprocess.run", return_value=Mock(
                returncode=0, stdout="offline plan", stderr=""
            )) as run:
                manager.preview(payload)
            offline = run.call_args.args[0]
            self.assertEqual(offline[offline.index("--o6-can") + 1], "can1")
            self.assertNotIn("--execute", offline)
            guard.assert_not_called()
            with patch.object(manager, "_launch", return_value={"phase": "preparing"}) as launch:
                manager.start({**payload, "confirmation": "PREPARE"})
            guard.assert_called_once()
            execute = launch.call_args.args[0]
            self.assertEqual(execute[execute.index("--o6-can") + 1], "can1")
            self.assertIn("--execute", execute)

    def test_restore_needs_prepare_but_no_episode_preview(self):
        manager = RobotReplayManager(
            lambda _id: (_ for _ in ()).throw(AssertionError("no episode lookup")),
            lambda: None, python="/fake/python",
        )
        with self.assertRaisesRegex(ValueError, "PREPARE"):
            manager.restore({"countdown": 5})
        with self.assertRaisesRegex(ValueError, "不接受 episode"):
            manager.restore({"confirmation": "PREPARE", "id": "episode_65"})
        with patch.object(manager, "_launch", return_value={"phase": "preparing"}) as launch:
            result = manager.restore({"confirmation": "PREPARE", "countdown": 5})
        self.assertEqual(result["phase"], "preparing")
        command, episode_id, kind = launch.call_args.args
        self.assertIn("restore_tron2_fixed_start.py", command[2])
        self.assertEqual((episode_id, kind), ("", "restore"))
        for forbidden in ("--file", "--with-o6-hand", "--axis-map", "--max-displacement"):
            self.assertNotIn(forbidden, command)

    def test_restore_task_finishes_without_replay_confirmation(self):
        manager = RobotReplayManager(lambda _id: Path("unused"), lambda: None)
        child = "print('固定起点恢复完成；未发送轨迹。', flush=True)"
        manager._launch([sys.executable, "-u", "-c", child], "", "restore")
        deadline = time.monotonic() + 2
        while manager.status()["phase"] != "restored" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(manager.status()["phase"], "restored")
        self.assertEqual(manager.status()["exit_code"], 0)
        with self.assertRaisesRegex(RuntimeError, "未等待回放确认"):
            manager.confirm({"confirmation": "REPLAY"})


    def test_http_motion_endpoint_rejects_missing_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = Mock()
            manager.status.return_value = {"phase": "idle", "lines": []}
            dashboard = DashboardServer(
                "127.0.0.1", 0, lambda: {}, lambda _: None,
                lambda *_: {}, lambda _: {}, Path(directory), robot_replay=manager,
            )
            dashboard.start()
            try:
                port = dashboard.httpd.server_address[1]
                url = f"http://127.0.0.1:{port}/api/robot-replay/start"
                body = json.dumps({"confirmation": "PREPARE"}).encode()
                request = Request(url, body, {"Content-Type": "application/json"})
                with self.assertRaises(HTTPError) as error:
                    urlopen(request, timeout=2)
                self.assertEqual(error.exception.code, 403)
                manager.start.assert_not_called()

                request = Request(url, body, {
                    "Content-Type": "application/json",
                    "Origin": f"http://127.0.0.1:{port}",
                })
                manager.start.return_value = {"phase": "preparing"}
                with urlopen(request, timeout=2) as response:
                    self.assertEqual(json.load(response)["phase"], "preparing")
                manager.start.assert_called_once()

                restore_url = f"http://127.0.0.1:{port}/api/robot-replay/restore"
                request = Request(restore_url, body, {"Content-Type": "application/json"})
                with self.assertRaises(HTTPError) as error:
                    urlopen(request, timeout=2)
                self.assertEqual(error.exception.code, 403)
                manager.restore.assert_not_called()
                request = Request(restore_url, body, {
                    "Content-Type": "application/json",
                    "Origin": f"http://127.0.0.1:{port}",
                })
                manager.restore.return_value = {"phase": "preparing", "kind": "restore"}
                with urlopen(request, timeout=2) as response:
                    self.assertEqual(json.load(response)["kind"], "restore")
                manager.restore.assert_called_once()
            finally:
                dashboard.stop()

    def test_two_stage_web_confirmation_uses_separate_process_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_000065.hdf5"
            path.touch()
            manager = RobotReplayManager(lambda _id: path, lambda: None)
            with patch("robot_replay_web.subprocess.run", return_value=Mock(
                returncode=0, stdout="offline", stderr="")):
                manager.preview({"id": path.name})
            child = (
                "import sys;"
                "print('ROBOT_REPLAY_EVENT {\"phase\":\"awaiting_replay\"}',flush=True);"
                "word=sys.stdin.readline().strip();"
                "print('ROBOT_REPLAY_EVENT {\"phase\":\"replaying\"}',flush=True);"
                "print('ROBOT_REPLAY_EVENT {\"phase\":\"completed\"}',flush=True);"
                "sys.exit(0 if word=='REPLAY' else 2)"
            )
            with patch.object(manager, "_command", return_value=[
                sys.executable, "-u", "-c", child,
            ]):
                manager.start({"id": path.name, "confirmation": "PREPARE"})
            deadline = time.monotonic() + 2
            while manager.status()["phase"] != "awaiting_replay" and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(manager.status()["phase"], "awaiting_replay")
            manager.confirm({"confirmation": "REPLAY"})
            while manager.status()["phase"] != "completed" and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(manager.status()["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
