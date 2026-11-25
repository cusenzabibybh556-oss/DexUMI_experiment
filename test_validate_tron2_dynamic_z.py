"""Validation-tool contracts; offline tests must never construct a robot client."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from tools import validate_tron2_dynamic_z as validation


class DynamicValidationTests(unittest.TestCase):
    def test_small_path_returns_to_anchor_and_bounds_nominal_speed(self):
        t, z = validation.targets()
        self.assertEqual(len(t), 6001)
        self.assertEqual(t[-1], 20)
        self.assertEqual(z[0], 0)
        self.assertEqual(z[-1], 0)
        self.assertAlmostEqual(z.max(), .02)
        self.assertGreaterEqual(z.min(), 0)
        self.assertLessEqual(np.max(abs(np.diff(z)/np.diff(t))), .005)

    def test_offline_planning_never_constructs_transport_or_session(self):
        snapshot = {'payload_reply': {'data': {'result': 'success',
                    'data': [.396, 0, 0, 0, .001, 0, 0, 0]}}}
        anchor = SimpleNamespace(left_position=np.array([.5, .2, -.3]))
        with TemporaryDirectory() as folder:
            source = Path(folder)/'snapshot.json'
            source.write_text(json.dumps(snapshot))
            out = Path(folder)/'planned'
            with patch.object(validation, 'prepare_envelope', return_value=(None, anchor, {})), \
                 patch.object(validation, 'Tron2JsonClient') as sdk, \
                 patch.object(validation, 'RawWSClient') as ws, \
                 patch.object(validation, 'ZReplaySession') as session:
                self.assertEqual(validation.main(['--mode', 'continuous', '--snapshot', str(source),
                                                 '--output', str(out)]), 0)
                sdk.assert_not_called(); ws.assert_not_called(); session.assert_not_called()
            self.assertEqual(json.loads((out/'result.json').read_text())['result'], 'planned_only')

    def test_unexpected_payload_stops_before_path_or_transport(self):
        with TemporaryDirectory() as folder:
            source = Path(folder)/'snapshot.json'
            source.write_text(json.dumps({'payload_reply': {'data': {'result': 'success',
                'data': [.2, 0, 0, 0, .001, 0, 0, 0]}}}))
            with patch.object(validation, 'prepare_envelope') as envelope, \
                 patch.object(validation, 'Tron2JsonClient') as sdk:
                self.assertEqual(validation.main(['--mode', 'continuous', '--snapshot', str(source),
                    '--output', str(Path(folder)/'rejected')]), 2)
                envelope.assert_not_called(); sdk.assert_not_called()


if __name__ == '__main__':
    unittest.main()
