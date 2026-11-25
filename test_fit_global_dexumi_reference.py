"""离线合并必须保持身份与留出隔离，测试不读取硬件。"""
import unittest
from unittest.mock import patch

import numpy as np

import fit_global_dexumi_reference as module
from solve_tron2_tracker_calibration import PoseGroup


class GlobalSelectionTests(unittest.TestCase):
    def test_overlap_rejected_before_loading(self):
        with patch.object(module, 'load_diagnostic') as loader:
            with self.assertRaisesRegex(ValueError, '留出'):
                module.load_selected('unused', {}, 'raw-xyzw', [1, 2], [2, 3])
            loader.assert_not_called()

    def test_duplicate_fit_rejected_before_loading(self):
        with patch.object(module, 'load_diagnostic') as loader:
            with self.assertRaisesRegex(ValueError, '重复'):
                module.load_selected('unused', {}, 'raw-xyzw', [1, 1], [3])
            loader.assert_not_called()

    def test_status18_not_salvaged_as_fit(self):
        groups = [PoseGroup(1, '', np.eye(4), np.eye(4), [2, 18], 20)]
        with patch.object(module, 'load_diagnostic', return_value=({}, groups, [], 'sha', [1])), \
                patch.object(module, 'check_compatible'):
            with self.assertRaisesRegex(ValueError, 'status2'):
                module.load_selected('unused', {}, 'raw-xyzw', [1], [])

    def test_selects_explicit_roles_only(self):
        groups = [PoseGroup(i, '', np.eye(4), np.eye(4), [2], 20) for i in range(1, 5)]
        with patch.object(module, 'load_diagnostic', return_value=({}, groups, [], 'sha', [1, 2, 3, 4])), \
                patch.object(module, 'check_compatible'):
            _, fit, held, _, _ = module.load_selected('unused', {}, 'raw-xyzw', [2, 1], [3])
        self.assertEqual([g.id for g in fit], [2, 1])
        self.assertEqual([g.id for g in held], [3])


if __name__ == '__main__':
    unittest.main()
