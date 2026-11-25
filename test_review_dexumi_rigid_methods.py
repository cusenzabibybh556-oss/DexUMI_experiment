import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from review_dexumi_rigid_methods import mean_world, multi_reference_fit, review
from solve_tron2_tracker_calibration import PoseGroup, inverse, transform


class RigidMethodsTests(unittest.TestCase):
    def setUp(self):
        self.X = transform(Rotation.from_euler('xyz', [15, -30, 55], degrees=True).as_matrix(), [.03, -.04, .06])
        self.Y = transform(Rotation.from_euler('xyz', [50, 20, -70], degrees=True).as_matrix(), [1, -.5, .2])
        self.groups = []
        for i, angle in enumerate(([0, 0, 0], [20, 0, 0], [0, 20, 0], [0, 0, 20], [10, -10, 10], [-10, 10, 0]), 1):
            P = transform(Rotation.from_euler('xyz', angle, degrees=True).as_matrix(), [i * .02, -.01 * i, .04])
            self.groups.append(PoseGroup(i, 'synthetic', P, self.Y @ P @ self.X, [2], 10))
        self.cal = {'T_ET': self.X.tolist(), 'T_FT': self.X.tolist(), 'T_WB': self.Y.tolist(),
                    'T_BW': inverse(self.Y).tolist(), 'fit_group_ids': [1, 2, 3], 'validation_group_ids': [4, 5, 6]}

    def test_mean_world_preserves_exact_rigid_relationship(self):
        np.testing.assert_allclose(mean_world(self.groups, self.X), self.Y, atol=1e-12)

    def test_multi_reference_uses_original_relative_formula(self):
        def optimizer(fun, x0, **kwargs):
            self.assertLess(fun(x0), 1e-12)
            perturbed = x0.copy(); perturbed[0] += .01
            self.assertGreater(fun(perturbed), 1e-4)
            self.assertEqual(kwargs['method'], 'L-BFGS-B')
            self.assertEqual(len(kwargs['bounds']), 6)
            return SimpleNamespace(success=True, fun=fun(x0), x=x0, message='synthetic check')
        with patch('review_dexumi_rigid_methods.minimize', side_effect=optimizer):
            X, Y, info = multi_reference_fit(self.groups, self.X)
        np.testing.assert_allclose(X, self.X, atol=1e-12)
        np.testing.assert_allclose(Y, self.Y, atol=1e-12)
        self.assertFalse(info['translation_bound_hit'])

    def test_auxiliary_fits_never_receive_validation_groups(self):
        def multiref(groups, seed):
            self.assertEqual([g.id for g in groups], [1, 2, 3])
            return self.X, self.Y, {}
        def joint(groups):
            self.assertEqual([g.id for g in groups], [1, 2, 3])
            return self.X, self.Y, {}
        with patch('review_dexumi_rigid_methods.multi_reference_fit', side_effect=multiref), \
                patch('review_dexumi_rigid_methods.fit_transforms', side_effect=joint):
            out = review(self.groups, self.cal, [0, 0, 0])
        self.assertFalse(out['scale_fitted'])
        self.assertEqual(out['methods']['joint_SE3_fit_only']['metrics']['held_out']['count'], 3)

    def test_status18_cannot_silently_replace_training_group(self):
        self.groups[0].statuses = [18]
        with self.assertRaisesRegex(ValueError, '非status2'):
            review(self.groups, self.cal, [0, 0, 0])

    def test_overlapping_splits_rejected(self):
        self.cal['validation_group_ids'] = [3, 4, 5]
        with self.assertRaisesRegex(ValueError, '不独立'):
            review(self.groups, self.cal, [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
