import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tron2_hardware_residual import (FEATURE_NAMES, OUTPUT_NAMES, SCHEMA,
    HardwareResidualModel, feature_vector)


class HardwareResidualTests(unittest.TestCase):
    def payload(self):
        return dict(schema=SCHEMA, feature_names=FEATURE_NAMES, output_names=OUTPUT_NAMES,
            feature_mean=[0.]*len(FEATURE_NAMES), feature_scale=[1.]*len(FEATURE_NAMES),
            adjustment_coefficients=np.zeros((6,len(FEATURE_NAMES))).tolist(),
            adjustment_intercept=[.001,-.002,.003,0,0,0],
            prediction_limits=dict(maximum_total_translation_m=.04,maximum_total_rotation_deg=5.),
            feedback_gate=dict(full_scale_translation_m=.001,full_scale_rotation_deg=.2,
                               zero_residual_zero_adjustment=True))

    def test_feature_contract_and_prediction(self):
        x=feature_vector([.01,0,.02],[0,0,0],np.zeros(7),[.01,0,.02],[0,0,0],
                         [.002,0,0],[0,0,0])
        self.assertEqual(x.shape,(len(FEATURE_NAMES),))
        model=HardwareResidualModel(self.payload())
        translation,rotation,residual=model.predict_total([.01,0,.02],[0,0,0],np.zeros(7),
            [.01,0,.02],[0,0,0],[0,0,0],[0,0,0])
        np.testing.assert_allclose(translation,[.01,0,.02])
        np.testing.assert_allclose(rotation,np.eye(3))
        np.testing.assert_allclose(residual[:3],[0,0,0])
        translation,_,adjustment=model.predict_total([.01,0,.02],[0,0,0],np.zeros(7),
            [.01,0,.02],[0,0,0],[.002,0,0],[0,0,0])
        np.testing.assert_allclose(translation,[.011,-.002,.023])
        np.testing.assert_allclose(adjustment[:3],[.001,-.002,.003])

    def test_load_hash_and_reject_widened_limits(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'model.json';path.write_text(json.dumps(self.payload()))
            model=HardwareResidualModel.load(path)
            self.assertEqual(len(model.sha256),64)
            bad=self.payload();bad['prediction_limits']['maximum_total_translation_m']=.05
            with self.assertRaises(ValueError):HardwareResidualModel(bad)


if __name__ == '__main__':unittest.main()
