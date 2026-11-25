from copy import deepcopy
import unittest

from filter_tron2_tracker_status2 import filter_sample


def sample(statuses=None):
    statuses = statuses or [2] * 20
    trackers, robots, pairs = [], [], []
    for i, status in enumerate(statuses):
        ts = 50_000_000 + i * 100_000_000
        trackers.append({'position_m': [0., 0., 0.], 'quaternion_wxyz': [1., 0., 0., 0.],
                         'received_monotonic_ns': ts, 'tracking_status': status})
        robots.append({'position_m': [0., 0., 0.], 'quaternion_wxyz': [1., 0., 0., 0.],
                       'received_monotonic_ns': ts + 1_000_000})
        pairs.append({'robot_index': i, 'tracker_index': i, 'receive_gap_ms': 1.})
    return {'event': 'sample', 'id': 17, 'label': 'synthetic', 'excluded': False,
            'started_monotonic_ns': 0, 'ended_monotonic_ns': 2_000_000_000,
            'tracker': trackers, 'robot': robots, 'pairs': pairs}


class Status2SubsetTests(unittest.TestCase):
    def test_drop_original_non2_pairs_without_rematching_or_source_mutation(self):
        statuses = [2] * 20; statuses[3] = statuses[12] = 18
        raw = sample(statuses); before = deepcopy(raw)
        derived, report = filter_sample(raw, 4)
        self.assertEqual(raw, before)
        self.assertTrue(report['eligible'])
        self.assertEqual(report['kept_pair_count'], 18)
        self.assertNotIn(3, [p['robot_index'] for p in report['kept_source_pairs']])
        self.assertNotIn(12, [p['robot_index'] for p in report['kept_source_pairs']])
        self.assertTrue(all(t['tracking_status'] == 2 for t in derived['tracker']))
        self.assertEqual(derived['id'], 17)
        self.assertEqual(report['source_sample_index'], 4)

    def test_insufficient_pair_count_rejected(self):
        statuses = [2 if i in {0, 2, 4, 6, 8, 10, 12, 15, 19} else 18 for i in range(20)]
        derived, report = filter_sample(sample(statuses), 0)
        self.assertFalse(report['eligible'])
        self.assertTrue(derived['excluded'])
        self.assertIn('保留的原始status2配对不足10对', report['reasons'])

    def test_count_does_not_replace_time_span_requirement(self):
        _, report = filter_sample(sample([18] * 5 + [2] * 10 + [18] * 5), 0)
        self.assertFalse(report['eligible'])
        self.assertIn('tracker保留帧时间跨度不足1秒', report['reasons'])

    def test_long_gap_is_rejected_even_with_count_and_span(self):
        _, report = filter_sample(sample([2] * 6 + [18] * 8 + [2] * 6), 0)
        self.assertEqual(report['kept_pair_count'], 12)
        self.assertGreater(report['tracker_retained_span_seconds'], 1)
        self.assertFalse(report['eligible'])
        self.assertTrue(any('超过0.5秒' in r for r in report['reasons']))

    def test_stationarity_recomputed_on_retained_pairs(self):
        raw = sample(); raw['tracker'][10]['position_m'][0] = .01
        _, report = filter_sample(raw, 0)
        self.assertFalse(report['eligible'])
        self.assertTrue(any('未静止' in r for r in report['reasons']))

    def test_original_pairs_must_match_raw_frame_times(self):
        raw = sample(); raw['pairs'][3]['tracker_index'] = 4
        _, report = filter_sample(raw, 0)
        self.assertFalse(report['eligible'])
        self.assertTrue(any('原始pairs索引' in r for r in report['reasons']))

    def test_source_quality_rejection_is_not_bypassed(self):
        _, report = filter_sample(sample(), 0, '机器人时间戳重复')
        self.assertFalse(report['eligible'])
        self.assertEqual(report['kept_pair_count'], 0)
        self.assertTrue(any('机器人时间戳重复' in r for r in report['reasons']))


if __name__ == '__main__':
    unittest.main()
