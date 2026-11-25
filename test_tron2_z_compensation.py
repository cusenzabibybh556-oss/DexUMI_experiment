"""Pure synthetic feedback/plant tests; no hardware or SDK imports."""

from collections import deque
from dataclasses import replace
import math
import unittest

from tron2_z_compensation import Config, ZCompensationController


def ns(seconds):
    return round(seconds*1e9)


def observe(controller, sequence, seconds, actual=-.01, nominal=0., **kwargs):
    return controller.observe(sequence, ns(seconds)-1_000_000,
                              ns(seconds), ns(seconds), nominal, actual, **kwargs)


class ControllerTests(unittest.TestCase):
    def controller(self, **kwargs):
        controller = ZCompensationController(replace(Config(), **kwargs))
        controller.advance(0)
        return controller

    def test_configuration_rejects_nonfinite_and_invalid_bounds(self):
        for key in Config.__dataclass_fields__:
            for value in (math.nan, math.inf, -math.inf, True, '1', -1.):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    replace(Config(), **{key: value})
        for key in ('max_rate_m_s', 'feedback_max_age_s', 'feedback_max_gap_s',
                    'max_dt_s', 'max_rtt_s'):
            with self.assertRaises(ValueError):
                replace(Config(), **{key: 0.})

    def test_first_feedback_primes_and_first_advance_never_jumps(self):
        controller = ZCompensationController()
        self.assertTrue(observe(controller, 'a', 1.))
        self.assertEqual(controller.status, 'primed')
        self.assertEqual(controller.advance(ns(1.)), 0.)
        self.assertTrue(observe(controller, 'b', 1.05))
        self.assertAlmostEqual(controller.target_offset, .00005)
        self.assertEqual(controller.advance(ns(1.05)), .00005)

    def test_duplicate_old_guid_never_integrates_or_refreshes_freshness(self):
        controller = self.controller()
        observe(controller, 'a', .01)
        observe(controller, 'b', .06)
        target = controller.target_offset
        self.assertFalse(observe(controller, 'b', .10))
        self.assertEqual(controller.target_offset, target)
        controller.advance(ns(.10))
        applied = controller.applied_offset
        controller.advance(ns(.211))
        self.assertEqual(controller.status, 'stale')
        self.assertFalse(observe(controller, 'a', .22))
        self.assertEqual(controller.status, 'stale')
        self.assertEqual(controller.target_offset, applied)
        self.assertEqual(controller.advance(ns(.3)), applied)

    def test_target_uses_applied_instead_of_unapplied_integral_queue(self):
        controller = self.controller()
        observe(controller, 1, .01)
        for index in range(2, 10):
            observe(controller, index, .01+.05*(index-1))
        self.assertEqual(controller.applied_offset, 0.)
        self.assertAlmostEqual(controller.target_offset, .00005)

    def test_expiry_freezes_before_catching_up_and_recovery_primes(self):
        controller = self.controller()
        observe(controller, 1, .01)
        observe(controller, 2, .06)
        controller.advance(ns(.07))
        applied = controller.applied_offset
        controller.advance(ns(.5))
        self.assertEqual(controller.target_offset, applied)
        self.assertTrue(observe(controller, 3, .51))
        self.assertEqual(controller.status, 'primed')
        self.assertEqual(controller.target_offset, applied)
        observe(controller, 4, .56)
        self.assertGreater(controller.target_offset, applied)

    def test_long_sample_gap_and_dt_cap_do_not_catch_up(self):
        controller = self.controller(feedback_max_age_s=.5)
        observe(controller, 1, .01)
        observe(controller, 2, .21)
        self.assertEqual(controller.status, 'gap')
        self.assertEqual(controller.target_offset, 0.)
        observe(controller, 3, .33)
        self.assertAlmostEqual(controller.target_offset, .0001)

    def test_age_rtt_and_out_of_order_samples_freeze_and_reprime(self):
        for fault in ('age', 'rtt', 'order'):
            with self.subTest(fault=fault):
                controller = self.controller()
                observe(controller, 1, .01)
                observe(controller, 2, .06)
                if fault == 'age':
                    args = (3, ns(.061), ns(.062), ns(.213), 0., -.01)
                elif fault == 'rtt':
                    args = (3, ns(.001), ns(.102), ns(.102), 0., -.01)
                else:
                    args = (3, ns(.02), ns(.03), ns(.10), 0., -.01)
                self.assertFalse(controller.observe(*args))
                self.assertEqual(controller.target_offset, controller.applied_offset)
                observe(controller, 4, .25)
                self.assertEqual(controller.status, 'primed')

    def test_deadband_and_negative_error_cannot_command_below_zero(self):
        controller = self.controller()
        observe(controller, 1, .01)
        observe(controller, 2, .06, actual=-.0003)
        self.assertEqual(controller.target_offset, 0.)
        self.assertEqual(controller.status, 'deadband')
        observe(controller, 3, .11, actual=.01)
        self.assertEqual(controller.target_offset, 0.)
        self.assertEqual(controller.status, 'saturated_low')
        observe(controller, 4, .16, actual=-.0005)
        self.assertAlmostEqual(controller.target_offset, .5*(.0005-.0003)*.05)

    def test_saturation_then_reverse_has_no_windup(self):
        controller = self.controller(max_offset_m=.0002)
        for index in range(1, 50):
            t=.01+.05*(index-1)
            observe(controller, index, t)
            controller.advance(ns(t))
        self.assertEqual(controller.applied_offset, .0002)
        self.assertEqual(controller.status, 'saturated_high')
        observe(controller, 50, 2.46, actual=.01)
        self.assertLess(controller.target_offset, controller.applied_offset)
        controller.advance(ns(2.46))
        self.assertLess(controller.applied_offset, .0002)

    def test_disabled_freezes_without_unwinding_and_reenable_primes(self):
        controller = self.controller()
        observe(controller, 1, .01)
        observe(controller, 2, .06)
        controller.advance(ns(.07))
        applied=controller.applied_offset
        self.assertFalse(observe(controller, 3, .08, enabled=False))
        self.assertEqual(controller.status, 'disabled')
        self.assertEqual(controller.advance(ns(.10)), applied)
        observe(controller, 4, .11)
        self.assertEqual(controller.status, 'primed')
        self.assertEqual(controller.target_offset, applied)

    def test_shared_host_clock_rejects_reverse_and_equal_time_is_idempotent(self):
        controller = self.controller()
        observe(controller, 1, .01)
        controller.advance(ns(.02))
        self.assertEqual(controller.advance(ns(.02)), 0.)
        with self.assertRaises(ValueError):
            observe(controller, 2, .019)
        self.assertEqual(controller.status, 'invalid')
        self.assertEqual(controller.applied_offset, 0.)
        with self.assertRaises(ValueError):
            controller.advance(ns(.019))

    def test_nonfinite_or_invalid_timestamps_never_move_applied(self):
        bad_cases = [
            ('a', 0, 0, 0, math.nan, 0.),
            ('a', 0, 0, 0, 0., math.inf),
            ('a', 0, 0, 0, 1e308, -1e308),
            ('a', 2, 1, 3, 0., 0.),
            ('a', 0, 3, 2, 0., 0.),
            ('a', -1, 0, 0, 0., 0.),
            ('a', .1, 1, 2, 0., 0.),
            ('a', 0, 0, True, 0., 0.),
            ('', 0, 0, 0, 0., 0.),
            (True, 0, 0, 0, 0., 0.),
        ]
        for args in bad_cases:
            controller = ZCompensationController()
            with self.subTest(args=args), self.assertRaises(ValueError):
                controller.observe(*args)
            self.assertEqual(controller.applied_offset, 0.)
            self.assertEqual(controller.target_offset, 0.)

    def test_exact_age_rtt_gap_boundaries_and_zero_caps(self):
        controller = self.controller(max_offset_m=0.)
        self.assertTrue(controller.observe(1, 0, ns(.1), ns(.25), 0., -.01))
        self.assertTrue(controller.observe(2, ns(.15), ns(.25), ns(.25), 0., -.01))
        self.assertNotEqual(controller.status, 'gap')
        self.assertEqual(controller.advance(ns(.25)), 0.)

    def test_300hz_output_obeys_rate_in_both_directions(self):
        controller = self.controller()
        values=[]
        for index in range(1, 601):
            t=index/300
            if index % 15 == 0:
                observe(controller, index, t, actual=-.01 if t<1 else .01)
            values.append(controller.advance(ns(t)))
        self.assertTrue(all(0. <= x <= .02 for x in values))
        self.assertLessEqual(max(abs(b-a) for a,b in zip(values,values[1:])), .001/300+1e-12)
        self.assertGreater(max(values), 0.)
        self.assertLess(values[-1], max(values))

    def test_runtime_limit_defaults_to_5mm_and_expansion_has_no_jump(self):
        controller=self.controller()
        self.assertEqual(controller.active_limit, .005)
        for index in range(1, 150):
            t=.01+.05*(index-1)
            observe(controller,index,t)
            controller.advance(ns(t))
        self.assertEqual(controller.applied_offset, .005)
        applied,target=controller.applied_offset,controller.target_offset
        controller.set_active_limit(.02)
        self.assertEqual(controller.active_limit, .02)
        self.assertEqual((controller.applied_offset,controller.target_offset),(applied,target))
        observe(controller,150,7.46)
        self.assertGreater(controller.target_offset, applied)

    def test_runtime_limit_reduction_is_atomic_and_clamps_pending_only(self):
        controller=self.controller()
        observe(controller,1,.01)
        observe(controller,2,.06)
        controller.advance(ns(.06))
        observe(controller,3,.11)
        applied=controller.applied_offset
        for value in (-1., .021, math.nan, math.inf, True, applied/2):
            before=(controller.active_limit,controller.applied_offset,controller.target_offset)
            with self.subTest(value=value),self.assertRaises(ValueError):
                controller.set_active_limit(value)
            self.assertEqual((controller.active_limit,controller.applied_offset,controller.target_offset),before)
        controller.set_active_limit(applied)
        self.assertEqual(controller.target_offset,applied)
        self.assertEqual(controller.advance(ns(.11)),applied)

    def test_freeze_cancels_pending_target_and_next_feedback_only_primes(self):
        controller=self.controller()
        observe(controller,1,.01)
        observe(controller,2,.06)
        self.assertGreater(controller.target_offset,controller.applied_offset)
        controller.freeze(ns(.06))
        self.assertEqual(controller.status,'frozen')
        self.assertEqual(controller.advance(ns(.07)),0.)
        observe(controller,3,.11)
        self.assertEqual(controller.status,'primed')
        self.assertEqual(controller.target_offset,0.)
        observe(controller,4,.16)
        controller.advance(ns(.16))
        applied=controller.applied_offset
        self.assertGreater(applied,0.)
        observe(controller,5,.21)
        controller.freeze()
        self.assertEqual(controller.target_offset,applied)
        self.assertEqual(controller.advance(ns(.21)),applied)

    def test_new_guid_with_replayed_timestamp_or_backward_sequence_is_rejected(self):
        controller=self.controller()
        observe(controller,10,.01)
        self.assertFalse(observe(controller,9,.06))
        self.assertEqual(controller.status,'rejected_order')
        observe(controller,11,.11)
        self.assertEqual(controller.status,'primed')
        self.assertFalse(controller.observe('fresh-guid',ns(.109),ns(.11),ns(.12),0.,-.01))
        self.assertEqual(controller.status,'rejected_order')


class DelayedPlantTests(unittest.TestCase):
    @staticmethod
    def simulate(bias, gain, cap, duration=60.):
        """150ms pure plant delay, 20Hz telemetry, 300Hz command generation.

        Query RTT is 1ms; physical response delay is deliberately separate.
        y = gain * applied_offset(t - delay) - static_bias.
        """
        controller=ZCompensationController(Config(max_offset_m=cap), active_limit_m=cap)
        controller.advance(0)
        history=deque([0.]*45, maxlen=45)
        errors=[]
        for index in range(1, round(duration*300)+1):
            t=index/300
            actual=gain*history[0]-bias
            if index % 15 == 0:
                observe(controller, index//15, t, actual=actual)
            applied=controller.advance(ns(t))
            history.append(applied)
            if t>duration-5:
                errors.append(-actual)
            assert 0. <= applied <= cap
        return controller, errors

    def test_20mm_range_converges_for_delayed_gain_and_static_bias_matrix(self):
        for bias in (.010, .013):
            for gain in (.9, 1.05):
                with self.subTest(bias=bias, gain=gain):
                    controller, errors=self.simulate(bias,gain,.02)
                    self.assertLess(max(abs(e) for e in errors), .0004)
                    self.assertLess(controller.applied_offset, .02)
                    self.assertNotEqual(controller.status, 'saturated_high')

    def test_5mm_cap_remains_saturated_with_honest_residual_error(self):
        for bias in (.010, .013):
            for gain in (.9, 1.05):
                with self.subTest(bias=bias,gain=gain):
                    controller, errors=self.simulate(bias,gain,.005)
                    self.assertAlmostEqual(controller.applied_offset, .005)
                    self.assertEqual(controller.status, 'saturated_high')
                    expected=bias-gain*.005
                    self.assertAlmostEqual(min(errors), expected)
                    self.assertGreaterEqual(min(errors), .00475-1e-12)


if __name__ == '__main__':
    unittest.main()
