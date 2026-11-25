import pytest

from linkerhand_retarget.motion.linkerforce.hand.linkerforce_o6 import RightHand


@pytest.mark.parametrize("span", [0.50, -0.50])
def test_thumb_leaves_neutral_and_returns_after_a_slow_sweep(span):
    hand = RightHand(handcore=None)
    neutral = 4.47
    hand.calibrationoriginal = [0, neutral]
    hand.calibrationfistpose = [0, neutral + span]
    update = hand._filter_thumb_abduction

    for offset in [0.0, 0.005, -0.005] * 4:
        assert update(neutral + offset) == neutral
    assert update(neutral + span) == neutral  # One spike after a steady hold.
    for step in range(51):
        update(neutral + span * step / 50)
    for _ in range(9):
        value = update(neutral + span)
    assert value == pytest.approx(neutral + span)
    for _ in range(30):
        value = update(neutral)
    assert value == pytest.approx(neutral)
