from pathlib import Path
import sys

import can
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from LinkerHand.core.can.linker_hand_o6_can import LinkerHandO6Can


def test_b1_palm_b6_frames_fill_all_210_cells_without_motion():
    driver = LinkerHandO6Can.__new__(LinkerHandO6Can)
    driver.can_id = 0x28
    driver.palm_enabled = True
    driver.palm_matrix = np.full((14, 15), -1, np.int16)
    driver.palm_generation = 0
    driver.palm_complete_generation = 0
    driver.palm_fragments = set()
    driver.serial_parts = {}

    values = np.arange(210, dtype=np.uint8)
    for offset in range(0, 210, 5):
        row, col = divmod(offset, 15)
        frame = can.Message(
            arbitration_id=0x28,
            data=[0xb6, row, col, *values[offset:offset + 5].tolist()],
            is_extended_id=False,
        )
        driver.process_response(frame)

    assert driver.palm_generation == 42
    assert driver.palm_complete_generation == 1
    assert np.array_equal(driver.palm_matrix.ravel(), values)
    assert driver.palm_matrix[13, 14] == 209

    driver.process_response(can.Message(
        arbitration_id=0x28, data=[0xb6, 14, 0, 255], is_extended_id=False,
    ))
    assert driver.palm_matrix[13, 14] == 209


def test_b1_palm_query_uses_official_read_only_opcode():
    driver = LinkerHandO6Can.__new__(LinkerHandO6Can)
    driver.palm_enabled = True
    driver.palm_matrix = np.zeros((14, 15), np.int16)
    driver.palm_generation = 0
    driver.palm_complete_generation = 0
    driver.palm_fragments = set()
    sent = []

    def fake_send(opcode, payload, sleep=0):
        sent.append((opcode, payload))
        driver.palm_complete_generation += 1

    driver.send_frame = fake_send
    matrix = driver.get_palm_matrix_touch()
    assert sent == [(0xb6, [0x0e, 0x0f])]
    assert matrix.shape == (14, 15)


def test_b1_finger_a4_frames_preserve_legacy_hdf5_padding():
    driver = LinkerHandO6Can.__new__(LinkerHandO6Can)
    driver.can_id = 0x28
    driver.thumb_matrix = np.full((12, 6), -1, np.int16)
    driver.index_matrix = np.full((12, 6), -1, np.int16)
    driver.matrix_map = {row * 16: row for row in range(12)}
    for opcode in (0xb1, 0xb2):
        for row in range(10):
            driver.process_response(can.Message(
                arbitration_id=0x28,
                data=[opcode, row * 16, 0, 0, row, 0],
                is_extended_id=False,
            ))
    for matrix in (driver.thumb_matrix, driver.index_matrix):
        assert np.all(matrix[:10, :4] >= 0)
        assert np.all(matrix[10:, :] == -1)
        assert np.all(matrix[:, 4:] == -1)
