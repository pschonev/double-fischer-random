import pytest
from src.positions import get_chess960_position, get_scharnagl_number
from src.utils import is_valid_chess960_position


def test_generate_all_positions():
    # Test that we generate 960 unique positions
    positions = [get_chess960_position(i) for i in range(960)]

    # Check we have exactly 960 unique positions
    assert len(positions) == 960
    assert len(set(positions)) == 960

    # Check all positions are valid Chess960 positions
    assert all(is_valid_chess960_position(pos) for pos in positions)


def test_bijective_mapping():
    # Test that converting number->position->number gives the same number
    for original_number in range(960):
        position = get_chess960_position(original_number)
        recovered_number = get_scharnagl_number(position)
        assert original_number == recovered_number, (
            f"Number {original_number} converted to position {position} "
            f"but converted back to {recovered_number}"
        )


def test_invalid_scharnagl_numbers():
    # Test that invalid numbers raise ValueError
    with pytest.raises(ValueError):
        get_chess960_position(-1)

    with pytest.raises(ValueError):
        get_chess960_position(960)
