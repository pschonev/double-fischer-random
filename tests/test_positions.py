import pytest

from dfrc_analysis.positions.positions import (
    chess960_to_dfrc_uid,
    dfrc_to_chess960_uids,
    get_chess960_position,
    get_scharnagl_number,
    is_valid_chess960_position,
)


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


def test_dfrc_uid_conversion():
    n = 960
    seen_uids = set()

    for white_id in range(n):
        for black_id in range(n):
            uid = chess960_to_dfrc_uid(white_id, black_id)
            assert uid not in seen_uids, f"UID {uid} was already generated"
            seen_uids.add(uid)

            recovered_white_id, recovered_black_id = dfrc_to_chess960_uids(uid)
            assert recovered_white_id == white_id and recovered_black_id == black_id, (
                f"Conversion mismatch: ({white_id}, {black_id}) generated UID {uid} "
                f"but converted back to ({recovered_white_id}, {recovered_black_id})"
            )

    # Ensure we have 960 * 960 unique UIDs
    assert len(seen_uids) == n * n, (
        f"Expected {n * n} unique UIDs, found {len(seen_uids)}"
    )
