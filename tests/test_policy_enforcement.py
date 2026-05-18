import pytest

from backend.database import (
    OutOfStockError,
    PolicyViolationError,
    get_vehicle_by_id,
    reserve_vehicle,
)


def test_reserve_decrements_stock(isolated_db):
    before = get_vehicle_by_id(16)
    assert before is not None
    after = reserve_vehicle(16)
    assert after.stock_count == before.stock_count - 1


def test_reserve_idempotent_guard_out_of_stock(isolated_db):
    vehicle = get_vehicle_by_id(5)
    assert vehicle is not None
    with pytest.raises(PolicyViolationError):
        reserve_vehicle(5)


def test_out_of_stock_vehicle(isolated_db):
    vehicle = get_vehicle_by_id(32)
    assert vehicle is not None
    assert vehicle.stock_count == 0
    with pytest.raises(OutOfStockError):
        reserve_vehicle(32)
