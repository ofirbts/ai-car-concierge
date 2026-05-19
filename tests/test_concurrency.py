import concurrent.futures

from backend.database import (
    OutOfStockError,
    get_connection,
    get_vehicle_by_id,
    reserve_vehicle,
)


def test_parallel_reserve_one_succeeds(isolated_db):
    vehicle_id = 16
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE vehicles SET stock_count = 1 WHERE id = ?",
            (vehicle_id,),
        )
        conn.commit()
    finally:
        conn.close()

    def attempt(key: str) -> str:
        try:
            reserve_vehicle(vehicle_id, idempotency_key=key)
            return "ok"
        except OutOfStockError:
            return "out_of_stock"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, ["parallel-a", "parallel-b"]))

    assert results.count("ok") == 1
    assert results.count("out_of_stock") == 1
    assert get_vehicle_by_id(vehicle_id).stock_count == 0
