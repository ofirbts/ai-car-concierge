from backend import database as db
from backend.database import VehicleSearchFilters, VehicleSort


def test_init_db_loads_hundred_vehicles(isolated_db):
    assert db.count_vehicles() == 100


def test_search_by_make(isolated_db):
    results = db.search_vehicles(VehicleSearchFilters(make="Tesla", limit=50))
    assert results
    assert all("tesla" in v.make.lower() for v in results)


def test_search_by_year(isolated_db):
    results = db.search_vehicles(VehicleSearchFilters(year=2020))
    assert len(results) == 3
    assert all(v.year == 2020 for v in results)


def test_search_price_range(isolated_db):
    results = db.search_vehicles(
        VehicleSearchFilters(price_min=80000, price_max=90000, limit=100)
    )
    assert results
    assert all(80000 <= v.price <= 90000 for v in results)


def test_pre_2022_flagged_pending_delisting(isolated_db):
    results = db.search_vehicles(VehicleSearchFilters(year=2019, limit=10))
    assert results
    assert all(v.pending_delisting for v in results)


def test_sort_price_asc(isolated_db):
    results = db.search_vehicles(
        VehicleSearchFilters(make="Tesla", limit=5, sort=VehicleSort.PRICE_ASC)
    )
    prices = [v.price for v in results]
    assert prices == sorted(prices)


def test_get_vehicle_by_id(isolated_db):
    vehicle = db.get_vehicle_by_id(16)
    assert vehicle is not None
    assert vehicle.id == 16


def test_in_stock_only_filter(isolated_db):
    all_zero_stock = db.search_vehicles(
        VehicleSearchFilters(make="Jaguar", model="F-PACE", year=2020)
    )
    assert len(all_zero_stock) == 1
    assert all_zero_stock[0].stock_count == 0

    in_stock = db.search_vehicles(
        VehicleSearchFilters(
            make="Jaguar", model="F-PACE", year=2020, in_stock_only=True
        )
    )
    assert in_stock == []
