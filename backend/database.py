from __future__ import annotations

import hashlib
import sqlite3
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = PROJECT_ROOT / "data" / "inventory.sql"
DEFAULT_DB_PATH = PROJECT_ROOT / "car_inventory.db"
SALES_MIN_YEAR = 2022
POLICY_BLOCK_MESSAGE = (
    "This vehicle is Pending De-listing (model year before 2022) and cannot be "
    "sold or reserved under our 2022+ Sales Policy."
)

_db_path: Path = DEFAULT_DB_PATH


class VehicleSort(str, Enum):
    YEAR_DESC_PRICE_ASC = "year_desc_price_asc"
    PRICE_ASC = "price_asc"
    PRICE_DESC = "price_desc"


_SORT_SQL = {
    VehicleSort.YEAR_DESC_PRICE_ASC: "year DESC, price ASC",
    VehicleSort.PRICE_ASC: "price ASC",
    VehicleSort.PRICE_DESC: "price DESC",
}


class Vehicle(BaseModel):
    id: int
    make: str
    model: str
    year: int
    color: str
    price: float
    fuel_type: str
    stock_count: int
    pending_delisting: bool = False


class VehicleSearchFilters(BaseModel):
    make: str | None = None
    model: str | None = None
    year: int | None = None
    year_min: int | None = None
    year_max: int | None = None
    color: str | None = None
    fuel_type: str | None = None
    price_min: float | None = None
    price_max: float | None = None
    in_stock_only: bool = False
    limit: int = Field(default=20, ge=1, le=100)
    sort: VehicleSort = VehicleSort.YEAR_DESC_PRICE_ASC


class VehicleNotFoundError(Exception):
    pass


class OutOfStockError(Exception):
    pass


class PolicyViolationError(Exception):
    def __init__(self, message: str, vehicle_id: int | None = None):
        self.vehicle_id = vehicle_id
        super().__init__(message)


def set_db_path(path: Path) -> None:
    global _db_path
    _db_path = path


def get_db_path() -> Path:
    return _db_path


def _hash_path() -> Path:
    return _db_path.with_suffix(_db_path.suffix + ".sqlhash")


def _inventory_sql_hash() -> str:
    return hashlib.sha256(SQL_PATH.read_bytes()).hexdigest()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(force: bool = False) -> None:
    if not SQL_PATH.is_file():
        raise FileNotFoundError(f"Inventory SQL not found: {SQL_PATH}")
    current_hash = _inventory_sql_hash()
    hash_path = _hash_path()
    if _db_path.exists() and not force:
        if hash_path.is_file() and hash_path.read_text(encoding="utf-8") == current_hash:
            return
    script = SQL_PATH.read_text(encoding="utf-8")
    if _db_path.exists():
        _db_path.unlink()
    conn = get_connection()
    try:
        conn.executescript(script)
        conn.commit()
    finally:
        conn.close()
    hash_path.write_text(current_hash, encoding="utf-8")


def assert_sellable(vehicle: Vehicle) -> None:
    if vehicle.pending_delisting:
        raise PolicyViolationError(POLICY_BLOCK_MESSAGE, vehicle.id)


def _row_to_vehicle(row: sqlite3.Row) -> Vehicle:
    year = int(row["year"])
    return Vehicle(
        id=int(row["id"]),
        make=str(row["make"]),
        model=str(row["model"]),
        year=year,
        color=str(row["color"]),
        price=float(row["price"]),
        fuel_type=str(row["fuel_type"]),
        stock_count=int(row["stock_count"]),
        pending_delisting=year < SALES_MIN_YEAR,
    )


def search_vehicles(filters: VehicleSearchFilters | None = None) -> list[Vehicle]:
    f = filters or VehicleSearchFilters()
    clauses: list[str] = []
    params: list[Any] = []

    if f.make:
        clauses.append("LOWER(make) LIKE LOWER(?)")
        params.append(f"%{f.make.strip()}%")
    if f.model:
        clauses.append("LOWER(model) LIKE LOWER(?)")
        params.append(f"%{f.model.strip()}%")
    if f.year is not None:
        clauses.append("year = ?")
        params.append(f.year)
    if f.year_min is not None:
        clauses.append("year >= ?")
        params.append(f.year_min)
    if f.year_max is not None:
        clauses.append("year <= ?")
        params.append(f.year_max)
    if f.color:
        clauses.append("LOWER(color) LIKE LOWER(?)")
        params.append(f"%{f.color.strip()}%")
    if f.fuel_type:
        clauses.append("LOWER(fuel_type) LIKE LOWER(?)")
        params.append(f"%{f.fuel_type.strip()}%")
    if f.price_min is not None:
        clauses.append("price >= ?")
        params.append(f.price_min)
    if f.price_max is not None:
        clauses.append("price <= ?")
        params.append(f.price_max)
    if f.in_stock_only:
        clauses.append("stock_count > 0")

    order = _SORT_SQL.get(f.sort, _SORT_SQL[VehicleSort.YEAR_DESC_PRICE_ASC])
    sql = "SELECT * FROM vehicles"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += f" ORDER BY {order} LIMIT ?"
    params.append(f.limit)

    conn = get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [_row_to_vehicle(row) for row in rows]


def get_vehicle_by_id(vehicle_id: int) -> Vehicle | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM vehicles WHERE id = ?",
            (vehicle_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return _row_to_vehicle(row)


def reserve_vehicle(vehicle_id: int) -> Vehicle:
    vehicle = get_vehicle_by_id(vehicle_id)
    if vehicle is None:
        raise VehicleNotFoundError(f"Vehicle {vehicle_id} not found")
    assert_sellable(vehicle)
    if vehicle.stock_count <= 0:
        raise OutOfStockError(f"Vehicle {vehicle_id} is out of stock")

    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            UPDATE vehicles
            SET stock_count = stock_count - 1
            WHERE id = ? AND stock_count > 0
            """,
            (vehicle_id,),
        )
        conn.commit()
        if cursor.rowcount == 0:
            raise OutOfStockError(f"Vehicle {vehicle_id} is out of stock")
    finally:
        conn.close()

    updated = get_vehicle_by_id(vehicle_id)
    if updated is None:
        raise VehicleNotFoundError(f"Vehicle {vehicle_id} not found")
    return updated


def count_vehicles() -> int:
    conn = get_connection()
    try:
        row = conn.execute("SELECT COUNT(*) AS c FROM vehicles").fetchone()
    finally:
        conn.close()
    return int(row["c"]) if row else 0
