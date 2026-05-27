from __future__ import annotations

import hashlib
import json
import sqlite3
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = PROJECT_ROOT / "data" / "inventory.sql"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "car_inventory.db"
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


class IdempotencyConflictError(Exception):
    pass


RESERVATIONS_DDL = """
CREATE TABLE IF NOT EXISTS reservations (
    idempotency_key TEXT PRIMARY KEY,
    vehicle_id INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

PURCHASE_NOTIFICATIONS_DDL = """
CREATE TABLE IF NOT EXISTS purchase_notifications (
    idempotency_key TEXT PRIMARY KEY,
    customer_email TEXT NOT NULL,
    vehicle_id INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

ACTION_AUDIT_DDL = """
CREATE TABLE IF NOT EXISTS action_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    action TEXT NOT NULL,
    vehicle_id INTEGER,
    customer_email TEXT,
    outcome TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

JOB_BROKER_DDL = """
CREATE TABLE IF NOT EXISTS job_broker_jobs (
    job_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    result_json TEXT,
    error TEXT,
    metadata_json TEXT NOT NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0
);
"""


def set_db_path(path: Path) -> None:
    global _db_path
    _db_path = path


def get_db_path() -> Path:
    return _db_path


def _hash_path() -> Path:
    return _db_path.with_suffix(_db_path.suffix + ".sqlhash")


def _inventory_sql_hash() -> str:
    return hashlib.sha256(SQL_PATH.read_bytes()).hexdigest()


def _ensure_db_parent() -> None:
    _db_path.parent.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    _ensure_db_parent()
    conn = sqlite3.connect(_db_path, timeout=10.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(force: bool = False) -> None:
    _ensure_db_parent()
    if not SQL_PATH.is_file():
        raise FileNotFoundError(f"Inventory SQL not found: {SQL_PATH}")
    current_hash = _inventory_sql_hash()
    hash_path = _hash_path()
    if _db_path.exists() and not force:
        if hash_path.is_file() and hash_path.read_text(encoding="utf-8") == current_hash:
            ensure_app_tables()
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
    ensure_app_tables()


def ensure_app_tables() -> None:
    conn = get_connection()
    try:
        conn.executescript(
            RESERVATIONS_DDL
            + PURCHASE_NOTIFICATIONS_DDL
            + ACTION_AUDIT_DDL
            + JOB_BROKER_DDL
            + """
CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_id TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""
        )
        conn.commit()
    finally:
        conn.close()


def save_job_record(record: dict[str, Any]) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO job_broker_jobs (
                job_id, state, created_at, started_at, finished_at,
                result_json, error, metadata_json, cancel_requested
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                state=excluded.state,
                created_at=excluded.created_at,
                started_at=excluded.started_at,
                finished_at=excluded.finished_at,
                result_json=excluded.result_json,
                error=excluded.error,
                metadata_json=excluded.metadata_json,
                cancel_requested=excluded.cancel_requested
            """,
            (
                record["job_id"],
                record["state"],
                record["created_at"],
                record.get("started_at"),
                record.get("finished_at"),
                json.dumps(record.get("result"), ensure_ascii=False) if record.get("result") is not None else None,
                record.get("error"),
                json.dumps(record.get("metadata", {}), ensure_ascii=False),
                1 if record.get("cancel_requested") else 0,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_job_record(job_id: str) -> dict[str, Any] | None:
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT job_id, state, created_at, started_at, finished_at,
                   result_json, error, metadata_json, cancel_requested
            FROM job_broker_jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "job_id": str(row["job_id"]),
        "state": str(row["state"]),
        "created_at": float(row["created_at"]),
        "started_at": float(row["started_at"]) if row["started_at"] is not None else None,
        "finished_at": float(row["finished_at"]) if row["finished_at"] is not None else None,
        "result": json.loads(row["result_json"]) if row["result_json"] else None,
        "error": str(row["error"]) if row["error"] is not None else None,
        "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        "cancel_requested": bool(row["cancel_requested"]),
    }


def request_job_cancel(job_id: str) -> bool:
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            UPDATE job_broker_jobs
            SET cancel_requested = 1
            WHERE job_id = ? AND state IN ('pending', 'running')
            """,
            (job_id,),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def try_claim_purchase_notification(
    idempotency_key: str,
    customer_email: str,
    vehicle_id: int | None,
) -> bool:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO purchase_notifications (idempotency_key, customer_email, vehicle_id)
            VALUES (?, ?, ?)
            """,
            (idempotency_key, customer_email, vehicle_id),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def record_audit(
    request_id: str,
    action: str,
    outcome: str,
    *,
    vehicle_id: int | None = None,
    customer_email: str | None = None,
    detail: str | None = None,
) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO action_audit (
                request_id, action, vehicle_id, customer_email, outcome, detail
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (request_id, action, vehicle_id, customer_email, outcome, detail),
        )
        conn.commit()
    finally:
        conn.close()


def count_audit_rows() -> int:
    conn = get_connection()
    try:
        row = conn.execute("SELECT COUNT(*) AS c FROM action_audit").fetchone()
    finally:
        conn.close()
    return int(row["c"]) if row else 0


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


def make_exists_in_inventory(make: str) -> bool:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM vehicles WHERE LOWER(make) LIKE LOWER(?) LIMIT 1",
            (f"%{make.strip()}%",),
        ).fetchone()
    finally:
        conn.close()
    return row is not None


def model_exists_for_make(make: str, model: str) -> bool:
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT 1 FROM vehicles
            WHERE LOWER(make) LIKE LOWER(?) AND LOWER(model) LIKE LOWER(?)
            LIMIT 1
            """,
            (f"%{make.strip()}%", f"%{model.strip()}%"),
        ).fetchone()
    finally:
        conn.close()
    return row is not None


def _get_reservation_vehicle_id(idempotency_key: str) -> int | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT vehicle_id FROM reservations WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
    finally:
        conn.close()
    return int(row["vehicle_id"]) if row else None


def reserve_vehicle(vehicle_id: int, idempotency_key: str | None = None) -> Vehicle:
    if idempotency_key:
        existing_vehicle_id = _get_reservation_vehicle_id(idempotency_key)
        if existing_vehicle_id is not None:
            if existing_vehicle_id != vehicle_id:
                raise IdempotencyConflictError(
                    f"Idempotency key already used for vehicle #{existing_vehicle_id}"
                )
            vehicle = get_vehicle_by_id(existing_vehicle_id)
            if vehicle is None:
                raise VehicleNotFoundError(f"Vehicle {existing_vehicle_id} not found")
            return vehicle

    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM vehicles WHERE id = ?",
            (vehicle_id,),
        ).fetchone()
        if row is None:
            raise VehicleNotFoundError(f"Vehicle {vehicle_id} not found")
        vehicle = _row_to_vehicle(row)
        assert_sellable(vehicle)
        if vehicle.stock_count <= 0:
            raise OutOfStockError(f"Vehicle {vehicle_id} is out of stock")

        cursor = conn.execute(
            """
            UPDATE vehicles
            SET stock_count = stock_count - 1
            WHERE id = ? AND stock_count > 0
            """,
            (vehicle_id,),
        )
        if cursor.rowcount == 0:
            raise OutOfStockError(f"Vehicle {vehicle_id} is out of stock")

        if idempotency_key:
            conn.execute(
                "INSERT INTO reservations (idempotency_key, vehicle_id) VALUES (?, ?)",
                (idempotency_key, vehicle_id),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    updated = get_vehicle_by_id(vehicle_id)
    if updated is None:
        raise VehicleNotFoundError(f"Vehicle {vehicle_id} not found")
    return updated


def list_distinct_makes() -> list[str]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT make FROM vehicles ORDER BY LENGTH(make) DESC, make ASC"
        ).fetchall()
    finally:
        conn.close()
    return [str(row["make"]) for row in rows]


def count_vehicles() -> int:
    conn = get_connection()
    try:
        row = conn.execute("SELECT COUNT(*) AS c FROM vehicles").fetchone()
    finally:
        conn.close()
    return int(row["c"]) if row else 0
