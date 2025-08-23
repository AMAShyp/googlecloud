# db_handler.py
from __future__ import annotations

import json
import os
from typing import Any, Iterable, Mapping, Optional

import pandas as pd
import sqlalchemy as sa
import streamlit as st
from google.cloud.sql.connector import Connector, IPTypes
from google.oauth2 import service_account


# ───────────────────────────────────────────────────────────────
# 1) One small pooled engine per process (NOT 1 connection/session)
#    • Avoids exhausting Postgres max_connections (53300).
#    • Uses Cloud SQL Connector + pg8000 under SQLAlchemy.
# ───────────────────────────────────────────────────────────────

def _normalize_cfg(raw: Mapping[str, Any]) -> dict:
    """Normalize/validate the expected Cloud SQL config."""
    cfg = {
        "instance_connection_name": raw.get("instance_connection_name") or os.getenv("INSTANCE_CONNECTION_NAME"),
        "user": raw.get("user") or os.getenv("DB_USER"),
        "password": raw.get("password") or os.getenv("DB_PASSWORD"),  # None if using IAM auth
        "db": raw.get("db") or os.getenv("DB_NAME"),
        "ip_type": (raw.get("ip_type") or os.getenv("DB_IP_TYPE", "PRIVATE")).upper(),
        "enable_iam_auth": bool(raw.get("enable_iam_auth") or os.getenv("DB_IAM_AUTH", "false").lower() == "true"),
        "application_name": raw.get("application_name") or os.getenv("DB_APP_NAME", "streamlit-app"),
    }
    missing = [k for k in ("instance_connection_name", "user", "db") if not cfg.get(k)]
    if missing:
        raise RuntimeError(f"Missing Cloud SQL config keys: {', '.join(missing)}")
    return cfg


def _engine_key_from_cfg(cfg: Mapping[str, Any]) -> str:
    """Produce a stable cache key for the engine resource."""
    auth_mode = "iam" if cfg.get("enable_iam_auth") else "pwd"
    return f"{cfg['instance_connection_name']}|{cfg['db']}|{cfg['user']}|{cfg.get('ip_type','PRIVATE')}|{auth_mode}"


@st.cache_resource(show_spinner=False)
def _engine_from_cfg_key(engine_key: str, cfg_json: str) -> sa.Engine:
    """
    Cached factory for a tiny SQLAlchemy engine backed by the Cloud SQL Connector.

    NOTE: Cache key = (engine_key, cfg_json). If you rotate secrets or switch DBs,
    the cache will naturally invalidate.
    """
    cfg: dict = json.loads(cfg_json)

    # Prefer explicit SA creds from Streamlit secrets if present; otherwise ADC.
    creds = None
    if "gcp_service_account" in st.secrets:
        creds = service_account.Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"])
        )

    connector = Connector(credentials=creds) if creds is not None else Connector()

    def _getconn():
        return connector.connect(
            cfg["instance_connection_name"],
            "pg8000",
            user=cfg["user"],
            db=cfg["db"],
            password=cfg.get("password"),
            enable_iam_auth=cfg.get("enable_iam_auth", False),
            ip_type=IPTypes.PRIVATE if str(cfg.get("ip_type", "PRIVATE")).upper() == "PRIVATE" else IPTypes.PUBLIC,
            timeout=10,
            application_name=cfg.get("application_name", "streamlit-app"),
        )

    # Keep the pool TINY so this process uses at most 1 slot concurrently.
    pool_size = int(os.getenv("DB_POOL_SIZE", "1"))
    max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "0"))

    engine = sa.create_engine(
        "postgresql+pg8000://",
        creator=_getconn,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,   # transparently recycle stale sockets
        pool_recycle=1800,    # seconds
        pool_timeout=30,      # seconds
        future=True,
    )

    # Keep a handle so we can close the connector at disposal time if needed.
    engine._cloudsql_connector = connector  # type: ignore[attr-defined]
    return engine


def _dispose_all_cached_engines() -> None:
    """Dispose all engines in this process and clear the cache (used by a 'Disconnect' button)."""
    try:
        # st.cache_resource doesn't give handles for each instance; clear() will
        # dispose when objects get GC'd. We also try to close the connector if present.
        _engine_from_cfg_key.clear()
    except Exception:
        pass


# ───────────────────────────────────────────────────────────────
# 2) DatabaseManager – small, explicit API built on the engine
#    Public methods are compatible with your existing code.
# ───────────────────────────────────────────────────────────────

class DatabaseManager:
    """DB helper built on a pooled SQLAlchemy engine (Cloud SQL Connector + pg8000)."""

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None):
        # Prefer env vars (Cloud Run/App Engine), then Streamlit secrets
        if cfg is None:
            if "cloudsql" not in st.secrets:
                raise RuntimeError("st.secrets['cloudsql'] is required when no cfg is provided.")
            cfg_in = dict(st.secrets["cloudsql"])
        else:
            cfg_in = dict(cfg)

        self.cfg = _normalize_cfg(cfg_in)
        self._engine_key = _engine_key_from_cfg(self.cfg)
        self._cfg_json = json.dumps(self.cfg, sort_keys=True)

        self.engine: sa.Engine = _engine_from_cfg_key(self._engine_key, self._cfg_json)

    # ────────── internals ──────────

    @staticmethod
    def _set_timeouts(conn: sa.Connection, statement_timeout_ms: int = 8000, lock_timeout_ms: int = 2000) -> None:
        """Use SET LOCAL so timeouts reset on commit/rollback automatically."""
        conn.exec_driver_sql(f"SET LOCAL statement_timeout = '{int(statement_timeout_ms)}ms';")
        conn.exec_driver_sql(f"SET LOCAL lock_timeout = '{int(lock_timeout_ms)}ms';")
        conn.exec_driver_sql("SET LOCAL idle_in_transaction_session_timeout = '15000ms';")

    def _run_query(self, sql: str, params: Optional[Mapping[str, Any] | Iterable[Any]] = None) -> pd.DataFrame:
        """
        Execute a SELECT (or any rows-returning) statement and return a DataFrame.
        Accepts pg8000-style `%s` placeholders with a tuple/list, or dict with named binds if you use them.
        """
        with self.engine.connect() as conn:
            with conn.begin():
                self._set_timeouts(conn)
                result = conn.exec_driver_sql(sql, params or ())
                if not result.returns_rows:
                    return pd.DataFrame()
                rows = result.fetchall()
                cols = result.keys()
                return pd.DataFrame(rows, columns=cols)

    def _run_command(
        self,
        sql: str,
        params: Optional[Mapping[str, Any] | Iterable[Any]] = None,
        returning: bool = False,
    ):
        """Execute DDL/DML; optionally fetch a single RETURNING row value/record."""
        with self.engine.connect() as conn:
            with conn.begin():
                self._set_timeouts(conn)
                result = conn.exec_driver_sql(sql, params or ())
                if returning:
                    try:
                        return result.fetchone()
                    except Exception:
                        return None
                return result.rowcount if result.rowcount is not None else -1

    # ────────── public API (kept compatible) ──────────

    def fetch_data(self, query: str, params: Optional[Mapping[str, Any] | Iterable[Any]] = None) -> pd.DataFrame:
        return self._run_query(query, params)

    def execute_command(self, query: str, params: Optional[Mapping[str, Any] | Iterable[Any]] = None) -> int:
        return int(self._run_command(query, params, returning=False))

    def execute_command_returning(self, query: str, params: Optional[Mapping[str, Any] | Iterable[Any]] = None):
        return self._run_command(query, params, returning=True)

    # ─────────── Dropdown Management ───────────
    def get_all_sections(self) -> list[str]:
        df = self.fetch_data("SELECT DISTINCT section FROM dropdowns;")
        return df["section"].tolist() if not df.empty else []

    def get_dropdown_values(self, section: str) -> list[str]:
        q = "SELECT value FROM dropdowns WHERE section = %s;"
        df = self.fetch_data(q, (section,))
        return df["value"].tolist() if not df.empty else []

    # ─────────── Supplier Management ───────────
    def get_suppliers(self) -> pd.DataFrame:
        return self.fetch_data("SELECT supplierid, suppliername FROM supplier;")

    # ─────────── Inventory Management ───────────
    def add_inventory(self, data: dict) -> int:
        cols = ", ".join(data.keys())
        ph = ", ".join(["%s"] * len(data))
        q = f"INSERT INTO inventory ({cols}) VALUES ({ph});"
        return self.execute_command(q, list(data.values()))

    # ─────────── Foreign-key checks ───────────
    def check_foreign_key_references(
        self,
        referenced_table: str,
        referenced_column: str,
        value: Any,
    ) -> list[str]:
        """
        Return a list of tables that still reference the given value through a
        FOREIGN KEY constraint. Empty list → safe to delete.
        """
        fk_sql = """
            SELECT tc.table_schema,
                   tc.table_name
            FROM   information_schema.table_constraints AS tc
            JOIN   information_schema.key_column_usage AS kcu
                   ON tc.constraint_name = kcu.constraint_name
            JOIN   information_schema.constraint_column_usage AS ccu
                   ON ccu.constraint_name = tc.constraint_name
            WHERE  tc.constraint_type = 'FOREIGN KEY'
              AND  ccu.table_name      = %s
              AND  ccu.column_name     = %s;
        """
        fks = self.fetch_data(fk_sql, (referenced_table, referenced_column))

        conflicts: list[str] = []
        for _, row in fks.iterrows():
            schema = row["table_schema"]
            table = row["table_name"]

            # NOTE: identifiers here come from information_schema; if you ever pass
            # raw user input, quote identifiers properly.
            exists_sql = f"""
                SELECT EXISTS(
                    SELECT 1
                    FROM   {schema}.{table}
                    WHERE  {referenced_column} = %s
                );
            """
            exists_df = self.fetch_data(exists_sql, (value,))
            exists = bool(exists_df.iat[0, 0]) if not exists_df.empty else False
            if exists:
                conflicts.append(f"{schema}.{table}")

        return sorted(set(conflicts))

    # ─────────── Lifecycle ───────────
    def close(self) -> None:
        """Dispose this process's pooled engine and close the Cloud SQL connector."""
        try:
            connector = getattr(self.engine, "_cloudsql_connector", None)
            self.engine.dispose()
            if connector is not None:
                try:
                    connector.close()
                except Exception:
                    pass
        finally:
            # Also clear the cache so a new engine can be built cleanly next time.
            _dispose_all_cached_engines()
