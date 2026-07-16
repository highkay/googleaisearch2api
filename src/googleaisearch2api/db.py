from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class ServiceConfigRow(Base):
    __tablename__ = "service_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    default_model: Mapped[str] = mapped_column(String(128))
    search_engine: Mapped[str] = mapped_column(String(32), default="google")
    api_token: Mapped[str] = mapped_column(String(512))
    browser_channel: Mapped[str | None] = mapped_column(String(64), nullable=True)
    browser_executable_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    browser_headless: Mapped[bool] = mapped_column(Boolean, default=True)
    browser_user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    browser_locale: Mapped[str] = mapped_column(String(32))
    browser_base_url: Mapped[str] = mapped_column(Text)
    browser_timeout_ms: Mapped[int] = mapped_column(Integer, default=90_000)
    answer_timeout_ms: Mapped[int] = mapped_column(Integer, default=45_000)
    browser_proxy_server: Mapped[str | None] = mapped_column(Text, nullable=True)
    browser_proxy_username: Mapped[str | None] = mapped_column(Text, nullable=True)
    browser_proxy_password: Mapped[str | None] = mapped_column(Text, nullable=True)
    browser_proxy_bypass: Mapped[str | None] = mapped_column(Text, nullable=True)
    resin_sticky_session_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class RequestLogRow(Base):
    __tablename__ = "request_logs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    endpoint: Mapped[str] = mapped_column(String(64))
    engine: Mapped[str] = mapped_column(String(32), default="google")
    status: Mapped[str] = mapped_column(String(32), default="pending")
    model_name: Mapped[str] = mapped_column(String(128))
    prompt_preview: Mapped[str] = mapped_column(Text, default="")
    response_preview: Mapped[str] = mapped_column(Text, default="")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    citations_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    client_ip: Mapped[str | None] = mapped_column(String(128), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stream: Mapped[bool] = mapped_column(Boolean, default=False)
    headless: Mapped[bool] = mapped_column(Boolean, default=True)
    proxy_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    resin_sticky_session_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    proxy_session_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    proxy_base_username: Mapped[str | None] = mapped_column(Text, nullable=True)
    proxy_username: Mapped[str | None] = mapped_column(Text, nullable=True)
    proxy_primary_ip: Mapped[str | None] = mapped_column(String(128), nullable=True)
    proxy_ip_vector_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    proxy_iplark_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    google_block_ips_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    google_block_mismatch: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ProxySessionRow(Base):
    __tablename__ = "proxy_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    proxy_base_username: Mapped[str] = mapped_column(Text, nullable=False)
    session_name: Mapped[str] = mapped_column(String(128), nullable=False)
    proxy_username: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="new")
    epoch: Mapped[int] = mapped_column(Integer, default=0)
    primary_ip: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ip_vector_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip_vector_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    asn: Mapped[str | None] = mapped_column(String(64), nullable=True)
    organization: Mapped[str | None] = mapped_column(Text, nullable=True)
    iplark_min_quality_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    iplark_usage_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    iplark_category: Mapped[str | None] = mapped_column(String(128), nullable=True)
    iplark_public_proxy: Mapped[bool] = mapped_column(Boolean, default=False)
    iplark_threat: Mapped[bool] = mapped_column(Boolean, default=False)
    iplark_tag: Mapped[str | None] = mapped_column(Text, nullable=True)
    google_canary_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    google_canary_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    google_canary_checked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    duck_canary_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    duck_canary_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duck_canary_checked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    request_success_count: Mapped[int] = mapped_column(Integer, default=0)
    request_block_count: Mapped[int] = mapped_column(Integer, default=0)
    request_error_count: Mapped[int] = mapped_column(Integer, default=0)
    canary_success_count: Mapped[int] = mapped_column(Integer, default=0)
    canary_block_count: Mapped[int] = mapped_column(Integer, default=0)
    duck_canary_success_count: Mapped[int] = mapped_column(Integer, default=0)
    duck_canary_rate_limit_count: Mapped[int] = mapped_column(Integer, default=0)
    duck_canary_error_count: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_of_session_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_selected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_blocked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cooldown_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retire_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ProxyIpObservationRow(Base):
    __tablename__ = "proxy_ip_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    proxy_session_id: Mapped[int] = mapped_column(Integer, nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    ip: Mapped[str] = mapped_column(String(128), nullable=False)
    asn: Mapped[str | None] = mapped_column(String(64), nullable=True)
    organization: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ProxySessionEventRow(Base):
    __tablename__ = "proxy_session_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    proxy_session_id: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str] = mapped_column(Text, default="")
    raw_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


def create_db_engine(db_path: str):
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 30},
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=30000")
        finally:
            cursor.close()

    return engine


def create_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def create_tables(engine) -> None:
    Base.metadata.create_all(engine)
    _ensure_service_config_columns(engine)
    _ensure_request_log_columns(engine)
    _ensure_proxy_session_columns(engine)


def _ensure_service_config_columns(engine) -> None:
    _ensure_column(engine, "service_config", "search_engine", "VARCHAR(32) DEFAULT 'google'")
    _ensure_column(engine, "service_config", "browser_user_agent", "TEXT")
    _ensure_column(engine, "service_config", "resin_sticky_session_enabled", "BOOLEAN DEFAULT 0")


def _ensure_request_log_columns(engine) -> None:
    _ensure_column(engine, "request_logs", "engine", "VARCHAR(32) DEFAULT 'google'")
    _ensure_column(engine, "request_logs", "resin_sticky_session_enabled", "BOOLEAN DEFAULT 0")
    _ensure_column(engine, "request_logs", "proxy_session_id", "INTEGER")
    _ensure_column(engine, "request_logs", "proxy_base_username", "TEXT")
    _ensure_column(engine, "request_logs", "proxy_username", "TEXT")
    _ensure_column(engine, "request_logs", "proxy_primary_ip", "VARCHAR(128)")
    _ensure_column(engine, "request_logs", "proxy_ip_vector_hash", "VARCHAR(64)")
    _ensure_column(engine, "request_logs", "proxy_iplark_score", "INTEGER")
    _ensure_column(engine, "request_logs", "google_block_ips_json", "TEXT")
    _ensure_column(engine, "request_logs", "google_block_mismatch", "BOOLEAN DEFAULT 0")


def _ensure_proxy_session_columns(engine) -> None:
    _ensure_column(engine, "proxy_sessions", "duck_canary_status", "VARCHAR(32)")
    _ensure_column(engine, "proxy_sessions", "duck_canary_error", "TEXT")
    _ensure_column(engine, "proxy_sessions", "duck_canary_checked_at", "DATETIME")
    _ensure_column(engine, "proxy_sessions", "duck_canary_success_count", "INTEGER DEFAULT 0")
    _ensure_column(engine, "proxy_sessions", "duck_canary_rate_limit_count", "INTEGER DEFAULT 0")
    _ensure_column(engine, "proxy_sessions", "duck_canary_error_count", "INTEGER DEFAULT 0")


def _ensure_column(engine, table_name: str, column_name: str, column_sql: str) -> None:
    with engine.begin() as connection:
        rows = connection.exec_driver_sql(f"PRAGMA table_info({table_name})").all()
        existing = {row[1] for row in rows}
        if column_name not in existing:
            connection.exec_driver_sql(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"
            )
