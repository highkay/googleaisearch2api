from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class ServiceConfigRow(Base):
    __tablename__ = "service_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    default_model: Mapped[str] = mapped_column(String(128))
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
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class RequestLogRow(Base):
    __tablename__ = "request_logs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    endpoint: Mapped[str] = mapped_column(String(64))
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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


def create_db_engine(db_path: str):
    return create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        future=True,
    )


def create_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def create_tables(engine) -> None:
    Base.metadata.create_all(engine)
    _ensure_service_config_columns(engine)


def _ensure_service_config_columns(engine) -> None:
    _ensure_column(engine, "service_config", "browser_user_agent", "TEXT")


def _ensure_column(engine, table_name: str, column_name: str, column_sql: str) -> None:
    with engine.begin() as connection:
        rows = connection.exec_driver_sql(f"PRAGMA table_info({table_name})").all()
        existing = {row[1] for row in rows}
        if column_name not in existing:
            connection.exec_driver_sql(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"
            )
