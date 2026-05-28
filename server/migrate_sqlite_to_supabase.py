import os
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from database import _normalize_database_url, create_engine_kwargs
from models import Base, ChatHistory, SearchHistory, User


SERVER_DIR = Path(__file__).resolve().parent
DEFAULT_SQLITE_PATH = SERVER_DIR / "data" / "nongsan_v2.sqlite3"
TABLES = (User, SearchHistory, ChatHistory)


def _sqlite_url() -> str:
    raw = os.getenv("SQLITE_DATABASE_URL")
    if raw:
        return raw
    return f"sqlite:///{DEFAULT_SQLITE_PATH}"


def _target_url() -> str:
    raw = os.getenv("SUPABASE_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not raw:
        raise RuntimeError("Set SUPABASE_DATABASE_URL to your Supabase Postgres connection string.")
    normalized = _normalize_database_url(raw)
    if normalized.startswith("sqlite"):
        raise RuntimeError("Target database must be a Supabase/Postgres URL, not SQLite.")
    return normalized


def _row_data(row) -> dict:
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}


def _copy_table(source_db, target_db, model) -> int:
    copied = 0
    for row in source_db.query(model).order_by(model.id).all():
        if target_db.get(model, row.id):
            continue
        target_db.add(model(**_row_data(row)))
        copied += 1
    target_db.commit()
    return copied


def _reset_postgres_sequences(target_db) -> None:
    for model in TABLES:
        table_name = model.__tablename__
        target_db.execute(
            text(
                "SELECT setval("
                "pg_get_serial_sequence(:table_name, 'id'), "
                f"COALESCE((SELECT MAX(id) FROM {table_name}), 1), "
                f"(SELECT COUNT(*) > 0 FROM {table_name})"
                ")"
            ),
            {"table_name": table_name},
        )
    target_db.commit()


def main() -> None:
    source_engine = create_engine(_sqlite_url(), connect_args={"check_same_thread": False})
    target_url = _target_url()
    target_engine = create_engine(target_url, **create_engine_kwargs(target_url))

    Base.metadata.create_all(bind=target_engine)

    SourceSession = sessionmaker(bind=source_engine)
    TargetSession = sessionmaker(bind=target_engine)
    source_db = SourceSession()
    target_db = TargetSession()
    try:
        for model in TABLES:
            copied = _copy_table(source_db, target_db, model)
            print(f"{model.__tablename__}: copied {copied} rows")
        _reset_postgres_sequences(target_db)
        print("SQLite data migration to Supabase completed.")
    except Exception:
        target_db.rollback()
        raise
    finally:
        source_db.close()
        target_db.close()


if __name__ == "__main__":
    main()
