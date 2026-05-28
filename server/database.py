import os
import shutil
import tempfile
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOCAL_SQLITE_PATH = os.path.join(DATA_DIR, "nongsan_v2.sqlite3")


def mask_db_url(url: str) -> str:
    if not url:
        return ""
    if "@" not in url:
        return url
    try:
        scheme, rest = url.split("://", 1)
        auth, host = rest.split("@", 1)
        if ":" in auth:
            user, _ = auth.split(":", 1)
            return f"{scheme}://{user}:***@{host}"
        return f"{scheme}://***@{host}"
    except Exception:
        return "masked_db_url"


def _normalize_database_url(url: str) -> str:
    """Make common hosted Postgres URLs work with SQLAlchemy."""
    normalized = url.strip()
    if normalized.startswith("postgres://"):
        normalized = "postgresql+psycopg://" + normalized[len("postgres://") :]
    elif normalized.startswith("postgresql://"):
        normalized = "postgresql+psycopg://" + normalized[len("postgresql://") :]

    parts = urlsplit(normalized)
    if "supabase" in parts.hostname.lower() if parts.hostname else False:
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query.setdefault("sslmode", "require")
        normalized = urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
        )
    return normalized


def _fallback_sqlite_url() -> str:
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=DATA_DIR, delete=True, encoding="utf-8") as f:
            f.write("test")
        return f"sqlite:///{LOCAL_SQLITE_PATH}"
    except Exception as exc:
        print(f"[DB] Warning: local data directory is not writable: {exc}")
        tmp_path = os.path.join(tempfile.gettempdir(), "nongsan_v2.sqlite3")
        if os.path.exists(LOCAL_SQLITE_PATH) and not os.path.exists(tmp_path):
            try:
                shutil.copy2(LOCAL_SQLITE_PATH, tmp_path)
                os.chmod(tmp_path, 0o666)
                print(f"[DB] Pre-filled SQLite database copied to {tmp_path}")
            except Exception as copy_exc:
                print(f"[DB] Error: failed to copy pre-filled database: {copy_exc}")
        return f"sqlite:///{tmp_path}"


def get_database_url() -> str:
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        final_url = _normalize_database_url(env_url)
        print(f"[DB] Using database URL from environment: {mask_db_url(final_url)}")
        return final_url
    return _fallback_sqlite_url()


def create_engine_kwargs(database_url: str) -> dict:
    kwargs = {"pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = int(os.getenv("DB_POOL_SIZE", "5"))
        kwargs["max_overflow"] = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    return kwargs


SQLALCHEMY_DATABASE_URL = get_database_url()
print(f"[DB] Final Database URL: {mask_db_url(SQLALCHEMY_DATABASE_URL)}")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    **create_engine_kwargs(SQLALCHEMY_DATABASE_URL),
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
