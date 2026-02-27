import logging
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker


logger = logging.getLogger("leaflens.database")

# SQLite database URL; can be swapped for PostgreSQL in production, e.g.:
# postgresql+psycopg2://user:password@host:port/dbname
DATABASE_URL: str = "sqlite:///./leaflens.db"

# Engine and session factory are created once at import time.
# No global sessions are created or shared; callers must use SessionLocal().
engine = create_engine(
	DATABASE_URL,
	connect_args={"check_same_thread": False},
	future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
	"""Yield a database session and ensure proper cleanup.

	This is intended for use as a FastAPI dependency:

		Depends(get_db)
	"""
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()


def init_db() -> None:
	"""Initialize the database schema in an idempotent way.

	This function imports all ORM models so that they are registered with
	SQLAlchemy's metadata, then creates any missing tables.
	It is safe to call at application startup.
	"""
	try:
		# Import models so that they are registered with the Base metadata
		import database.models  # noqa: F401

		Base.metadata.create_all(bind=engine)
		logger.info("Database initialized; all tables created if not existing.")
	except Exception:
		logger.exception("Failed to initialize database.")
		raise

