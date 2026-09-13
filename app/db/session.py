"""
Database connection engine, session makers, and dependency providers.
Supports asynchronous FastAPI requests and synchronous background workers.
"""

import logging
from typing import AsyncGenerator, Generator
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)

# Async Engine for FastAPI Gateway
async_database_url = settings.get_database_url()
async_engine = create_async_engine(
    async_database_url,
    echo=False,
    pool_pre_ping=True,
    future=True,
)
async_session_factory = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Sync Engine for Celery Worker & CLI
sync_database_url = settings.get_database_sync_url()
sync_engine = create_engine(
    sync_database_url,
    echo=False,
    pool_pre_ping=True,
    future=True,
)
sync_session_factory = sessionmaker(
    bind=sync_engine,
    class_=Session,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_db() -> Generator[Session, None, None]:
    """Context generator yielding a synchronous database session."""
    session = sync_session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
