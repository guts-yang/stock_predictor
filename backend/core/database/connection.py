"""
数据库连接管理器
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from backend.core.config import (
    DATABASE_URL, SQLALCHEMY_ECHO, SQLALCHEMY_POOL_SIZE,
    SQLALCHEMY_MAX_OVERFLOW, SQLALCHEMY_POOL_RECYCLE, SQLALCHEMY_POOL_PRE_PING
)

# 创建数据库引擎
engine = create_engine(
    DATABASE_URL,
    echo=SQLALCHEMY_ECHO,
    poolclass=QueuePool,
    pool_size=SQLALCHEMY_POOL_SIZE,
    max_overflow=SQLALCHEMY_MAX_OVERFLOW,
    pool_recycle=SQLALCHEMY_POOL_RECYCLE,
    pool_pre_ping=SQLALCHEMY_POOL_PRE_PING,
    pool_use_lifo=True,  # 使用LIFO策略减少连接数
)

# 创建Session工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建Base类
Base = declarative_base()


@contextmanager
def get_db_session():
    """
    获取数据库会话的上下文管理器
    用法:
        with get_db_session() as session:
            session.query(Stock).all()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db():
    """
    获取数据库会话（用于FastAPI/Flask依赖注入）
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_database():
    """
    初始化数据库，创建所有表
    """
    from backend.core.database.models import Stock, StockPrice, Model, ModelWeight, TrainingHistory, Prediction, DataCache
    Base.metadata.create_all(bind=engine)
    print("✅ 数据库表创建成功")


def drop_database():
    """
    删除所有表（谨慎使用）
    """
    Base.metadata.drop_all(bind=engine)
    print("⚠️  所有数据库表已删除")
