"""
SQLAlchemy ORM模型定义
映射到PostgreSQL表结构
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Numeric, BigInteger, LargeBinary, Date, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.core.database.connection import Base


class Stock(Base):
    """股票基本信息表"""
    __tablename__ = 'stocks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts_code = Column(String(10), unique=True, nullable=False, index=True, comment='股票代码')
    name = Column(String(50), comment='股票名称')
    industry = Column(String(50), comment='所属行业')
    market = Column(String(10), comment='市场类型(SZ/SH)')
    created_at = Column(DateTime, server_default=func.now(), comment='创建时间')
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment='更新时间')

    # 关系
    prices = relationship("StockPrice", back_populates="stock", cascade="all, delete-orphan")
    models = relationship("Model", back_populates="stock", cascade="all, delete-orphan")
    cache_entries = relationship("DataCache", back_populates="stock", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Stock({self.ts_code} {self.name})>"


class StockPrice(Base):
    """股票价格数据表"""
    __tablename__ = 'stock_prices'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey('stocks.id', ondelete='CASCADE'), nullable=False, comment='股票ID')
    trade_date = Column(Date, nullable=False, comment='交易日期')
    open = Column(Numeric(10, 2), comment='开盘价')
    high = Column(Numeric(10, 2), comment='最高价')
    low = Column(Numeric(10, 2), comment='最低价')
    close = Column(Numeric(10, 2), comment='收盘价')
    pre_close = Column(Numeric(10, 2), comment='昨收价')
    change = Column(Numeric(10, 2), comment='涨跌额')
    pct_chg = Column(Numeric(8, 3), comment='涨跌幅(%)')
    vol = Column(BigInteger, comment='成交量(手)')
    amount = Column(BigInteger, comment='成交额(元)')
    ma5 = Column(Numeric(10, 2), comment='5日均线')
    ma10 = Column(Numeric(10, 2), comment='10日均线')
    v_ma5 = Column(BigInteger, comment='5日成交量均值')
    v_ma10 = Column(BigInteger, comment='10日成交量均值')
    pct_change = Column(Numeric(8, 3), comment='涨跌幅')
    range_val = Column(Numeric(10, 2), comment='振幅')
    created_at = Column(DateTime, server_default=func.now(), comment='创建时间')

    # 关系
    stock = relationship("Stock", back_populates="prices")

    # 唯一约束和索引
    __table_args__ = (
        UniqueConstraint('stock_id', 'trade_date', name='uq_stock_date'),
        Index('idx_prices_stock_date', 'stock_id', 'trade_date'),
        Index('idx_prices_date', 'trade_date'),
    )

    def __repr__(self):
        return f"<StockPrice({self.stock_id} {self.trade_date} close={self.close})>"


class Model(Base):
    """模型元数据表"""
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey('stocks.id', ondelete='CASCADE'), nullable=False, comment='股票ID')
    model_type = Column(String(20), nullable=False, comment='模型类型(baseline/kline_lstm)')
    model_name = Column(String(100), nullable=False, comment='模型唯一标识')
    input_size = Column(Integer, nullable=False, comment='输入维度')
    hidden_size = Column(Integer, nullable=False, comment='隐藏层大小')
    output_size = Column(Integer, nullable=False, comment='输出维度')
    num_layers = Column(Integer, nullable=False, comment='LSTM层数')
    sequence_length = Column(Integer, nullable=False, comment='序列长度')
    training_loss = Column(Numeric(10, 6), comment='训练损失')
    validation_loss = Column(Numeric(10, 6), comment='验证损失')
    epochs = Column(Integer, comment='训练轮数')
    trained_at = Column(DateTime, comment='训练完成时间')
    is_active = Column(Boolean, default=True, comment='是否为激活模型')
    created_at = Column(DateTime, server_default=func.now(), comment='创建时间')
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment='更新时间')

    # 关系
    stock = relationship("Stock", back_populates="models")
    weights = relationship("ModelWeight", back_populates="model", uselist=False, cascade="all, delete-orphan")
    training_history = relationship("TrainingHistory", back_populates="model", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="model", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint('stock_id', 'model_type', 'model_name', name='uq_stock_model'),
        Index('idx_models_stock', 'stock_id'),
        Index('idx_models_active', 'is_active'),
    )

    def __repr__(self):
        return f"<Model({self.model_name} type={self.model_type} active={self.is_active})>"


class ModelWeight(Base):
    """模型权重表"""
    __tablename__ = 'model_weights'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('models.id', ondelete='CASCADE'), nullable=False, unique=True, comment='模型ID')
    weight_data = Column(LargeBinary, nullable=False, comment='PyTorch模型权重(二进制)')
    file_size = Column(BigInteger, comment='文件大小(字节)')
    compression = Column(String(20), default='gzip', comment='压缩方式')
    created_at = Column(DateTime, server_default=func.now(), comment='创建时间')

    # 关系
    model = relationship("Model", back_populates="weights")

    __table_args__ = (
        Index('idx_weights_model', 'model_id'),
    )

    def __repr__(self):
        return f"<ModelWeight(model_id={self.model_id} size={self.file_size})>"


class TrainingHistory(Base):
    """训练历史表"""
    __tablename__ = 'training_history'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('models.id', ondelete='CASCADE'), nullable=False, comment='模型ID')
    epoch = Column(Integer, nullable=False, comment='训练轮数')
    train_loss = Column(Numeric(10, 6), comment='训练损失')
    val_loss = Column(Numeric(10, 6), comment='验证损失')
    learning_rate = Column(Numeric(10, 8), comment='学习率')
    timestamp = Column(DateTime, server_default=func.now(), comment='时间戳')

    # 关系
    model = relationship("Model", back_populates="training_history")

    __table_args__ = (
        Index('idx_history_model', 'model_id', 'epoch'),
    )

    def __repr__(self):
        return f"<TrainingHistory(model_id={self.model_id} epoch={self.epoch})>"


class Prediction(Base):
    """预测结果表"""
    __tablename__ = 'predictions'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('models.id', ondelete='CASCADE'), nullable=False, comment='模型ID')
    stock_id = Column(Integer, ForeignKey('stocks.id', ondelete='CASCADE'), nullable=False, comment='股票ID')
    prediction_date = Column(Date, nullable=False, comment='预测日期')
    predicted_price = Column(Numeric(10, 2), comment='预测价格')
    confidence = Column(Numeric(5, 4), comment='置信度')
    actual_price = Column(Numeric(10, 2), comment='实际价格')
    error_rate = Column(Numeric(8, 4), comment='误差率')
    created_at = Column(DateTime, server_default=func.now(), comment='创建时间')

    # 关系
    model = relationship("Model", back_populates="predictions")

    __table_args__ = (
        Index('idx_predictions_stock_date', 'stock_id', 'prediction_date'),
        Index('idx_predictions_model', 'model_id'),
    )

    def __repr__(self):
        return f"<Prediction(stock_id={self.stock_id} date={self.prediction_date} price={self.predicted_price})>"


class DataCache(Base):
    """数据缓存表"""
    __tablename__ = 'data_cache'

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey('stocks.id', ondelete='CASCADE'), nullable=False, comment='股票ID')
    start_date = Column(Date, nullable=False, comment='起始日期')
    end_date = Column(Date, nullable=False, comment='结束日期')
    source = Column(String(50), default='tushare_api', comment='数据源')
    row_count = Column(Integer, comment='行数')
    file_size = Column(BigInteger, comment='文件大小')
    expires_at = Column(DateTime, nullable=False, comment='过期时间')
    created_at = Column(DateTime, server_default=func.now(), comment='创建时间')

    # 关系
    stock = relationship("Stock", back_populates="cache_entries")

    __table_args__ = (
        UniqueConstraint('stock_id', 'start_date', 'end_date', name='uq_cache_range'),
        Index('idx_cache_stock', 'stock_id'),
        Index('idx_cache_expires', 'expires_at'),
    )

    def __repr__(self):
        return f"<DataCache(stock_id={self.stock_id} {self.start_date}-{self.end_date})>"
