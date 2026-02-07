"""
数据访问层（Repository Pattern）
封装所有数据库操作，提供统一的数据访问接口
"""
import io
import lz4.frame
import gzip
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from backend.core.database.models import Stock, StockPrice, Model, ModelWeight, Prediction, DataCache
from backend.core.config import MODEL_WEIGHTS_COMPRESS, COMPRESSION_ALGORITHM


class StockRepository:
    """股票数据Repository"""

    @staticmethod
    def get_or_create_stock(session: Session, ts_code: str, name: str = None) -> Stock:
        """获取或创建股票记录"""
        stock = session.query(Stock).filter(Stock.ts_code == ts_code).first()
        if not stock:
            market = ts_code.split('.')[1] if '.' in ts_code else None
            stock = Stock(ts_code=ts_code, name=name, market=market)
            session.add(stock)
            session.flush()
        return stock

    @staticmethod
    def save_stock_data(session: Session, ts_code: str, df: pd.DataFrame) -> Tuple[Stock, int]:
        """
        保存股票价格数据到数据库
        返回: (Stock对象, 插入的记录数)
        """
        # 获取或创建股票记录
        stock = StockRepository.get_or_create_stock(session, ts_code)

        # 先删除该股票的所有历史数据（避免重复）
        session.query(StockPrice).filter(StockPrice.stock_id == stock.id).delete()

        # 去重：按trade_date去重，保留第一条记录
        print(f"    原始记录数: {len(df)}", end="")
        df = df.drop_duplicates(subset=['trade_date'], keep='first')
        print(f", 去重后: {len(df)}")

        # 转换DataFrame为列表并批量插入
        count = 0
        for _, row in df.iterrows():
            try:
                # 明确指定日期格式为YYYYMMDD
                trade_date = pd.to_datetime(str(row['trade_date']), format='%Y%m%d').date()
                price = StockPrice(
                    stock_id=stock.id,
                    trade_date=trade_date,
                    open=float(row.get('open', 0)) if pd.notna(row.get('open')) else 0,
                    high=float(row.get('high', 0)) if pd.notna(row.get('high')) else 0,
                    low=float(row.get('low', 0)) if pd.notna(row.get('low')) else 0,
                    close=float(row.get('close', 0)) if pd.notna(row.get('close')) else 0,
                    pre_close=float(row.get('pre_close', 0)) if pd.notna(row.get('pre_close')) else 0,
                    change=float(row.get('change', 0)) if pd.notna(row.get('change')) else 0,
                    pct_chg=float(row.get('pct_chg', 0)) if pd.notna(row.get('pct_chg')) else 0,
                    vol=int(row.get('vol', 0)) if pd.notna(row.get('vol')) else 0,
                    amount=int(row.get('amount', 0)) if pd.notna(row.get('amount')) else 0,
                    ma5=float(row.get('ma5', 0)) if pd.notna(row.get('ma5')) else 0,
                    ma10=float(row.get('ma10', 0)) if pd.notna(row.get('ma10')) else 0,
                    v_ma5=int(row.get('v_ma5', 0)) if pd.notna(row.get('v_ma5')) else 0,
                    v_ma10=int(row.get('v_ma10', 0)) if pd.notna(row.get('v_ma10')) else 0,
                    pct_change=float(row.get('pct_change', 0)) if pd.notna(row.get('pct_change')) else 0,
                    range_val=float(row.get('range', 0)) if pd.notna(row.get('range')) else 0
                )
                session.add(price)
                count += 1
            except Exception as e:
                print(f"    ⚠️  跳过异常记录: {row.get('trade_date')} - {e}")
                continue

        session.commit()
        return stock, count

    @staticmethod
    def get_stock_data(session: Session, ts_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        从数据库获取股票数据
        返回: DataFrame
        """
        stock = session.query(Stock).filter(Stock.ts_code == ts_code).first()
        if not stock:
            return pd.DataFrame()

        query = session.query(StockPrice).filter(StockPrice.stock_id == stock.id)

        if start_date:
            query = query.filter(StockPrice.trade_date >= pd.to_datetime(start_date).date())
        if end_date:
            query = query.filter(StockPrice.trade_date <= pd.to_datetime(end_date).date())

        query = query.order_by(StockPrice.trade_date.asc())

        # 转换为DataFrame
        prices = query.all()
        if not prices:
            return pd.DataFrame()

        data = []
        for price in prices:
            data.append({
                'ts_code': ts_code,
                'trade_date': price.trade_date.strftime('%Y%m%d'),
                'open': float(price.open) if price.open else 0,
                'high': float(price.high) if price.high else 0,
                'low': float(price.low) if price.low else 0,
                'close': float(price.close) if price.close else 0,
                'pre_close': float(price.pre_close) if price.pre_close else 0,
                'change': float(price.change) if price.change else 0,
                'pct_chg': float(price.pct_chg) if price.pct_chg else 0,
                'vol': int(price.vol) if price.vol else 0,
                'amount': int(price.amount) if price.amount else 0,
                'ma5': float(price.ma5) if price.ma5 else 0,
                'ma10': float(price.ma10) if price.ma10 else 0,
                'v_ma5': int(price.v_ma5) if price.v_ma5 else 0,
                'v_ma10': int(price.v_ma10) if price.v_ma10 else 0,
                'pct_change': float(price.pct_change) if price.pct_change else 0,
                'range': float(price.range_val) if price.range_val else 0,
            })

        return pd.DataFrame(data)

    @staticmethod
    def check_cache_exists(session: Session, ts_code: str, start_date: str, end_date: str) -> bool:
        """检查缓存是否存在且未过期"""
        stock = session.query(Stock).filter(Stock.ts_code == ts_code).first()
        if not stock:
            return False

        cache = session.query(DataCache).filter(
            and_(
                DataCache.stock_id == stock.id,
                DataCache.start_date == pd.to_datetime(start_date).date(),
                DataCache.end_date == pd.to_datetime(end_date).date(),
                DataCache.expires_at > datetime.now()
            )
        ).first()

        return cache is not None

    @staticmethod
    def update_cache(session: Session, ts_code: str, start_date: str, end_date: str, ttl_days: int = 7):
        """更新缓存记录"""
        stock = session.query(Stock).filter(Stock.ts_code == ts_code).first()
        if not stock:
            return

        # 删除旧缓存
        session.query(DataCache).filter(
            and_(
                DataCache.stock_id == stock.id,
                DataCache.start_date == pd.to_datetime(start_date).date(),
                DataCache.end_date == pd.to_datetime(end_date).date()
            )
        ).delete()

        # 创建新缓存
        expires_at = datetime.now() + timedelta(days=ttl_days)
        cache = DataCache(
            stock_id=stock.id,
            start_date=pd.to_datetime(start_date).date(),
            end_date=pd.to_datetime(end_date).date(),
            source='tushare_api',
            expires_at=expires_at
        )
        session.add(cache)
        session.commit()


class ModelRepository:
    """模型数据Repository"""

    @staticmethod
    def save_model(session: Session, ts_code: str, model, model_metadata: dict) -> Model:
        """
        保存模型到数据库
        model_metadata应包含:
        - model_type: baseline/kline_lstm
        - input_size, hidden_size, output_size, num_layers, sequence_length
        - training_loss, validation_loss, epochs
        """
        import torch
        from backend.core.database.repositories import StockRepository

        stock = StockRepository.get_or_create_stock(session, ts_code)

        # 创建模型元数据记录
        db_model = Model(
            stock_id=stock.id,
            model_type=model_metadata['model_type'],
            model_name=f"{ts_code}_{model_metadata['model_type']}_best",
            input_size=model_metadata['input_size'],
            hidden_size=model_metadata['hidden_size'],
            output_size=model_metadata['output_size'],
            num_layers=model_metadata['num_layers'],
            sequence_length=model_metadata['sequence_length'],
            training_loss=model_metadata.get('training_loss'),
            validation_loss=model_metadata.get('validation_loss'),
            epochs=model_metadata.get('epochs'),
            trained_at=datetime.now(),
            is_active=True
        )

        # 添加到session并flush以获取ID
        session.add(db_model)
        session.flush()

        # 序列化模型权重
        buffer = io.BytesIO()
        torch.save({
            'model_state_dict': model.state_dict(),
            'ts_code': ts_code,
            **model_metadata
        }, buffer)
        weight_data = buffer.getvalue()

        # 压缩权重数据
        if MODEL_WEIGHTS_COMPRESS:
            if COMPRESSION_ALGORITHM == 'lz4':
                compressed_data = lz4.frame.compress(weight_data)
            else:  # gzip
                compressed_data = gzip.compress(weight_data)
            weight_data = compressed_data

        # 创建权重记录
        model_weight = ModelWeight(
            model_id=db_model.id,
            weight_data=weight_data,
            file_size=len(weight_data),
            compression=COMPRESSION_ALGORITHM if MODEL_WEIGHTS_COMPRESS else None
        )

        session.add(model_weight)
        session.commit()

        return db_model

    @staticmethod
    def load_model(session: Session, ts_code: str, model_type: str = 'baseline'):
        """
        从数据库加载模型
        返回: (checkpoint, db_model)
        """
        import torch
        from backend.core.database.repositories import StockRepository

        stock = session.query(Stock).filter(Stock.ts_code == ts_code).first()
        if not stock:
            return None, None

        # 查找激活的模型
        db_model = session.query(Model).filter(
            and_(
                Model.stock_id == stock.id,
                Model.model_type == model_type,
                Model.is_active == True
            )
        ).first()

        if not db_model or not db_model.weights:
            return None, None

        # 解压权重数据
        weight_data = db_model.weights.weight_data
        if db_model.weights.compression:
            if db_model.weights.compression == 'lz4':
                weight_data = lz4.frame.decompress(weight_data)
            else:  # gzip
                weight_data = gzip.decompress(weight_data)

        # 反序列化模型
        buffer = io.BytesIO(weight_data)
        checkpoint = torch.load(buffer)

        return checkpoint, db_model

    @staticmethod
    def get_active_model(session: Session, ts_code: str, model_type: str = 'baseline') -> Optional[Model]:
        """获取激活的模型元数据"""
        stock = session.query(Stock).filter(Stock.ts_code == ts_code).first()
        if not stock:
            return None

        return session.query(Model).filter(
            and_(
                Model.stock_id == stock.id,
                Model.model_type == model_type,
                Model.is_active == True
            )
        ).first()

    @staticmethod
    def deactivate_old_models(session: Session, ts_code: str, model_type: str):
        """停用同一类型的旧模型"""
        from backend.core.database.repositories import StockRepository
        stock = StockRepository.get_or_create_stock(session, ts_code)
        if not stock:
            return

        session.query(Model).filter(
            and_(
                Model.stock_id == stock.id,
                Model.model_type == model_type,
                Model.is_active == True
            )
        ).update({'is_active': False})
        session.commit()


class PredictionRepository:
    """预测结果Repository"""

    @staticmethod
    def save_prediction(session: Session, ts_code: str, model_id: int, prediction_date: str,
                       predicted_price: float, confidence: float = None):
        """保存预测结果"""
        from backend.core.database.repositories import StockRepository
        stock = StockRepository.get_or_create_stock(session, ts_code)

        prediction = Prediction(
            model_id=model_id,
            stock_id=stock.id,
            prediction_date=pd.to_datetime(prediction_date).date(),
            predicted_price=predicted_price,
            confidence=confidence
        )
        session.add(prediction)
        session.commit()

    @staticmethod
    def get_predictions(session: Session, ts_code: str, limit: int = 10) -> List[Prediction]:
        """获取最近的预测结果"""
        stock = session.query(Stock).filter(Stock.ts_code == ts_code).first()
        if not stock:
            return []

        return session.query(Prediction).filter(
            Prediction.stock_id == stock.id
        ).order_by(desc(Prediction.prediction_date)).limit(limit).all()
