"""验证数据库迁移"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from backend.core.database.connection import get_db_session
from backend.core.database.models import Stock, StockPrice, Model

with get_db_session() as session:
    stocks = session.query(Stock).all()
    print(f"股票数量: {len(stocks)}")

    prices = session.query(StockPrice).all()
    print(f"价格记录总数: {len(prices)}")

    models = session.query(Model).all()
    print(f"模型数量: {len(models)}")

    print("\n股票明细:")
    for stock in stocks:
        print(f"  - {stock.ts_code}: {len(stock.prices)} 条价格记录")

    print("\n模型明细:")
    for model in models:
        print(f"  - {model.model_name}: type={model.model_type}, active={model.is_active}")
