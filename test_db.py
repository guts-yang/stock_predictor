"""测试数据库功能"""
import sys
import os

# Windows UTF-8 encoding fix
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(__file__))

from backend.core.database.repositories import StockRepository
from backend.core.database.connection import get_db_session
from backend.core.config import USE_DATABASE

print(f"USE_DATABASE = {USE_DATABASE}")

if not USE_DATABASE:
    print("错误: 数据库模式未启用")
    sys.exit(1)

# 测试数据获取
with get_db_session() as session:
    df = StockRepository.get_stock_data(session, '000001.SZ')
    print(f"\n从数据库获取了 {len(df)} 条记录 (000001.SZ)")

    if not df.empty:
        print("\n前3条记录:")
        print(df.head(3).to_string())
        print("\n✅ 数据库功能正常!")
    else:
        print("\n❌ 未获取到数据")
