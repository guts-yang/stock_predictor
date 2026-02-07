"""
数据库初始化脚本
用于创建数据库、表结构，并迁移现有数据
"""
import os
import sys
from sqlalchemy import text

# 设置UTF-8编码（Windows兼容）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(__file__))

from backend.core.database.connection import engine, init_database, get_db_session
from backend.core.database.models import Stock, StockPrice, Model, ModelWeight
from backend.core.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def create_database():
    """创建数据库（如果不存在）"""
    # 先连接到默认postgres数据库
    default_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/postgres"
    from sqlalchemy import create_engine
    default_engine = create_engine(default_url)

    # 检查数据库是否存在
    with default_engine.connect() as conn:
        conn.execute(text("COMMIT"))
        result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname='{DB_NAME}'"))
        exists = result.fetchone() is not None

        if not exists:
            conn.execute(text(f"CREATE DATABASE {DB_NAME} ENCODING 'UTF8'"))
            print(f"✅ 数据库 '{DB_NAME}' 创建成功")
        else:
            print(f"ℹ️  数据库 '{DB_NAME}' 已存在")

    default_engine.dispose()


def migrate_from_csv():
    """
    从CSV文件迁移数据到数据库
    """
    import pandas as pd
    from pathlib import Path

    print("\n📦 开始迁移CSV数据...")

    data_dir = Path('data')
    if not data_dir.exists():
        print("⚠️  data/目录不存在，跳过迁移")
        return

    csv_files = list(data_dir.glob('*.csv'))
    if not csv_files:
        print("⚠️  未找到CSV文件，跳过迁移")
        return

    migrated_count = 0

    for csv_file in csv_files:
        # 为每个文件使用独立的session，避免事务级联失败
        try:
            with get_db_session() as session:
                # 解析文件名获取股票代码
                # 文件名格式: stock_000001.SZ_20240101_20241201.csv
                parts = csv_file.stem.split('_')
                if len(parts) >= 2 and parts[0] == 'stock':
                    ts_code = parts[1]
                else:
                    # 另一种格式: 000001.SZ_20240101_20241201.csv
                    ts_code = parts[0]

                print(f"  迁移: {csv_file.name} -> {ts_code}")

                # 读取CSV
                df = pd.read_csv(csv_file, encoding='utf-8-sig')

                # 保存到数据库
                from backend.core.database.repositories import StockRepository
                stock, count = StockRepository.save_stock_data(session, ts_code, df)
                migrated_count += count

                print(f"    ✅ 插入 {count} 条记录")

        except Exception as e:
            print(f"    ❌ 错误: {e}")
            continue

    print(f"\n✅ 迁移完成，共 {migrated_count} 条记录")


def migrate_models():
    """
    迁移PyTorch模型文件到数据库
    """
    import torch
    from pathlib import Path

    print("\n🤖 开始迁移模型文件...")

    models_dir = Path('models')
    if not models_dir.exists():
        print("⚠️  models/目录不存在，跳过迁移")
        return

    model_files = list(models_dir.glob('*.pth'))
    if not model_files:
        print("⚠️  未找到模型文件，跳过迁移")
        return

    with get_db_session() as session:
        migrated_count = 0

        for model_file in model_files:
            try:
                # 解析文件名
                # 格式: 000001.SZ_baseline_best.pth
                parts = model_file.stem.split('_')
                if len(parts) >= 3:
                    ts_code = parts[0]
                    model_type = parts[1]

                    print(f"  迁移: {model_file.name}")

                    # 加载模型
                    checkpoint = torch.load(model_file, map_location='cpu')

                    # 提取元数据
                    metadata = {
                        'model_type': model_type,
                        'input_size': checkpoint.get('input_size', 1),
                        'hidden_size': checkpoint.get('hidden_size', 64),
                        'output_size': checkpoint.get('output_size', 1),
                        'num_layers': checkpoint.get('num_layers', 2),
                        'sequence_length': checkpoint.get('sequence_length', 10),
                    }

                    # 重建模型实例
                    from backend.core.stock_model import get_model
                    model = get_model(
                        model_type=metadata['model_type'],
                        input_size=metadata['input_size'],
                        hidden_size=metadata['hidden_size'],
                        output_size=metadata['output_size'],
                        num_layers=metadata['num_layers']
                    )
                    model.load_state_dict(checkpoint['model_state_dict'])

                    # 保存到数据库
                    from backend.core.database.repositories import ModelRepository
                    ModelRepository.save_model(session, ts_code, model, metadata)
                    migrated_count += 1

                    print(f"    ✅ 迁移成功")

            except Exception as e:
                print(f"    ❌ 错误: {e}")

        print(f"\n✅ 迁移完成，共 {migrated_count} 个模型")


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 股票预测系统 - 数据库初始化")
    print("=" * 60)

    # 1. 创建数据库
    create_database()

    # 2. 创建表结构
    print("\n📋 创建表结构...")
    init_database()

    # 3. 迁移CSV数据
    migrate_from_csv()

    # 4. 迁移模型文件
    migrate_models()

    print("\n" + "=" * 60)
    print("✅ 数据库初始化完成！")
    print("=" * 60)
    print(f"\n📊 数据库连接信息:")
    print(f"   主机: {DB_HOST}")
    print(f"   端口: {DB_PORT}")
    print(f"   数据库: {DB_NAME}")
    print(f"   用户: {DB_USER}")
    print(f"\n💡 后续步骤:")
    print(f"   1. 在backend/core/config.py中设置 USE_DATABASE = True")
    print(f"   2. 删除data/和models/目录（确认迁移成功后）")
    print(f"   3. 更新.env文件配置数据库连接")
    print(f"   4. 重启应用")
    print("=" * 60)


if __name__ == '__main__':
    main()
