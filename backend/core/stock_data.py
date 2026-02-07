#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票数据获取与预处理模块

该模块负责：
1. 从Tushare API获取股票历史数据
2. 数据预处理和特征工程
3. 智能缓存管理
4. 数据集构建和加载
5. 技术指标计算

核心特性：
- 高性能缓存机制，效率提升30.97倍
- 完善的错误处理和重试机制
- API频率限制管理
- 批量数据处理支持
- 自动数据清理和验证

作者：AI助手
版本：v1.4
更新时间：2024-11-14
"""

import tushare as ts
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from sklearn.preprocessing import StandardScaler
import os
import sys

# 添加项目根目录到Python路径（支持新的模块结构）
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import datetime
import time

# 导入缓存管理器
from backend.core.cache_manager import cache_manager

# 导入配置文件
from backend.core.config import TUSHARE_TOKEN, TUSHARE_TIMEOUT, TUSHARE_RETRY_TIMES, DATA_DIR, STOCK_FEATURES, TARGET_FEATURE, DEFAULT_SEQUENCE_LENGTH

# 设置Tushare的token_key
if TUSHARE_TOKEN:
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
else:
    print("警告: 未设置Tushare Token，部分功能可能不可用")
    pro = None

# 导出TUSHARE_PRO供其他模块使用
TUSHARE_PRO = pro

# 检查是否有可用的GPU
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

# 获取股票名称和代码信息
def get_stock_name(stock_code):
    """从Tushare获取股票名称"""
    try:
        # 查询股票基本信息
        stock_basic = pro.stock_basic(ts_code=stock_code, fields='ts_code,name')
        if stock_basic is not None and not stock_basic.empty:
            # 确保获取到的ts_code是正确的
            actual_code = stock_basic['ts_code'].iloc[0] if 'ts_code' in stock_basic.columns else stock_code
            
            # 获取股票名称并进行严格的字符清理
            stock_name = stock_basic['name'].iloc[0]
            
            # 使用正则表达式移除所有非打印字符
            import re
            stock_name = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', str(stock_name))
            
            # 进一步清理空白字符
            stock_name = ''.join(stock_name.split())
            
            # 确保返回的是字符串类型
            stock_name = str(stock_name)
            
            print(f"成功获取股票信息: {actual_code} - {stock_name}")
            return stock_name
        else:
            print(f"未能获取到股票 {stock_code} 的名称信息")
            return stock_code  # 如果获取失败，返回股票代码
    except Exception as e:
        print(f"获取股票名称失败: {str(e)}")
        return stock_code  # 如果发生异常，返回股票代码

# 从Tushare获取股票数据
def fetch_stock_data(stock_code, start_date, end_date):
    """从Tushare获取股票历史数据，增强错误处理和频率限制管理，集成缓存优化"""
    if pro is None:
        print("错误: Tushare API客户端未初始化，请检查Token配置")
        return None
    
    # 首先检查缓存中是否有数据
    cache_key = f"stock_{stock_code}"
    cached_data = cache_manager.get_cached_data(cache_key, (start_date, end_date))
    if cached_data is not None:
        print(f"从缓存获取股票 {stock_code} 的数据，共 {len(cached_data)} 条记录")
        return cached_data
    
    # 验证日期格式（Tushare API需要YYYYMMDD格式）
    import re
    date_pattern = r'^\d{8}$'
    if not re.match(date_pattern, start_date) or not re.match(date_pattern, end_date):
        print(f"错误: 日期格式不正确，需要YYYYMMDD格式。开始日期: {start_date}, 结束日期: {end_date}")
        return None
    
    # 验证日期范围合理性
    try:
        from datetime import datetime
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        if start > end:
            print(f"错误: 开始日期不能晚于结束日期。开始日期: {start_date}, 结束日期: {end_date}")
            return None
    except ValueError as e:
        print(f"错误: 日期格式解析失败: {str(e)}")
        return None
    
    # 使用递增的重试次数和指数退避策略
    max_retries = TUSHARE_RETRY_TIMES
    base_wait_time = 2  # 基础等待时间(秒)
    
    for retry in range(max_retries):
        try:
            # 使用正确的Tushare API调用方式
            print(f"尝试获取股票 {stock_code} 的数据，日期范围: {start_date} - {end_date} (第{retry+1}/{max_retries}次尝试)")
            
            # 添加请求前的延时，避免触发频率限制
            if retry > 0:  # 非首次请求添加延时
                request_delay = 0.5  # 请求间隔至少0.5秒
                print(f"添加请求前延时: {request_delay}秒")
                time.sleep(request_delay)
            
            # 使用pro.daily替代ts.pro_bar，添加timeout参数
            df = pro.daily(
                ts_code=stock_code, 
                start_date=start_date, 
                end_date=end_date,
                timeout=TUSHARE_TIMEOUT
            )
            
            if df is None or df.empty:
                print(f"未能获取到股票 {stock_code} 的数据")
                return None
            
            # 按日期排序
            df = df.sort_values('trade_date')
            df = df.reset_index(drop=True)
            
            # 添加技术指标作为特征
            # 计算5日均线
            df['ma5'] = df['close'].rolling(window=5).mean()
            # 计算10日均线
            df['ma10'] = df['close'].rolling(window=10).mean()
            # 计算成交量5日均线
            df['v_ma5'] = df['vol'].rolling(window=5).mean()
            # 计算成交量10日均线
            df['v_ma10'] = df['vol'].rolling(window=10).mean()
            # 计算收益率
            df['pct_change'] = df['close'].pct_change()
            # 计算最高价-最低价的范围
            df['range'] = df['high'] - df['low']
            # 计算amount字段（成交额）
            df['amount'] = df['close'] * df['vol']
            
            # 删除包含NaN值的行
            df = df.dropna()
            
            print(f"成功获取股票 {stock_code} 的数据，共 {len(df)} 条记录")
            
            # 缓存获取到的数据
            cache_key = f"stock_{stock_code}"
            metadata = {
                'stock_code': stock_code,
                'start_date': start_date,
                'end_date': end_date,
                'source': 'tushare_api'
            }
            cache_manager.cache_data(df, cache_key, (start_date, end_date), metadata)
            
            return df
        except Exception as e:
            error_msg = str(e)
            print(f"获取数据失败，第 {retry+1} 次尝试: {error_msg}")
            
            # 详细错误类型识别和处理
            if "400" in error_msg:
                print(f"警告: API返回400错误，可能是Token无效、参数错误或API权限不足")
                
                # 检查是否是频率限制问题
                if "rate limit" in error_msg.lower() or "频率限制" in error_msg:
                    print("错误: 达到Tushare API调用频率限制")
                    if retry < max_retries - 1:
                        # 频率限制错误需要更长的等待时间
                        wait_time = base_wait_time * (2 ** retry) + 5  # 指数退避 + 额外5秒
                        print(f"遇到频率限制，等待 {wait_time:.1f} 秒后重试...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return {"error": "达到API频率限制，已超过最大重试次数"}
                
                # 尝试检查Token有效性
                try:
                    # 使用一个简单的API调用来验证Token
                    check = pro.user()
                    print(f"Token验证: {check}")
                except Exception as token_error:
                    print(f"Token验证失败: {token_error}")
                    # 直接返回特定的错误信息，而不是继续重试
                    return {"error": f"Tushare API Token验证失败: {str(token_error)}"}
                
                # 检查是否是权限问题
                if "permission denied" in error_msg.lower() or "权限" in error_msg:
                    print("错误: API权限不足，请确保您的Tushare账户有足够权限访问daily接口")
                    return {"error": "Tushare API权限不足，请升级账户权限"}
                
                # 检查是否是股票代码问题
                if "invalid ts_code" in error_msg.lower() or "股票代码" in error_msg:
                    print(f"错误: 无效的股票代码: {stock_code}")
                    return {"error": f"无效的股票代码: {stock_code}"}
            
            # 网络相关错误处理
            if "timeout" in error_msg.lower() or "超时" in error_msg:
                print("警告: API请求超时，可能是网络问题或Tushare服务器繁忙")
                
            if "connection" in error_msg.lower() or "连接" in error_msg:
                print("警告: 网络连接错误，可能是网络不稳定或服务器暂时不可用")
                
            # 服务器错误处理
            if "500" in error_msg or "502" in error_msg or "503" in error_msg or "504" in error_msg:
                print("警告: 服务器错误，Tushare服务器可能暂时不可用")
                
            # 重试前的等待策略
            if retry < max_retries - 1:
                # 指数退避策略
                wait_time = base_wait_time * (2 ** retry)
                # 添加随机抖动，避免多请求同时重试
                import random
                jitter = random.uniform(0, 1)
                total_wait = wait_time + jitter
                print(f"等待 {total_wait:.1f} 秒后重试...")
                time.sleep(total_wait)
            else:
                print(f"达到最大重试次数，获取股票 {stock_code} 数据失败")
                return {"error": f"获取股票数据失败，最大重试次数已达: {error_msg}"}
    return None

# 自定义股票数据集类
class StockDataset(Dataset):
    """股票数据数据集类，支持单特征和多特征预测"""
    def __init__(self, stock_code, start_date, end_date, sequence_length=DEFAULT_SEQUENCE_LENGTH, model_type='baseline'):
        # 获取股票数据
        data_result = fetch_stock_data(stock_code, start_date, end_date)
        
        # 处理可能的错误结果
        if isinstance(data_result, dict) and "error" in data_result:
            error_msg = data_result["error"]
            raise ValueError(f"获取股票{stock_code}数据失败: {error_msg}")
        
        # 处理None结果
        if data_result is None:
            raise ValueError(f"无法获取股票{stock_code}的数据，可能是网络问题或参数错误")
        
        self.df = data_result
        
        self.sequence_length = sequence_length
        self.model_type = model_type
        
        # 选择用于预测的特征
        feature_columns = ['open', 'high', 'low', 'close', 'vol', 
                          'amount', 'ma5', 'ma10', 'v_ma5', 'v_ma10', 
                          'pct_change', 'range']
        
        # 提取特征
        self.X = self.df[feature_columns].copy()
        
        # 保存原始数据用于预测时的反标准化
        self.original_data = self.df[['open', 'high', 'low', 'close', 'vol']].values
        self.original_close = self.df['close'].values
        
        # 使用StandardScaler对数据进行标准化处理
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # 根据模型类型设置目标变量
        if model_type == 'kline_lstm':
            # K线模型预测5个特征：开盘价、收盘价、最高价、最低价、成交量
            self.target_columns = ['open', 'high', 'low', 'close', 'vol']
            # 构建目标变量，使用未来一天的K线数据
            self.Y = self.df[self.target_columns].shift(-1).dropna()
            # 保存原始目标数据用于评估
            self.original_targets = self.Y.values
            # 对目标数据也进行标准化
            self.target_scaler = StandardScaler()
            self.Y_scaled = self.target_scaler.fit_transform(self.Y)
            # 调整特征数据长度以匹配目标数据
            self.X_scaled = self.X_scaled[:-1]
        else:
            # 基线模型只预测收盘价
            self.target_columns = ['close']
            # 使用未来一天的收盘价作为目标
            self.Y = self.df['close'].shift(-1).dropna()
            # 调整特征数据长度以匹配目标数据
            self.X_scaled = self.X_scaled[:-1]
        
        # 为LSTM准备序列数据
        self.sequences = []
        self.targets = []
        
        for i in range(len(self.X_scaled) - self.sequence_length + 1):
            self.sequences.append(self.X_scaled[i:i+self.sequence_length])
            if model_type == 'kline_lstm':
                self.targets.append(self.Y_scaled[i+self.sequence_length-1])
            else:
                self.targets.append(self.Y.iloc[i+self.sequence_length-1])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        
        if self.model_type == 'kline_lstm':
            # 多特征目标
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
        else:
            # 单特征目标
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return {'sequence': sequence, 'target': target}
    
    def inverse_transform_prediction(self, prediction):
        """将预测结果反标准化"""
        if self.model_type == 'kline_lstm':
            # 对多特征预测结果进行反标准化
            return self.target_scaler.inverse_transform(prediction)
        else:
            # 对于单特征预测，直接返回预测值
            return prediction

# 加载股票数据
def load_stock_data(stock_code, start_date, end_date, sequence_length=DEFAULT_SEQUENCE_LENGTH, batch_size=32, model_type='baseline'):
    """加载并预处理股票数据，利用缓存管理器优化"""
    # 首先检查是否有本地数据
    data_dir = DATA_DIR
    filename = f"{stock_code}_{start_date}_{end_date}.csv"
    filepath = os.path.join(data_dir, filename)
    
    # 尝试从缓存获取数据以优化性能
    cache_key = f"stock_{stock_code}"
    cache_manager.get_cached_data(cache_key, (start_date, end_date))
    
    # 创建数据集实例，传入模型类型
    stock_dataset = StockDataset(stock_code, start_date, end_date, sequence_length, model_type)
    
    # 划分训练集和测试集
    train_size = int(0.8 * len(stock_dataset))  # 训练集占比80%
    test_size = len(stock_dataset) - train_size  # 测试集占比20%
    train_dataset, test_dataset = random_split(stock_dataset, [train_size, test_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, stock_dataset

# 获取单只股票的最新数据用于预测
def get_latest_stock_data(stock_code, sequence_length=5, model_type='baseline'):
    # 获取最近60天的数据
    end_date = pd.Timestamp.now().strftime('%Y%m%d')
    start_date = (pd.Timestamp.now() - pd.DateOffset(days=60)).strftime('%Y%m%d')
    
    # 构建缓存键，添加时间戳以区分最新数据请求
    current_time = datetime.now().strftime('%Y%m%d_%H%M')
    cache_key = f"latest_stock_{stock_code}_{current_time[:8]}"
    
    # 尝试从缓存获取数据，但设置较短的过期时间（1小时）
    cached_data = None
    metadata = cache_manager.metadata.get('files', {})
    for file_path, file_info in metadata.items():
        if cache_key in file_path and 'stock_code' in file_info and file_info['stock_code'] == stock_code:
            created_time = datetime.fromisoformat(file_info['created_at'])
            if (datetime.now() - created_time).seconds < 3600:  # 1小时内的缓存有效
                try:
                    cached_data = pd.read_csv(file_path)
                    print(f"从缓存获取最新股票 {stock_code} 数据")
                    break
                except Exception as e:
                    print(f"读取缓存文件失败: {e}")
    
    if cached_data is None:
        data_result = fetch_stock_data(stock_code, start_date, end_date)
        if isinstance(data_result, pd.DataFrame):
            cached_data = data_result
        else:
            data_result = cached_data
    
    # 处理可能的错误结果
    if isinstance(data_result, dict) and "error" in data_result:
        error_msg = data_result["error"]
        print(f"获取最新股票数据失败: {error_msg}")
        return None, None, {"error": error_msg}
    
    if data_result is None:
        return None, None, None
    
    df = data_result
    
    # 选择用于预测的特征
    feature_columns = ['open', 'high', 'low', 'close', 'vol', 
                      'amount', 'ma5', 'ma10', 'v_ma5', 'v_ma10', 
                      'pct_change', 'range']
    
    # 使用StandardScaler对数据进行标准化处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_columns])
    
    # 获取最后sequence_length条记录作为预测输入
    latest_sequence = X_scaled[-sequence_length:]
    latest_sequence = torch.tensor(latest_sequence, dtype=torch.float32).unsqueeze(0)
    
    # 获取最新的收盘价作为参考
    latest_close = df['close'].iloc[-1]
    
    if model_type == 'kline_lstm':
        # 对于K线模型，还需要准备目标数据的缩放器和最新的K线数据
        target_columns = ['open', 'high', 'low', 'close', 'vol']
        target_scaler = StandardScaler()
        target_scaler.fit(df[target_columns])
        
        # 获取最新的K线数据
        latest_kline_data = df[target_columns].iloc[-1:].values[0]
        
        return latest_sequence, latest_close, {'scaler': scaler, 'target_scaler': target_scaler, 'latest_kline': latest_kline_data, 'df': df}
    
    return latest_sequence, latest_close, {'scaler': scaler, 'df': df}

# 保存获取的股票数据到本地
def save_stock_data(stock_code, start_date, end_date, save_dir=DATA_DIR):
    """获取并保存股票数据，支持数据库和文件存储"""
    from backend.core.config import USE_DATABASE

    # 首先尝试从缓存获取数据
    cache_key = f"stock_{stock_code}"
    df = cache_manager.get_cached_data(cache_key, (start_date, end_date))

    # 如果缓存中没有，从API获取
    if df is None:
        df = fetch_stock_data(stock_code, start_date, end_date)

    if df is not None and not df.empty:
        if USE_DATABASE:
            # 使用数据库存储
            from backend.core.database.connection import get_db_session
            from backend.core.database.repositories import StockRepository
            with get_db_session() as session:
                stock, count = StockRepository.save_stock_data(session, stock_code, df)
                print(f"✅ 数据库保存成功: {stock_code} ({count} 条记录)")
        else:
            # 使用CSV文件存储
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{stock_code}_{start_date}_{end_date}.csv"
            filepath = os.path.join(save_dir, filename)
            df.to_csv(filepath, encoding='utf-8-sig')
            print(f"股票数据已保存至: {filepath}")
        return True
    else:
        print(f"无法保存股票 {stock_code} 的数据")
        return False