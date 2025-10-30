# 股票数据获取与预处理模块
import tushare as ts
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import time

# 导入配置文件
from config import TUSHARE_TOKEN, TUSHARE_TIMEOUT, TUSHARE_RETRY_TIMES, DATA_DIR, STOCK_FEATURES, TARGET_FEATURE, DEFAULT_SEQUENCE_LENGTH

# 设置Tushare的token_key
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

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
    """从Tushare获取股票历史数据"""
    for retry in range(TUSHARE_RETRY_TIMES):
        try:
            # 获取股票交易数据
            df = ts.pro_bar(ts_code=stock_code, adj='qfq', start_date=start_date, end_date=end_date)
            
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
            
            # 删除包含NaN值的行
            df = df.dropna()
            
            return df
        except Exception as e:
            print(f"获取数据失败，第 {retry+1} 次重试: {e}")
            if retry < TUSHARE_RETRY_TIMES - 1:
                time.sleep(2)  # 重试间隔
            else:
                print(f"达到最大重试次数，获取股票 {stock_code} 数据失败")
                return None
    return None

# 自定义股票数据集类
class StockDataset(Dataset):
    """股票数据数据集类"""
    def __init__(self, stock_code, start_date, end_date, sequence_length=DEFAULT_SEQUENCE_LENGTH):
        # 获取股票数据
        self.df = fetch_stock_data(stock_code, start_date, end_date)
        if self.df is None:
            raise ValueError(f"无法获取股票{stock_code}的数据")
        
        self.sequence_length = sequence_length
        
        # 选择用于预测的特征
        feature_columns = ['open', 'high', 'low', 'close', 'vol', 
                          'amount', 'ma5', 'ma10', 'v_ma5', 'v_ma10', 
                          'pct_change', 'range']
        
        # 提取特征和目标变量
        self.X = self.df[feature_columns]
        # 使用未来一天的收盘价作为目标
        self.Y = self.df['close'].shift(-1).dropna()
        # 调整特征数据长度以匹配目标数据
        self.X = self.X.iloc[:-1]
        
        # 保存原始数据用于预测时的反标准化
        self.original_close = self.df['close'].values
        
        # 使用StandardScaler对数据进行标准化处理
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # 为LSTM准备序列数据
        self.sequences = []
        self.targets = []
        
        for i in range(len(self.X_scaled) - self.sequence_length + 1):
            self.sequences.append(self.X_scaled[i:i+self.sequence_length])
            self.targets.append(self.Y.iloc[i+self.sequence_length-1])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return {'sequence': sequence, 'target': target}

# 加载股票数据
def load_stock_data(stock_code, start_date, end_date, sequence_length=DEFAULT_SEQUENCE_LENGTH, batch_size=32):
    """加载并预处理股票数据"""
    # 首先检查是否有本地数据
    data_dir = DATA_DIR
    filename = f"{stock_code}_{start_date}_{end_date}.csv"
    filepath = os.path.join(data_dir, filename)
    
    # 创建数据集实例
    stock_dataset = StockDataset(stock_code, start_date, end_date, sequence_length)
    
    # 划分训练集和测试集
    train_size = int(0.8 * len(stock_dataset))  # 训练集占比80%
    test_size = len(stock_dataset) - train_size  # 测试集占比20%
    train_dataset, test_dataset = random_split(stock_dataset, [train_size, test_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, stock_dataset

# 获取单只股票的最新数据用于预测
def get_latest_stock_data(stock_code, sequence_length=5):
    # 获取最近60天的数据
    end_date = pd.Timestamp.now().strftime('%Y%m%d')
    start_date = (pd.Timestamp.now() - pd.DateOffset(days=60)).strftime('%Y%m%d')
    
    df = fetch_stock_data(stock_code, start_date, end_date)
    if df is None:
        return None, None
    
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
    
    return latest_sequence, latest_close

# 保存获取的股票数据到本地
def save_stock_data(stock_code, start_date, end_date, save_dir=DATA_DIR):
    """获取并保存股票数据"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取股票数据
    df = fetch_stock_data(stock_code, start_date, end_date)
    if df is not None and not df.empty:
        # 生成文件名
        filename = f"{stock_code}_{start_date}_{end_date}.csv"
        filepath = os.path.join(save_dir, filename)
        
        # 保存数据
        df.to_csv(filepath, encoding='utf-8-sig')
        print(f"股票数据已保存至: {filepath}")
        return True
    else:
        print(f"无法保存股票 {stock_code} 的数据")
        return False