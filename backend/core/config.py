# 股票预测系统配置文件

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# Tushare API配置
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN', '')  # 从环境变量读取API Token
TUSHARE_TIMEOUT = 10  # API请求超时时间(秒)
TUSHARE_RETRY_TIMES = 3  # API请求失败重试次数

# 数据路径配置
DATA_DIR = 'data'  # 股票数据存储目录
MODELS_DIR = 'models'  # 模型存储目录
RESULTS_DIR = 'results'  # 预测结果存储目录
PLOTS_DIR = 'plots'  # 图表存储目录

# 默认数据参数
DEFAULT_START_DATE = '20240915'  # 默认开始日期
DEFAULT_END_DATE = '20250916'  # 默认结束日期
TEST_SIZE = 0.2  # 测试集比例
RANDOM_STATE = 42  # 随机种子

# 默认模型参数
DEFAULT_MODEL_TYPE = 'baseline'  # 默认模型类型 (现在只支持基线LSTM)
DEFAULT_HIDDEN_SIZE = 64  # 默认隐藏层大小
DEFAULT_NUM_LAYERS = 2  # 默认LSTM层数
DEFAULT_SEQUENCE_LENGTH = 5  # 默认序列长度
DEFAULT_BATCH_SIZE = 32  # 默认批次大小
DEFAULT_LEARNING_RATE = 0.001 # 默认学习率
DEFAULT_EPOCHS = 100  # 默认最大训练轮数
DEFAULT_PATIENCE = 10  # 默认早停耐心值

# 可视化配置
PLOT_FIGURE_SIZE = (12, 6)  # 图表尺寸
PLOT_DPI = 100  # 图表DPI
PLOT_STYLE = 'ggplot'  # 图表样式

# 日志配置
LOG_LEVEL = 'INFO'  # 日志级别
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # 日志格式

# 性能配置
USE_GPU = True  # 是否使用GPU
CPU_THREADS = 4  # CPU线程数

# 模型评估指标配置
METRICS = ['mse', 'mae', 'rmse', 'r2']  # 评估指标列表

# 数据特征配置
STOCK_FEATURES = ['open', 'high', 'low', 'close', 'volume', 'amount']  # 使用的股票特征
TARGET_FEATURE = 'close'  # 目标预测特征

# 交易日期相关配置
TRADING_DAYS_PER_YEAR = 250  # 每年交易日数量
TRADING_DAYS_PER_MONTH = 20  # 每月交易日数量

# 批量预测配置
BATCH_PREDICT_LIMIT = 20  # 单次批量预测的股票数量上限
MAX_BATCH_PREDICTION = BATCH_PREDICT_LIMIT  # 与BATCH_PREDICT_LIMIT保持一致，用于API接口

# 数据缓存配置
CACHE_ENABLED = True  # 是否启用数据缓存
CACHE_EXPIRE_DAYS = 7  # 缓存过期天数

# 模型选择配置
MODEL_COMPARISON_METRIC = 'rmse'  # 模型比较指标

# 预测周期配置
PREDICTION_HORIZON = 5  # 预测未来天数

# 数据标准化配置
SCALER_TYPE = 'standard'  # 标准化方法 (standard/minmax/robust)

# 早停机制配置
EARLY_STOPPING_ENABLED = True  # 是否启用早停机制
EARLY_STOPPING_MIN_DELTA = 0.001  # 早停最小改进值

# 学习率调度配置
LR_SCHEDULER_ENABLED = True  # 是否启用学习率调度
LR_SCHEDULER_STEP_SIZE = 30  # 学习率衰减步长
LR_SCHEDULER_GAMMA = 0.1  # 学习率衰减系数

# 结果导出配置
EXPORT_FORMAT = 'csv'  # 导出格式 (csv/excel)
EXPORT_ENCODING = 'utf-8-sig'  # 导出文件编码

# 自动保存配置
AUTO_SAVE_MODEL = True  # 是否自动保存模型
AUTO_SAVE_INTERVAL = 10  # 自动保存间隔轮数

# 异常处理配置
MAX_RETRY_COUNT = 5  # 最大重试次数
RETRY_INTERVAL_SECONDS = 2  # 重试间隔秒数

# 系统版本
SYSTEM_VERSION = '1.0.0'  # 系统版本号
SYSTEM_NAME = 'Stock Predictor System'  # 系统名称