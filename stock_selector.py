# 智能选股模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import time
import random
from sklearn.preprocessing import MinMaxScaler
import tushare as ts
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

# 导入配置
from config import (
    TUSHARE_TOKEN, DATA_DIR, PLOTS_DIR, TRADING_DAYS_PER_YEAR, 
    TRADING_DAYS_PER_MONTH, EXPORT_FORMAT, EXPORT_ENCODING
)

# 设置Tushare的token_key
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

class StockSelector:
    """智能选股器类，提供多维度的股票分析和选择功能"""
    
    def __init__(self, cache_dir=DATA_DIR):
        """初始化选股器
        
        Args:
            cache_dir: 缓存数据的目录
        """
        self.cache_dir = cache_dir
        self.stock_pool = None  # 股票池
        self.analysis_results = None  # 分析结果
        self.selected_stocks = None  # 选中的股票
        self.request_interval = 0.6  # API请求间隔(秒)，避免触发频率限制
        self.max_batch_size = 20  # 批量处理时的最大股票数量
        
        # 初始化Tushare API连接
        ts.set_token(TUSHARE_TOKEN)
        self.pro = ts.pro_api()
        
        # 确保目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)
        
        print("选股器初始化完成，已设置请求间隔为{:.1f}秒，最大批量大小为{}".format(
            self.request_interval, self.max_batch_size))
    
    def fetch_stock_pool(self, market='A', industry=None, pool_type='all'):
        """获取股票池
        
        Args:
            market: 市场类型，默认'A'股
            industry: 行业筛选，默认为None（不筛选）
            pool_type: 股票池类型，目前支持'all'（所有A股）
            
        Returns:
            pandas DataFrame，包含股票基础信息
        """
        try:
            # 获取股票基本信息
            fields = 'ts_code,symbol,name,industry,area,list_date'
            # 使用与直接调用相同的方式，不使用market参数
            stock_basic = self.pro.stock_basic(exchange='', list_status='L', fields=fields)
            
            # 筛选市场（根据ts_code后缀）
            if market and not market.strip() == '' and market.upper() != 'A':
                market_suffix = market.upper()
                # 根据ts_code后缀筛选市场（SZ表示深圳，SH表示上海）
                # 当market='A'时，表示所有A股，不进行筛选
                stock_basic = stock_basic[stock_basic['ts_code'].str.endswith(f'.{market_suffix}')]
            
            # 筛选行业
            if industry and not industry.strip() == '':
                # 使用包含匹配而不是精确匹配，增加灵活性
                # 对于计算机相关行业，匹配包含'软件'、'通信'、'电子'、'计算机'等关键词的行业
                if industry in ['计算机', '科技']:
                    # 计算机相关行业关键词
                    computer_keywords = ['软件', '通信', '电子', '计算机', 'IT', '互联网']
                    mask = stock_basic['industry'].str.contains('|'.join(computer_keywords), na=False)
                    stock_basic = stock_basic[mask]
                else:
                    # 其他行业使用包含匹配
                    stock_basic = stock_basic[stock_basic['industry'].str.contains(industry, na=False)]
            
            # 过滤掉创业板和科创板（如果需要）
            # stock_basic = stock_basic[~stock_basic['ts_code'].str.endswith('SZ') | ~stock_basic['symbol'].str.startswith(('300', '688'))]
            
            # 保存股票池和基础信息
            self.stock_pool = stock_basic
            self.stock_basic_info = stock_basic
            print(f"成功获取{len(stock_basic)}只股票")
            return stock_basic
        except Exception as e:
            print(f"获取股票池失败: {str(e)}")
            # 返回空的DataFrame而不是None
            return pd.DataFrame()
    
    def calculate_returns(self, stock_code, start_date, end_date, periods=None):
        """计算不同周期的收益率
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            periods: 计算周期列表，如[5, 10, 20, 60, 120, 250]（交易日）
            
        Returns:
            returns: 不同周期的收益率字典
        """
        try:
            # 默认周期（日、周、月、季、半年、年）
            if periods is None:
                periods = [1, 5, 20, 60, 120, 250]
            
            # 加载股票数据
            df = self._load_stock_data(stock_code, start_date, end_date)
            if df is None or df.empty:
                return None
            
            returns = {}
            
            # 计算日收益率
            df['daily_return'] = df['close'].pct_change()
            
            # 计算各周期收益率
            for period in periods:
                if len(df) >= period:
                    # 计算区间收益率
                    returns[f'{period}d_return'] = (df['close'].iloc[-1] / df['close'].iloc[-period] - 1)
                else:
                    returns[f'{period}d_return'] = np.nan
            
            # 计算平均日收益率
            returns['avg_daily_return'] = df['daily_return'].mean()
            
            # 计算总收益率
            returns['total_return'] = (df['close'].iloc[-1] / df['close'].iloc[0] - 1)
            
            return returns
        except Exception as e:
            print(f"计算收益率失败: {e}")
            return None
    
    def calculate_volatility(self, stock_code, start_date, end_date, window=20):
        """计算波动率指标
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            window: 计算窗口大小
            
        Returns:
            volatility: 波动率指标字典
        """
        try:
            # 加载股票数据
            df = self._load_stock_data(stock_code, start_date, end_date)
            if df is None or df.empty:
                return None
            
            # 计算日收益率
            df['daily_return'] = df['close'].pct_change()
            
            # 计算标准差（波动率）
            volatility = {
                'std_returns': df['daily_return'].std(),  # 日收益率标准差
                'annualized_vol': df['daily_return'].std() * np.sqrt(TRADING_DAYS_PER_YEAR),  # 年化波动率
                'windowed_vol': df['daily_return'].rolling(window=window).std().iloc[-1]  # 窗口波动率
            }
            
            # 计算最高最低价的波动范围
            volatility['price_range_pct'] = (df['high'].max() - df['low'].min()) / df['low'].min()
            
            # 计算涨跌幅统计
            volatility['max_daily_gain'] = df['daily_return'].max()
            volatility['max_daily_loss'] = df['daily_return'].min()
            volatility['positive_days_pct'] = (df['daily_return'] > 0).sum() / len(df)
            
            return volatility
        except Exception as e:
            print(f"计算波动率失败: {e}")
            return None
    
    def calculate_beta(self, stock_code, start_date, end_date, benchmark='000001.SH'):
        """计算β系数
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            benchmark: 基准指数代码，默认为上证指数
            
        Returns:
            beta: β系数
        """
        try:
            # 加载股票数据
            stock_df = self._load_stock_data(stock_code, start_date, end_date)
            if stock_df is None or stock_df.empty:
                return None
            
            # 加载基准指数数据
            benchmark_df = self._load_stock_data(benchmark, start_date, end_date)
            if benchmark_df is None or benchmark_df.empty:
                return None
            
            # 合并数据
            merged_df = pd.merge(stock_df[['trade_date', 'close']], 
                                benchmark_df[['trade_date', 'close']], 
                                on='trade_date', suffixes=('_stock', '_benchmark'))
            
            # 计算收益率
            merged_df['return_stock'] = merged_df['close_stock'].pct_change()
            merged_df['return_benchmark'] = merged_df['close_benchmark'].pct_change()
            
            # 计算协方差和方差
            cov_matrix = np.cov(merged_df['return_stock'].dropna(), merged_df['return_benchmark'].dropna())
            beta = cov_matrix[0, 1] / cov_matrix[1, 1]
            
            return beta
        except Exception as e:
            print(f"计算β系数失败: {e}")
            return None
    
    def calculate_technical_indicators(self, stock_code, start_date, end_date):
        """计算技术指标
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            indicators: 技术指标字典
        """
        try:
            # 加载股票数据
            df = self._load_stock_data(stock_code, start_date, end_date)
            if df is None or df.empty:
                return None
            
            indicators = {}
            
            # 计算均线
            for ma in [5, 10, 20, 60]:
                df[f'ma{ma}'] = df['close'].rolling(window=ma).mean()
                # 价格是否在均线上方
                indicators[f'price_above_ma{ma}'] = 1 if df['close'].iloc[-1] > df[f'ma{ma}'].iloc[-1] else 0
                # 均线斜率
                indicators[f'ma{ma}_slope'] = (df[f'ma{ma}'].iloc[-1] / df[f'ma{ma}'].iloc[-5] - 1)
            
            # 计算MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            indicators['macd'] = macd.iloc[-1]
            indicators['macd_signal'] = signal.iloc[-1]
            indicators['macd_hist'] = (macd - signal).iloc[-1]
            
            # 计算RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi'] = rsi.iloc[-1]
            
            # 计算布林带
            df['sma20'] = df['close'].rolling(window=20).mean()
            df['std20'] = df['close'].rolling(window=20).std()
            df['upper_band'] = df['sma20'] + (df['std20'] * 2)
            df['lower_band'] = df['sma20'] - (df['std20'] * 2)
            indicators['bollinger_width'] = (df['upper_band'].iloc[-1] - df['lower_band'].iloc[-1]) / df['sma20'].iloc[-1]
            
            return indicators
        except Exception as e:
            print(f"计算技术指标失败: {e}")
            return None
    
    def calculate_valuation_metrics(self, stock_code):
        """计算估值指标
        
        Args:
            stock_code: 股票代码
            
        Returns:
            valuation: 估值指标字典
        """
        try:
            # 获取估值数据
            # 注意：实际使用时可能需要获取更多估值数据，这里简化处理
            valuation = {}
            
            # 尝试获取市盈率
            try:
                pe_data = pro.daily_basic(ts_code=stock_code, fields='pe_ttm,pe')
                if not pe_data.empty:
                    valuation['pe_ttm'] = pe_data['pe_ttm'].iloc[0] if pd.notna(pe_data['pe_ttm'].iloc[0]) else None
                    valuation['pe'] = pe_data['pe'].iloc[0] if pd.notna(pe_data['pe'].iloc[0]) else None
            except:
                pass
            
            return valuation
        except Exception as e:
            print(f"计算估值指标失败: {e}")
            return None
    
    def comprehensive_analysis(self, stock_code, start_date, end_date):
        """综合分析单只股票
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            analysis: 综合分析结果
        """
        try:
            # 基本信息
            stock_info = self.stock_pool[self.stock_pool['ts_code'] == stock_code].iloc[0].to_dict() if self.stock_pool is not None else {}
            
            # 计算各类指标
            returns = self.calculate_returns(stock_code, start_date, end_date)
            volatility = self.calculate_volatility(stock_code, start_date, end_date)
            beta = self.calculate_beta(stock_code, start_date, end_date)
            technical = self.calculate_technical_indicators(stock_code, start_date, end_date)
            valuation = self.calculate_valuation_metrics(stock_code)
            
            # 合并分析结果
            analysis = {
                'ts_code': stock_code,
                'name': stock_info.get('name', stock_code),
                'industry': stock_info.get('industry', ''),
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                **(returns or {}),
                **(volatility or {}),
                **({'beta': beta} if beta is not None else {}),
                **(technical or {}),
                **(valuation or {})
            }
            
            return analysis
        except Exception as e:
            print(f"综合分析失败: {e}")
            return None
    
    def analyze_stock_pool(self, stock_pool=None, start_date=None, end_date=None, limit=50):
        """分析股票池中的股票，实现分批处理以避免API频率限制
        
        Args:
            stock_pool: 股票池DataFrame，默认为None（使用已加载的股票池）
            start_date: 开始日期，默认为None（使用90天前）
            end_date: 结束日期，默认为None（使用今天）
            limit: 分析的股票数量限制
            
        Returns:
            results: 分析结果DataFrame
        """
        try:
            # 使用默认日期
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
            
            # 使用指定的股票池
            if stock_pool is not None:
                self.stock_pool = stock_pool
            
            # 如果没有股票池，先获取
            if self.stock_pool is None:
                self.fetch_stock_pool()
            
            results = []
            
            # 限制分析的股票数量
            analyze_pool = self.stock_pool.head(limit)
            print(f"开始分析{len(analyze_pool)}只股票...")
            
            # 将股票池分成多个批次
            batches = [analyze_pool.iloc[i:i+self.max_batch_size] 
                       for i in range(0, len(analyze_pool), self.max_batch_size)]
            print(f"将股票分为{len(batches)}个批次进行分析")
            
            # 分析每批股票
            total_stocks = len(analyze_pool)
            processed_count = 0
            
            for batch_idx, batch in enumerate(batches):
                print(f"\n处理批次 {batch_idx + 1}/{len(batches)}: 包含 {len(batch)} 只股票")
                
                # 分析批次中的每只股票
                for _, stock in batch.iterrows():
                    processed_count += 1
                    stock_code = stock['ts_code']
                    stock_name = stock.get('name', stock_code)
                    print(f"[{processed_count}/{total_stocks}] 分析股票: {stock_code} ({stock_name})")
                    
                    analysis = self.comprehensive_analysis(stock_code, start_date, end_date)
                    if analysis:
                        results.append(analysis)
                
                # 批次间休息，避免连续调用过多
                if batch_idx < len(batches) - 1:
                    wait_time = self.request_interval * 2
                    print(f"批次处理完成，休息 {wait_time:.1f} 秒")
                    time.sleep(wait_time)
            
            # 转换为DataFrame
            self.analysis_results = pd.DataFrame(results)
            print(f"\n完成分析，共获得{len(self.analysis_results)}只股票的有效结果")
            
            return self.analysis_results
        except Exception as e:
            print(f"分析股票池失败: {e}")
            return None
    
    def score_stocks(self, weights=None):
        """为股票评分
        
        Args:
            weights: 评分权重字典
            
        Returns:
            scored_stocks: 带评分的股票DataFrame
        """
        try:
            if self.analysis_results is None or self.analysis_results.empty:
                print("没有可评分的分析结果")
                return None
            
            # 复制结果，避免修改原数据
            scored_df = self.analysis_results.copy()
            
            # 设置默认权重
            if weights is None:
                weights = {
                    'returns_score': 0.3,  # 收益率评分权重
                    'volatility_score': 0.3,  # 波动率评分权重
                    'technical_score': 0.3,  # 技术指标评分权重
                    'valuation_score': 0.1  # 估值评分权重
                }
            
            # 计算收益率评分（越高越好）
            # 综合考虑短期、中期、长期收益率
            returns_factors = []
            if '1d_return' in scored_df.columns:
                returns_factors.append('1d_return')
            if '5d_return' in scored_df.columns:
                returns_factors.append('5d_return')
            if '20d_return' in scored_df.columns:
                returns_factors.append('20d_return')
            if '60d_return' in scored_df.columns:
                returns_factors.append('60d_return')
            
            if returns_factors:
                # 创建收益率综合指标
                scored_df['returns_composite'] = scored_df[returns_factors].mean(axis=1)
                # 标准化并转换为0-100的评分
                scaler = MinMaxScaler()
                scored_df['returns_score'] = scaler.fit_transform(scored_df[['returns_composite']]) * 100
            else:
                scored_df['returns_score'] = 50  # 默认中间分数
            
            # 计算波动率评分（越低越好，所以需要反转）
            if 'annualized_vol' in scored_df.columns:
                # 标准化并反转（低波动率得高分）
                vol_scaler = MinMaxScaler()
                scored_df['volatility_score'] = (1 - vol_scaler.fit_transform(scored_df[['annualized_vol']])) * 100
            else:
                scored_df['volatility_score'] = 50  # 默认中间分数
            
            # 计算技术指标评分
            tech_scores = []
            # RSI评分（30-70为正常区间）
            if 'rsi' in scored_df.columns:
                rsi_scores = 100 - (np.abs(scored_df['rsi'] - 50) / 50 * 100)
                tech_scores.append(rsi_scores)
            
            # 均线评分（价格在均线上方为好）
            for ma in [5, 10, 20]:
                if f'price_above_ma{ma}' in scored_df.columns:
                    tech_scores.append(scored_df[f'price_above_ma{ma}'] * 100)
            
            # 均线斜率评分
            for ma in [5, 10, 20]:
                if f'ma{ma}_slope' in scored_df.columns:
                    # 标准化斜率
                    ma_slope_scaler = MinMaxScaler()
                    slope_col = f'ma{ma}_slope_score'
                    scored_df[slope_col] = (ma_slope_scaler.fit_transform(scored_df[[f'ma{ma}_slope']]) + 1) / 2 * 100
                    tech_scores.append(scored_df[slope_col])
            
            if tech_scores:
                scored_df['technical_score'] = np.mean(tech_scores, axis=0)
            else:
                scored_df['technical_score'] = 50  # 默认中间分数
            
            # 计算估值评分（PE较低为好）
            if 'pe_ttm' in scored_df.columns:
                # 移除异常值
                pe_data = scored_df['pe_ttm'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(pe_data) > 0:
                    # 找出合理的PE范围
                    pe_q1 = pe_data.quantile(0.1)
                    pe_q3 = pe_data.quantile(0.9)
                    # 限制在合理范围内
                    pe_normalized = scored_df['pe_ttm'].clip(lower=pe_q1, upper=pe_q3)
                    # 标准化并反转（低PE得高分）
                    pe_scaler = MinMaxScaler()
                    scored_df['valuation_score'] = (1 - pe_scaler.fit_transform(pe_normalized.values.reshape(-1, 1))) * 100
                else:
                    scored_df['valuation_score'] = 50
            else:
                scored_df['valuation_score'] = 50  # 默认中间分数
            
            # 计算综合评分
            scored_df['total_score'] = (
                scored_df['returns_score'] * weights['returns_score'] +
                scored_df['volatility_score'] * weights['volatility_score'] +
                scored_df['technical_score'] * weights['technical_score'] +
                scored_df['valuation_score'] * weights['valuation_score']
            )
            
            # 按总分排序
            scored_df = scored_df.sort_values('total_score', ascending=False).reset_index(drop=True)
            
            # 计算排名
            scored_df['rank'] = scored_df.index + 1
            
            self.analysis_results = scored_df
            return scored_df
        except Exception as e:
            print(f"评分失败: {e}")
            return None
    
    def select_stocks(self, top_n=10, min_score=None, conditions=None):
        """选择最优股票，支持分数选择和条件筛选
        
        Args:
            top_n: 选择前N只股票
            min_score: 最低分数限制
            conditions: 额外的筛选条件字典，格式为 {'indicator': {'operator': value}}
            
        Returns:
            selected: 选中的股票DataFrame
        """
        try:
            if self.analysis_results is None or self.analysis_results.empty:
                print("没有可选择的分析结果，请先运行分析")
                return None
            
            # 确保已经评分
            if 'total_score' not in self.analysis_results.columns:
                self.score_stocks()
            
            # 应用分数限制
            if min_score is not None:
                selected = self.analysis_results[self.analysis_results['total_score'] >= min_score]
            else:
                selected = self.analysis_results.copy()
            
            # 应用额外条件筛选（如果有）
            if conditions:
                original_count = len(selected)
                print(f"开始应用条件筛选，初始有{original_count}只股票符合分数要求")
                
                for idx, row in selected.iterrows():
                    match = True
                    
                    # 检查所有条件
                    for indicator, condition in conditions.items():
                        # 跳过不存在的指标
                        if indicator not in row:
                            match = False
                            break
                        
                        # 检查操作符和值
                        for operator, value in condition.items():
                            try:
                                if operator == '>':
                                    if not (row[indicator] > value):
                                        match = False
                                        break
                                elif operator == '<':
                                    if not (row[indicator] < value):
                                        match = False
                                        break
                                elif operator == '>=':
                                    if not (row[indicator] >= value):
                                        match = False
                                        break
                                elif operator == '<=':
                                    if not (row[indicator] <= value):
                                        match = False
                                        break
                                elif operator == '=' or operator == '==':
                                    if not (row[indicator] == value):
                                        match = False
                                        break
                                elif operator == '!=' or operator == '≠':
                                    if not (row[indicator] != value):
                                        match = False
                                        break
                            except TypeError:
                                # 处理类型不匹配的情况
                                print(f"警告: 股票 {row['ts_code']} 的指标 {indicator} 类型不匹配")
                                match = False
                                break
                        
                        if not match:
                            break
                    
                    # 如果不满足条件，从选中列表移除
                    if not match:
                        selected = selected.drop(idx)
                
                filtered_count = len(selected)
                print(f"条件筛选完成: 从{original_count}只股票中筛选出{filtered_count}只符合所有条件")
            
            # 选择前N只
            self.selected_stocks = selected.head(top_n)
            
            print(f"最终选择了{len(self.selected_stocks)}只最优股票")
            
            # 打印选中的股票信息
            if not self.selected_stocks.empty:
                print("\n选中的股票:")
                for _, stock in self.selected_stocks.iterrows():
                    print(f"{stock['name']} ({stock['ts_code']}) - 总分: {stock['total_score']:.2f}")
            
            return self.selected_stocks
        except Exception as e:
            print(f"选择股票失败: {e}")
            return None
    
    def export_results(self, filename=None, format=None):
        """导出分析结果
        
        Args:
            filename: 文件名
            format: 导出格式 (csv/excel)
            
        Returns:
            filepath: 导出文件路径
        """
        try:
            if self.selected_stocks is None or self.selected_stocks.empty:
                print("没有可导出的结果，请先选择股票")
                return None
            
            # 使用默认值
            if filename is None:
                filename = f"stock_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if format is None:
                format = EXPORT_FORMAT
            
            # 确保目录存在
            os.makedirs(RESULTS_DIR, exist_ok=True)
            
            # 导出文件路径
            filepath = os.path.join(RESULTS_DIR, f"{filename}.{format}")
            
            # 根据格式导出
            if format == 'csv':
                self.selected_stocks.to_csv(filepath, index=False, encoding=EXPORT_ENCODING)
            elif format == 'excel':
                self.selected_stocks.to_excel(filepath, index=False)
            
            print(f"结果已导出至: {filepath}")
            return filepath
        except Exception as e:
            print(f"导出结果失败: {e}")
            return None
    
    def visualize_results(self, filename=None):
        """可视化选股结果
        
        Args:
            filename: 图表文件名
            
        Returns:
            filepath: 图表文件路径
        """
        try:
            if self.selected_stocks is None or self.selected_stocks.empty:
                print("没有可视化的结果，请先选择股票")
                return None
            
            # 使用默认文件名
            if filename is None:
                filename = f"stock_selection_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 确保目录存在
            os.makedirs(PLOTS_DIR, exist_ok=True)
            
            # 设置图表风格
            plt.style.use('ggplot')
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 股票评分柱状图
            top_stocks = self.selected_stocks.head(10)
            axes[0, 0].barh(range(len(top_stocks)), top_stocks['total_score'], color='skyblue')
            axes[0, 0].set_yticks(range(len(top_stocks)))
            axes[0, 0].set_yticklabels(top_stocks['name'], fontsize=8)
            axes[0, 0].set_xlabel('总分')
            axes[0, 0].set_title('Top 10 股票评分')
            axes[0, 0].invert_yaxis()  # 最高分在顶部
            
            # 2. 行业分布饼图
            if 'industry' in self.selected_stocks.columns:
                industry_counts = self.selected_stocks['industry'].value_counts()
                axes[0, 1].pie(industry_counts.values, labels=industry_counts.index, autopct='%1.1f%%', startangle=90)
                axes[0, 1].set_title('选中股票行业分布')
            
            # 3. 收益率与波动率散点图
            if '20d_return' in self.selected_stocks.columns and 'annualized_vol' in self.selected_stocks.columns:
                scatter = axes[1, 0].scatter(self.selected_stocks['annualized_vol'], 
                                            self.selected_stocks['20d_return'], 
                                            c=self.selected_stocks['total_score'], 
                                            cmap='viridis', 
                                            alpha=0.7)
                axes[1, 0].set_xlabel('年化波动率')
                axes[1, 0].set_ylabel('20日收益率')
                axes[1, 0].set_title('收益率与波动率关系')
                plt.colorbar(scatter, ax=axes[1, 0], label='总分')
            
            # 4. 评分雷达图（取前3只股票）
            if len(top_stocks) >= 3:
                # 评分类别
                categories = ['收益率', '波动率', '技术面', '估值']
                
                # 为每只股票准备数据
                stock_data = []
                for i in range(3):
                    stock_data.append([
                        top_stocks['returns_score'].iloc[i],
                        top_stocks['volatility_score'].iloc[i],
                        top_stocks['technical_score'].iloc[i],
                        top_stocks['valuation_score'].iloc[i]
                    ])
                
                # 计算每个类别的角度
                N = len(categories)
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # 闭合雷达图
                
                # 设置雷达图
                axes[1, 1] = plt.subplot(224, polar=True)
                
                # 绘制每只股票的数据
                colors = ['blue', 'green', 'red']
                for i in range(3):
                    values = stock_data[i]
                    values += values[:1]  # 闭合雷达图
                    axes[1, 1].plot(angles, values, linewidth=2, linestyle='solid', color=colors[i], label=top_stocks['name'].iloc[i])
                    axes[1, 1].fill(angles, values, color=colors[i], alpha=0.1)
                
                # 添加标签
                plt.xticks(angles[:-1], categories)
                axes[1, 1].set_ylim(0, 100)
                axes[1, 1].set_title('Top 3 股票多维度评分')
                axes[1, 1].legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            filepath = os.path.join(PLOTS_DIR, f"{filename}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"可视化图表已保存至: {filepath}")
            return filepath
        except Exception as e:
            print(f"可视化失败: {e}")
            return None
    
    def _load_stock_data(self, stock_code, start_date, end_date, retry_count=0, max_retries=3):
        """加载股票数据（内部方法），增强了错误处理和重试机制
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            retry_count: 当前重试次数
            max_retries: 最大重试次数
            
        Returns:
            df: 股票数据DataFrame
        """
        # 尝试从缓存加载数据
        cache_file = os.path.join(self.cache_dir, f"{stock_code}_{start_date}_{end_date}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                print(f"从缓存加载股票 {stock_code} 的数据")
                return df
            except Exception as e:
                print(f"从缓存加载数据失败: {e}")
        
        # 添加API调用间隔，避免频率限制
        time.sleep(self.request_interval)
        
        # 缓存不存在或加载失败，从Tushare获取
        try:
            print(f"尝试获取股票 {stock_code} 的数据 (第{retry_count+1}/{max_retries+1}次)")
            df = ts.pro_bar(
                ts_code=stock_code, 
                adj='qfq', 
                start_date=start_date, 
                end_date=end_date,
                timeout=10  # 添加超时设置
            )
            
            if df is not None and not df.empty:
                # 按日期排序
                df = df.sort_values('trade_date')
                df = df.reset_index(drop=True)
                
                # 保存到缓存
                df.to_csv(cache_file, index=False, encoding='utf-8-sig')
                print(f"获取并缓存股票 {stock_code} 的数据")
                return df
            else:
                print(f"未能获取到股票 {stock_code} 的数据")
                return None
        except Exception as e:
            error_msg = str(e)
            print(f"获取股票 {stock_code} 数据失败: {error_msg}")
            
            # 检查是否需要重试
            if retry_count < max_retries:
                # 检查错误类型决定重试策略
                if any(keyword in error_msg.lower() for keyword in ['timeout', 'connection', 'network', 'rate limit', '频率限制']):
                    # 计算指数退避的等待时间
                    wait_time = 2 * (2 ** retry_count) + random.uniform(0, 1)
                    print(f"遇到可恢复的错误，等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
                    # 递归重试
                    return self._load_stock_data(stock_code, start_date, end_date, retry_count + 1, max_retries)
                else:
                    # 对于非网络错误，不进行重试
                    print(f"遇到不可恢复的错误，不进行重试: {error_msg}")
                    return None
            else:
                print(f"达到最大重试次数，获取股票 {stock_code} 数据失败")
                return None

# 主函数示例
if __name__ == "__main__":
    # 创建选股器实例
    selector = StockSelector()
    
    # 设置分析日期
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
    
    # 分析股票池
    selector.analyze_stock_pool(start_date=start_date, end_date=end_date, limit=30)
    
    # 评分并选择股票
    selector.score_stocks()
    selected = selector.select_stocks(top_n=10)
    
    # 显示结果
    if selected is not None:
        print("\n选中的股票:")
        print(selected[['rank', 'name', 'total_score']])
    
    # 导出结果
    selector.export_results()
    
    # 可视化
    selector.visualize_results()