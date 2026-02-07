#!/usr/bin/env python3
"""
专业级金融数据可视化模块
=====================================

作为量化交易专家设计的数据可视化模块，提供专业级金融指标可视化功能。

主要功能:
- K线图（OHLC）渲染
- 成交量分析图表
- MACD技术指标计算与可视化
- RSI相对强弱指数
- 布林带指标
- 移动平均线系统
- 实时数据更新支持

作者: 量化交易专家系统
版本: v1.0
更新: 2024-11-14
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime, timedelta
import warnings
import io
import base64
import logging

# 设置中文字体和样式
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# 抑制警告
warnings.filterwarnings('ignore')

class FinancialVisualization:
    """专业级金融数据可视化类"""
    
    def __init__(self):
        """初始化金融可视化器"""
        self.logger = self._setup_logger()
        self.color_palette = {
            'bullish': '#26A69A',    # 牛市绿色
            'bearish': '#EF5350',    # 熊市红色
            'neutral': '#42A5F5',    # 中性蓝色
            'volume': '#FF9800',     # 成交量橙色
            'ma5': '#9C27B0',        # 5日均线紫色
            'ma10': '#3F51B5',       # 10日均线蓝色
            'ma20': '#4CAF50',       # 20日均线绿色
            'ma60': '#FF5722',       # 60日均线深红
            'macd': '#2196F3',       # MACD蓝色
            'signal': '#FF9800',     # 信号线橙色
            'rsi_oversold': '#4CAF50',  # RSI超卖绿色
            'rsi_overbought': '#F44336',  # RSI超买红色
            'bb_upper': '#9E9E9E',   # 布林带上轨灰色
            'bb_lower': '#9E9E9E'    # 布林带下轨灰色
        }
    
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger('FinancialVisualization')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def calculate_technical_indicators(self, df):
        """
        计算专业技术指标
        
        Args:
            df (pd.DataFrame): 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含技术指标的DataFrame
        """
        try:
            if df.empty:
                self.logger.warning("输入数据为空，无法计算技术指标")
                return df
            
            # 确保必要的列存在
            required_columns = ['open', 'high', 'low', 'close', 'vol']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"缺少必要列: {missing_columns}")
                return df
            
            # 复制数据避免修改原数据
            result_df = df.copy()
            
            # 1. 移动平均线
            result_df['ma5'] = result_df['close'].rolling(window=5).mean()
            result_df['ma10'] = result_df['close'].rolling(window=10).mean()
            result_df['ma20'] = result_df['close'].rolling(window=20).mean()
            result_df['ma60'] = result_df['close'].rolling(window=60).mean()
            
            # 2. MACD指标
            ema12 = result_df['close'].ewm(span=12).mean()
            ema26 = result_df['close'].ewm(span=26).mean()
            result_df['macd'] = ema12 - ema26
            result_df['macd_signal'] = result_df['macd'].ewm(span=9).mean()
            result_df['macd_histogram'] = result_df['macd'] - result_df['macd_signal']
            
            # 3. RSI指标
            delta = result_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result_df['rsi'] = 100 - (100 / (1 + rs))
            
            # 4. 布林带
            result_df['bb_middle'] = result_df['close'].rolling(window=20).mean()
            bb_std = result_df['close'].rolling(window=20).std()
            result_df['bb_upper'] = result_df['bb_middle'] + (bb_std * 2)
            result_df['bb_lower'] = result_df['bb_middle'] - (bb_std * 2)
            
            # 5. 成交量移动平均
            result_df['volume_ma5'] = result_df['vol'].rolling(window=5).mean()
            
            # 6. 价格变化率
            result_df['price_change'] = result_df['close'].pct_change()
            result_df['price_change_pct'] = result_df['price_change'] * 100
            
            self.logger.info("技术指标计算完成")
            return result_df
            
        except Exception as e:
            self.logger.error(f"计算技术指标时出错: {e}")
            return df
    
    def create_candlestick_chart(self, df, title="K线图", height=600):
        """
        创建专业K线图（使用Plotly）
        
        Args:
            df (pd.DataFrame): 包含OHLCV数据的DataFrame
            title (str): 图表标题
            height (int): 图表高度
            
        Returns:
            str: base64编码的HTML字符串
        """
        try:
            if df.empty or len(df) < 2:
                self.logger.warning("数据不足，无法生成K线图")
                return ""
            
            # 确保日期列正确格式化
            if 'trade_date' in df.columns:
                dates = pd.to_datetime(df['trade_date'])
            else:
                dates = pd.to_datetime(df.index)
            
            # 创建子图：K线 + 成交量
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('K线图', '成交量'),
                row_width=[0.7, 0.3]
            )
            
            # 1. K线图
            fig.add_trace(
                go.Candlestick(
                    x=dates,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='K线',
                    increasing_line_color=self.color_palette['bullish'],
                    decreasing_line_color=self.color_palette['bearish'],
                    increasing_fillcolor=self.color_palette['bullish'],
                    decreasing_fillcolor=self.color_palette['bearish']
                ),
                row=1, col=1
            )
            
            # 2. 移动平均线
            for ma, color, name in [('ma5', self.color_palette['ma5'], 'MA5'),
                                   ('ma10', self.color_palette['ma10'], 'MA10'),
                                   ('ma20', self.color_palette['ma20'], 'MA20'),
                                   ('ma60', self.color_palette['ma60'], 'MA60')]:
                if ma in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=df[ma],
                            mode='lines',
                            name=name,
                            line=dict(color=color, width=1.5)
                        ),
                        row=1, col=1
                    )
            
            # 3. 布林带
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df['bb_upper'],
                        mode='lines',
                        name='布林带上轨',
                        line=dict(color=self.color_palette['bb_upper'], width=1, dash='dash')
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df['bb_lower'],
                        mode='lines',
                        name='布林带下轨',
                        line=dict(color=self.color_palette['bb_lower'], width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(158, 158, 158, 0.1)'
                    ),
                    row=1, col=1
                )
            
            # 4. 成交量柱状图
            colors = [self.color_palette['bullish'] if close >= open_price 
                     else self.color_palette['bearish'] 
                     for close, open_price in zip(df['close'], df['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=df['vol'],
                    name='成交量',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # 设置布局
            fig.update_layout(
                title=dict(
                    text=title,
                    font=dict(size=18, color='#2c3e50'),
                    x=0.5
                ),
                xaxis_rangeslider_visible=False,
                height=height,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # 设置x轴格式
            fig.update_xaxes(
                row=2, col=1,
                title_text="日期",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)'
            )
            
            # 设置y轴格式
            fig.update_yaxes(
                row=1, col=1,
                title_text="价格",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)'
            )
            fig.update_yaxes(
                row=2, col=1,
                title_text="成交量",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)'
            )
            
            # 转换为HTML
            html_string = pyo.plot(fig, output_type='div', include_plotlyjs=True)
            self.logger.info(f"K线图生成成功: {len(df)}个数据点")
            return {"html": html_string}
            
        except Exception as e:
            self.logger.error(f"生成K线图时出错: {e}")
            return ""
    
    def create_technical_indicators_chart(self, df, title="技术指标分析", height=800):
        """
        创建技术指标综合图表
        
        Args:
            df (pd.DataFrame): 包含技术指标的DataFrame
            title (str): 图表标题
            height (int): 图表高度
            
        Returns:
            str: base64编码的HTML字符串
        """
        try:
            if df.empty or 'rsi' not in df.columns:
                self.logger.warning("数据不足或缺少RSI指标，无法生成技术指标图表")
                return ""
            
            dates = pd.to_datetime(df['trade_date']) if 'trade_date' in df.columns else pd.to_datetime(df.index)
            
            # 创建四个子图：价格+均线、MACD、RSI、布林带
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('价格与移动平均线', 'MACD', 'RSI', '布林带'),
                row_heights=[0.3, 0.25, 0.25, 0.2]
            )
            
            # 1. 价格与移动平均线
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=df['close'],
                    mode='lines',
                    name='收盘价',
                    line=dict(color='black', width=2)
                ),
                row=1, col=1
            )
            
            for ma, color, name in [('ma5', self.color_palette['ma5'], 'MA5'),
                                   ('ma10', self.color_palette['ma10'], 'MA10'),
                                   ('ma20', self.color_palette['ma20'], 'MA20')]:
                if ma in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=df[ma],
                            mode='lines',
                            name=name,
                            line=dict(color=color, width=1.5)
                        ),
                        row=1, col=1
                    )
            
            # 2. MACD
            if 'macd' in df.columns:
                # MACD线
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df['macd'],
                        mode='lines',
                        name='MACD',
                        line=dict(color=self.color_palette['macd'], width=2)
                    ),
                    row=2, col=1
                )
                
                # 信号线
                if 'macd_signal' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=df['macd_signal'],
                            mode='lines',
                            name='信号线',
                            line=dict(color=self.color_palette['signal'], width=1.5)
                        ),
                        row=2, col=1
                    )
                
                # MACD柱状图
                if 'macd_histogram' in df.columns:
                    colors = [self.color_palette['bullish'] if x > 0 else self.color_palette['bearish'] 
                             for x in df['macd_histogram']]
                    fig.add_trace(
                        go.Bar(
                            x=dates,
                            y=df['macd_histogram'],
                            name='MACD柱状图',
                            marker_color=colors,
                            opacity=0.7
                        ),
                        row=2, col=1
                    )
            
            # 3. RSI
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=df['rsi'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=3, col=1
            )
            
            # RSI超买超卖线
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
            
            # 4. 布林带
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df['close'],
                        mode='lines',
                        name='收盘价',
                        line=dict(color='black', width=2),
                        showlegend=False
                    ),
                    row=4, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df['bb_upper'],
                        mode='lines',
                        name='布林带上轨',
                        line=dict(color=self.color_palette['bb_upper'], width=1, dash='dash')
                    ),
                    row=4, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df['bb_lower'],
                        mode='lines',
                        name='布林带下轨',
                        line=dict(color=self.color_palette['bb_lower'], width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(158, 158, 158, 0.1)'
                    ),
                    row=4, col=1
                )
            
            # 设置布局
            fig.update_layout(
                title=dict(
                    text=title,
                    font=dict(size=18, color='#2c3e50'),
                    x=0.5
                ),
                height=height,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # 更新子图标题
            for i in range(1, 5):
                fig.update_yaxes(title_text="价格" if i == 1 else 
                                "MACD" if i == 2 else 
                                "RSI" if i == 3 else "价格",
                                row=i, col=1)
            
            # 设置x轴
            fig.update_xaxes(
                row=4, col=1,
                title_text="日期",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)'
            )
            
            # 转换为HTML
            html_string = pyo.plot(fig, output_type='div', include_plotlyjs=True)
            self.logger.info("技术指标图表生成成功")
            return {"html": html_string}
            
        except Exception as e:
            self.logger.error(f"生成技术指标图表时出错: {e}")
            return ""
    
    def create_volume_analysis_chart(self, df, title="成交量分析", height=400):
        """
        创建成交量分析图表
        
        Args:
            df (pd.DataFrame): 包含OHLCV数据的DataFrame
            title (str): 图表标题
            height (int): 图表高度
            
        Returns:
            str: base64编码的HTML字符串
        """
        try:
            if df.empty:
                self.logger.warning("数据不足，无法生成成交量分析图表")
                return ""
            
            dates = pd.to_datetime(df['trade_date']) if 'trade_date' in df.columns else pd.to_datetime(df.index)
            
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=('成交量分析',)
            )
            
            # 成交量柱状图
            colors = [self.color_palette['bullish'] if close >= open_price 
                     else self.color_palette['bearish'] 
                     for close, open_price in zip(df['close'], df['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=df['vol'],
                    name='成交量',
                    marker_color=colors,
                    opacity=0.8
                ),
                row=1, col=1
            )
            
            # 成交量移动平均线
            if 'volume_ma5' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df['volume_ma5'],
                        mode='lines',
                        name='成交量MA5',
                        line=dict(color=self.color_palette['neutral'], width=2, dash='dash')
                    ),
                    row=1, col=1
                )
            
            # 添加成交量统计信息
            avg_volume = df['vol'].mean()
            fig.add_hline(
                y=avg_volume,
                line_dash="dot",
                line_color="gray",
                annotation_text=f"平均成交量: {avg_volume:,.0f}",
                row=1, col=1
            )
            
            # 设置布局
            fig.update_layout(
                title=dict(
                    text=title,
                    font=dict(size=16, color='#2c3e50'),
                    x=0.5
                ),
                height=height,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=50, r=50, t=60, b=50)
            )
            
            fig.update_yaxes(title_text="成交量", row=1, col=1)
            fig.update_xaxes(title_text="日期", row=1, col=1)
            
            html_string = pyo.plot(fig, output_type='div', include_plotlyjs=True)
            self.logger.info("成交量分析图表生成成功")
            return {"html": html_string}
            
        except Exception as e:
            self.logger.error(f"生成成交量分析图表时出错: {e}")
            return ""
    
    def generate_professional_report(self, df, stock_code, stock_name="", output_dir="plots"):
        """
        生成专业级股票分析报告
        
        Args:
            df (pd.DataFrame): 股票数据
            stock_code (str): 股票代码
            stock_name (str): 股票名称
            output_dir (str): 输出目录
            
        Returns:
            dict: 包含各种图表HTML的字典
        """
        try:
            # 计算技术指标
            df_with_indicators = self.calculate_technical_indicators(df)
            
            # 生成标题
            title_suffix = f"({stock_code})" + (f" - {stock_name}" if stock_name else "")
            
            # 生成各种图表
            charts = {}
            
            # 1. K线图
            kline_title = f"专业K线分析 {title_suffix}"
            charts['kline'] = self.create_candlestick_chart(df_with_indicators, kline_title)
            
            # 2. 技术指标图
            tech_title = f"技术指标分析 {title_suffix}"
            charts['technical'] = self.create_technical_indicators_chart(df_with_indicators, tech_title)
            
            # 3. 成交量分析
            volume_title = f"成交量分析 {title_suffix}"
            charts['volume'] = self.create_volume_analysis_chart(df_with_indicators, volume_title)
            
            # 计算一些统计信息
            current_price = df['close'].iloc[-1] if len(df) > 0 else 0
            price_change = df['close'].iloc[-1] - df['close'].iloc[0] if len(df) > 1 else 0
            price_change_pct = (price_change / df['close'].iloc[0] * 100) if len(df) > 1 and df['close'].iloc[0] != 0 else 0
            
            latest_rsi = df_with_indicators['rsi'].iloc[-1] if 'rsi' in df_with_indicators.columns and len(df_with_indicators) > 0 else 0
            latest_macd = df_with_indicators['macd'].iloc[-1] if 'macd' in df_with_indicators.columns and len(df_with_indicators) > 0 else 0
            
            # 技术信号分析
            signals = self._analyze_technical_signals(df_with_indicators)
            
            summary = {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'latest_rsi': latest_rsi,
                'latest_macd': latest_macd,
                'technical_signals': signals,
                'data_points': len(df),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            charts['summary'] = summary
            
            self.logger.info(f"专业分析报告生成完成: {stock_code}")
            return charts
            
        except Exception as e:
            self.logger.error(f"生成专业报告时出错: {e}")
            return {}
    
    def _analyze_technical_signals(self, df):
        """分析技术信号"""
        try:
            signals = {
                'rsi_signal': '中性',
                'macd_signal': '中性',
                'price_vs_ma20': '中性',
                'bollinger_signal': '中性',
                'overall_signal': '中性'
            }
            
            if 'rsi' in df.columns and len(df) > 0:
                latest_rsi = df['rsi'].iloc[-1]
                if latest_rsi > 70:
                    signals['rsi_signal'] = '超买'
                elif latest_rsi < 30:
                    signals['rsi_signal'] = '超卖'
            
            if 'macd' in df.columns and 'macd_signal' in df.columns and len(df) > 1:
                latest_macd = df['macd'].iloc[-1]
                latest_signal = df['macd_signal'].iloc[-1]
                prev_macd = df['macd'].iloc[-2]
                
                if latest_macd > latest_signal and prev_macd <= df['macd_signal'].iloc[-2]:
                    signals['macd_signal'] = '金叉'
                elif latest_macd < latest_signal and prev_macd >= df['macd_signal'].iloc[-2]:
                    signals['macd_signal'] = '死叉'
            
            if 'close' in df.columns and 'ma20' in df.columns and len(df) > 0:
                latest_close = df['close'].iloc[-1]
                latest_ma20 = df['ma20'].iloc[-1]
                
                if latest_close > latest_ma20:
                    signals['price_vs_ma20'] = '站上均线'
                else:
                    signals['price_vs_ma20'] = '跌破均线'
            
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns and len(df) > 0:
                latest_close = df['close'].iloc[-1]
                latest_bb_upper = df['bb_upper'].iloc[-1]
                latest_bb_lower = df['bb_lower'].iloc[-1]
                
                if latest_close >= latest_bb_upper:
                    signals['bollinger_signal'] = '触及上轨'
                elif latest_close <= latest_bb_lower:
                    signals['bollinger_signal'] = '触及下轨'
                else:
                    signals['bollinger_signal'] = '通道内'
            
            # 综合信号
            bullish_signals = 0
            bearish_signals = 0
            
            if signals['rsi_signal'] == '超卖':
                bullish_signals += 1
            elif signals['rsi_signal'] == '超买':
                bearish_signals += 1
            
            if signals['macd_signal'] == '金叉':
                bullish_signals += 1
            elif signals['macd_signal'] == '死叉':
                bearish_signals += 1
            
            if signals['price_vs_ma20'] == '站上均线':
                bullish_signals += 1
            elif signals['price_vs_ma20'] == '跌破均线':
                bearish_signals += 1
            
            if bullish_signals > bearish_signals:
                signals['overall_signal'] = '看涨'
            elif bearish_signals > bullish_signals:
                signals['overall_signal'] = '看跌'
            
            return signals
            
        except Exception as e:
            self.logger.error(f"分析技术信号时出错: {e}")
            return {}


# 工厂函数：创建金融可视化器实例
def create_financial_visualizer():
    """创建金融可视化器实例"""
    return FinancialVisualization()


# 测试函数
def test_financial_visualization():
    """测试金融可视化功能"""
    try:
        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', end='2024-11-14', freq='D')
        
        # 生成模拟OHLCV数据
        base_price = 100
        test_data = []
        
        for i, date in enumerate(dates):
            # 模拟价格波动
            change = np.random.normal(0, 0.02)  # 2%日波动
            base_price *= (1 + change)
            
            # 生成OHLC
            high = base_price * (1 + abs(np.random.normal(0, 0.01)))
            low = base_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = base_price * (1 + np.random.normal(0, 0.005))
            close = base_price
            volume = np.random.randint(1000000, 10000000)
            
            test_data.append({
                'trade_date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'vol': volume
            })
        
        df = pd.DataFrame(test_data)
        
        # 创建可视化器
        viz = create_financial_visualizer()
        
        # 生成专业报告
        charts = viz.generate_professional_report(df, 'TEST001', '测试股票')
        
        print("=== 金融可视化测试结果 ===")
        print(f"数据点数: {len(df)}")
        print(f"K线图生成: {'成功' if charts.get('kline') else '失败'}")
        print(f"技术指标图生成: {'成功' if charts.get('technical') else '失败'}")
        print(f"成交量分析图生成: {'成功' if charts.get('volume') else '失败'}")
        
        if 'summary' in charts:
            summary = charts['summary']
            print(f"当前价格: {summary['current_price']:.2f}")
            print(f"价格变化: {summary['price_change_pct']:.2f}%")
            print(f"最新RSI: {summary['latest_rsi']:.2f}")
            print(f"技术信号: {summary['technical_signals']}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False


if __name__ == "__main__":
    test_financial_visualization()