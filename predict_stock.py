#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票预测推理模块

该模块实现股票价格预测的推理功能，支持单股预测和批量预测。

主要功能：
- 单股价格预测
- 批量股票预测
- K线图预测可视化
- 模型加载和管理
- 预测结果分析和展示

核心特性：
- 支持多种LSTM模型类型
- 智能模型加载和兼容性检查
- 高效的批量预测算法
- 专业级K线图可视化
- 预测结果统计分析

作者：AI助手
版本：v1.4
更新时间：2024-11-14
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import mplfinance as mpf

from config import *
from stock_data import get_latest_stock_data

# 加载训练好的模型
def load_trained_model(stock_code, model_type='baseline', model_dir=None):
    """
    加载训练好的股票预测模型
    
    参数:
    stock_code: 股票代码
    model_type: 模型类型，可选 'baseline', 'lstm', 'kline_lstm'
    model_dir: 模型存储目录，如果是完整文件路径则直接使用
    
    返回:
    加载的模型对象，如果加载失败则返回None
    """
    try:
        model_path = None
        
        # 检查model_dir是否为完整的模型文件路径
        if model_dir and os.path.isfile(model_dir):
            # 如果model_dir是一个文件，直接使用它作为模型路径
            model_path = model_dir
        else:
            # 如果未指定模型目录，使用默认目录
            if model_dir is None:
                model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
            
            # 确保模型目录存在
            if not os.path.exists(model_dir):
                print(f"模型目录不存在: {model_dir}")
                return None
            
            # 根据模型类型构建文件名
            model_filename = f"{stock_code}_{model_type}_best.pth"
            model_path = os.path.join(model_dir, model_filename)
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return None
        
        # 打印详细的调试信息
        print(f"正在加载模型文件: {model_path}")
        print(f"文件大小: {os.path.getsize(model_path)} 字节")
        
        # 加载模型文件
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 检查是否是字典格式的模型文件
        if isinstance(checkpoint, dict):
            # 尝试从字典中获取模型
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                # 如果只有状态字典，尝试导入模型类并重建
                try:
                    from stock_model import get_model
                    # 尝试获取模型参数
                    model_type = checkpoint.get('model_type', model_type)
                    input_size = checkpoint.get('input_size', 1)
                    hidden_size = checkpoint.get('hidden_size', 64)
                    output_size = checkpoint.get('output_size', 1)
                    num_layers = checkpoint.get('num_layers', 1)
                    
                    # 创建模型实例
                    model = get_model(model_type, input_size, hidden_size, output_size, num_layers)
                    # 加载模型状态
                    model.load_state_dict(checkpoint['model_state_dict'])
                except Exception as inner_e:
                    print(f"重建模型失败: {str(inner_e)}")
                    # 如果无法重建，直接返回字典（可能需要在其他地方处理）
                    return checkpoint
            else:
                # 如果字典中没有预期的键，返回字典
                print("警告: 模型文件是字典格式但缺少预期的键")
                return checkpoint
        else:
            # 直接是模型对象
            model = checkpoint
        
        print(f"成功加载模型: {model_path}")
        return model
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 使用模型进行预测
def predict_stock_price(model, stock_code, device, sequence_length=60, model_type='baseline', prediction_days=1):
    """
    使用训练好的模型预测股票价格
    
    参数:
    model: 训练好的模型
    stock_code: 股票代码
    device: 运行设备
    sequence_length: 序列长度
    model_type: 模型类型
    prediction_days: 预测天数
    
    返回:
    预测结果，包括预测价格、最新收盘价和价格变化百分比
    """
    try:
        # 获取最新的股票数据
        latest_sequence, latest_close, metadata = get_latest_stock_data(stock_code, sequence_length, model_type)
        
        if latest_sequence is None:
            print(f"无法获取足够的历史数据用于预测: {stock_code}")
            return None
        
        # 对于lstm模型，直接使用返回的latest_sequence作为输入
        X = latest_sequence.to(device) if isinstance(latest_sequence, torch.Tensor) else latest_sequence
        
        # 获取缩放器
        if metadata and 'scaler' in metadata:
            scaler = metadata['scaler']
        
        # 设置模型为评估模式
        model.eval()
        
        with torch.no_grad():
            if model_type == 'kline_lstm':
                # K线模型预测多特征
                predictions = []
                input_sequence = X.clone() if isinstance(X, torch.Tensor) else X
                
                # 获取目标缩放器
                target_scaler = metadata.get('target_scaler', None)
                df = metadata.get('df', None)
                
                for i in range(prediction_days):
                    # 进行预测
                    pred = model(input_sequence)
                    
                    # 反标准化预测结果
                    if target_scaler is not None and isinstance(pred, torch.Tensor):
                        pred_np = pred.cpu().numpy()
                        pred_unscaled = target_scaler.inverse_transform(pred_np)[0]
                        predictions.append(pred_unscaled)
                    else:
                        # 如果没有缩放器，直接使用预测值
                        predictions.append(pred.cpu().numpy()[0] if isinstance(pred, torch.Tensor) else pred)
                    
                    # 更新输入序列
                    if i < prediction_days - 1 and isinstance(pred, torch.Tensor):
                        # 这里简化处理，实际应该根据模型需求构建新的特征向量
                        # 移除第一个时间步，添加新的预测特征
                        if len(input_sequence.shape) >= 3:
                            input_sequence = torch.cat([
                                input_sequence[:, 1:, :],
                                pred.unsqueeze(1)
                            ], dim=1)
                
                # 使用metadata中的最新K线数据
                latest_kline = metadata.get('latest_kline', None)
                if latest_kline and isinstance(latest_kline, (list, np.ndarray)) and len(latest_kline) >= 4:
                    latest_close = latest_kline[3]  # 收盘价通常是第四个元素
                
                return np.array(predictions), latest_close, metadata
            else:
                # 基线模型预测
                try:
                    # 检查模型类型并进行预测
                    if isinstance(model, torch.nn.Module):
                        prediction = model(X)
                        
                        # 如果模型返回的是张量，转换为数值
                        if isinstance(prediction, torch.Tensor):
                            predicted_price = prediction.item()
                        else:
                            predicted_price = prediction
                    else:
                        # 如果模型不是Module类型，可能是字典或其他格式，需要特殊处理
                        print(f"警告: 模型类型不是Module: {type(model)}")
                        # 尝试从模型字典中提取预测相关信息
                        if isinstance(model, dict) and 'prediction' in model:
                            predicted_price = model['prediction']
                        else:
                            # 使用默认预测逻辑
                            predicted_price = latest_close  # 使用最新收盘价作为默认预测
                    
                    # 计算价格变化百分比
                    price_change_percent = ((predicted_price - latest_close) / latest_close) * 100
                    
                    return predicted_price, latest_close, price_change_percent
                except Exception as e:
                    print(f"基线模型预测过程出错: {str(e)}")
                    # 返回简单的预测结果作为备选
                    return latest_close, latest_close, 0.0
    except Exception as e:
        print(f"预测股票价格时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 批量预测多只股票
def batch_predict_stocks(model, stock_codes, device, sequence_length=60, model_type='baseline', prediction_days=1):
    """
    批量预测多只股票的价格
    
    参数:
    model: 训练好的模型
    stock_codes: 股票代码列表
    device: 运行设备
    sequence_length: 序列长度
    model_type: 模型类型
    prediction_days: 预测天数
    
    返回:
    包含所有预测结果的字典
    """
    results = {}
    
    for stock_code in stock_codes:
        try:
            print(f"正在预测股票: {stock_code}")
            result = predict_stock_price(model, stock_code, device, sequence_length, model_type, prediction_days)
            
            if result is not None:
                if model_type == 'kline_lstm':
                    predictions, latest_close, _ = result
                    results[stock_code] = {
                        'predictions': predictions.tolist(),
                        'latest_close': latest_close
                    }
                else:
                    predicted_price, latest_close, price_change_percent = result
                    results[stock_code] = {
                        'predicted_price': predicted_price,
                        'latest_close': latest_close,
                        'price_change_percent': price_change_percent
                    }
            else:
                print(f"无法预测股票: {stock_code}")
        except Exception as e:
            print(f"预测股票 {stock_code} 时出错: {str(e)}")
    
    return results

# 绘制K线预测图
def plot_kline_prediction(stock_code, metadata, predictions, save_path=None):
    """
    绘制K线预测图
    
    参数:
    stock_code: 股票代码
    metadata: 元数据，包含最新K线数据和日期
    predictions: 预测的K线数据
    save_path: 图表保存路径
    """
    try:
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 准备历史数据（这里简化为只使用最新的K线）
        last_kline = metadata['latest_kline']
        last_date = datetime.strptime(metadata['last_date'], '%Y-%m-%d')
        
        # 创建历史数据点
        dates = [last_date]
        opens = [last_kline[0]]
        highs = [last_kline[1]]
        lows = [last_kline[2]]
        closes = [last_kline[3]]
        volumes = [last_kline[4]]
        
        # 添加预测数据
        for i, pred in enumerate(predictions):
            pred_date = last_date + timedelta(days=i+1)
            dates.append(pred_date)
            opens.append(pred[0])
            highs.append(pred[1])
            lows.append(pred[2])
            closes.append(pred[3])
            volumes.append(pred[4])
        
        # 创建DataFrame
        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes
        }, index=dates)
        
        # 创建蜡烛图
        mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc)
        
        # 绘制图表
        mpf.plot(df, type='candle', style=s, volume=True, ax=ax, 
                 title=f'{stock_code} 股价预测', 
                 ylabel='价格', ylabel_lower='成交量')
        
        # 标记预测开始位置
        ax.axvline(x=df.index[0], color='blue', linestyle='--', label='预测起点')
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"K线预测图已保存至: {save_path}")
        else:
            plt.show()
    except Exception as e:
        print(f"绘制K线预测图时出错: {str(e)}")

# 绘制预测结果与实际价格对比图
def plot_prediction_result(stock_code, actual_prices, predicted_prices, save_path=None):
    """
    绘制预测结果与实际价格对比图
    
    参数:
    stock_code: 股票代码
    actual_prices: 实际价格
    predicted_prices: 预测价格
    save_path: 图表保存路径
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # 绘制实际价格
        plt.plot(range(len(actual_prices)), actual_prices, 'b-', label='实际价格')
        
        # 绘制预测价格（只在最后一个点）
        plt.scatter([len(actual_prices)], predicted_prices, color='r', label='预测价格')
        
        plt.title(f'{stock_code} 股价预测结果')
        plt.xlabel('时间')
        plt.ylabel('价格')
        plt.legend()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测结果图已保存至: {save_path}")
        else:
            plt.show()
    except Exception as e:
        print(f"绘制预测结果图时出错: {str(e)}")

# 添加数值标签
def add_value_labels(ax, values, positions):
    """
    向图表添加数值标签
    
    参数:
    ax: 图表轴对象
    values: 要显示的数值列表
    positions: 标签位置列表
    """
    for i, (value, pos) in enumerate(zip(values, positions)):
        ax.text(pos, value, f'{value:.2f}', ha='center', va='bottom')

# 设置图表属性
def set_chart_properties(ax, title, xlabel, ylabel):
    """
    设置图表的标题和轴标签
    
    参数:
    ax: 图表轴对象
    title: 图表标题
    xlabel: x轴标签
    ylabel: y轴标签
    """
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

# 计算预测变化
def calculate_prediction_changes(latest_price, predicted_prices):
    """
    计算预测价格相对于最新价格的变化
    
    参数:
    latest_price: 最新价格
    predicted_prices: 预测价格列表
    
    返回:
    变化百分比列表
    """
    changes = []
    for pred_price in predicted_prices:
        change = ((pred_price - latest_price) / latest_price) * 100
        changes.append(change)
    return changes

# 获取默认模型路径
def get_default_model_path(stock_code, model_type='baseline'):
    """
    获取默认模型文件路径
    
    参数:
    stock_code: 股票代码
    model_type: 模型类型
    
    返回:
    模型文件的绝对路径
    """
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    model_filename = f"{stock_code}_{model_type}_best.pth"
    return os.path.join(model_dir, model_filename)

# 主预测函数
def main_predict(stock_code, model_type='baseline', sequence_length=60, prediction_days=1, model_dir=None):
    """
    主预测函数，用于预测指定股票的未来价格
    
    参数:
    stock_code: 股票代码
    model_type: 模型类型，可选 'baseline', 'lstm', 'kline_lstm'
    sequence_length: 序列长度
    prediction_days: 预测天数
    model_dir: 模型存储目录
    
    返回:
    (是否成功, 预测数据) 元组
    """
    try:
        # 检查GPU是否可用
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 加载模型
        model = load_trained_model(stock_code, model_type, model_dir)
        
        if model is None:
            print("无法加载模型，预测失败")
            return False, None
        
        # 检查模型是否是torch.nn.Module类型，如果是则移动到设备
        if isinstance(model, torch.nn.Module):
            model.to(device)
        else:
            print("警告: 模型不是torch.nn.Module类型，无法移动到指定设备")
        
        # 进行预测
        result = predict_stock_price(model, stock_code, device, sequence_length, model_type, prediction_days)
        
        if result is not None:
            # 创建图像保存目录
            plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            
            if model_type == 'kline_lstm':
                # K线模型预测结果
                if isinstance(result, tuple) and len(result) == 3:
                    predictions, latest_close, metadata = result
                    
                    # 打印预测结果
                    print(f"股票代码: {stock_code}")
                    print(f"最新收盘价: {latest_close:.2f}")
                    print(f"未来{prediction_days}天K线预测结果:")
                    
                    # 打印每天的预测结果
                    for i, pred in enumerate(predictions):
                        if len(pred) >= 5:
                            open_price, high, low, close, volume = pred[:5]
                            change_percent = ((close - latest_close) / latest_close) * 100
                            print(f"  第{i+1}天: 开盘={open_price:.2f}, 最高={high:.2f}, 最低={low:.2f}, 收盘={close:.2f}, 成交量={volume:.2f}, 涨跌幅={change_percent:+.2f}%")
                            latest_close = close  # 更新参考价格
                    
                    # 绘制K线预测图
                    kline_plot_path = os.path.join(plot_dir, f'{stock_code}_kline_prediction.png')
                    plot_kline_prediction(stock_code, metadata, predictions, kline_plot_path)
                    
                    # 返回预测数据和图表路径
                    prediction_data = {
                        'type': 'kline',
                        'predictions': predictions.tolist(),
                        'plot_path': kline_plot_path,
                        'metadata': {
                            'stock_code': stock_code,
                            'latest_close': metadata['latest_kline'][3],
                            'prediction_days': prediction_days
                        }
                    }
                    
                    return True, prediction_data
                else:
                    print(f"警告: K线模型预测结果格式不正确: {result}")
                    return False, None
            else:
                # 基线模型预测结果
                if isinstance(result, tuple) and len(result) == 3:
                    try:
                        predicted_price, latest_close, price_change_percent = result
                        
                        # 打印预测结果
                        print(f"股票代码: {stock_code}")
                        
                        # 检查latest_close是否为数值
                        if isinstance(latest_close, (int, float)):
                            print(f"最新收盘价: {latest_close:.2f}")
                        else:
                            print(f"最新收盘价: {latest_close}")
                        
                        # 检查predicted_price的类型
                        if isinstance(predicted_price, list):
                            # 如果是列表，取第一个元素（假设是收盘价）
                            predicted_close = predicted_price[0] if predicted_price else 0
                            if isinstance(predicted_close, (int, float)):
                                print(f"预测收盘价: {predicted_close:.2f}")
                            else:
                                print(f"预测收盘价: {predicted_close}")
                        elif isinstance(predicted_price, (int, float)):
                            print(f"预测收盘价: {predicted_price:.2f}")
                        else:
                            print(f"预测收盘价: {predicted_price}")
                        
                        # 检查price_change_percent的类型
                        if isinstance(price_change_percent, (int, float)):
                            print(f"预测{'上涨' if price_change_percent > 0 else '下跌'}: {abs(price_change_percent):.2f}%")
                        else:
                            print(f"预测变化: {price_change_percent}")
                    except Exception as e:
                        print(f"处理预测结果时出错: {str(e)}")
                        return False, None
                else:
                    print(f"警告: 预测结果格式不完整: {result}")
                    return False, None
        else:
            print(f"警告: 预测结果类型不正确: {type(result)}")
            return False, None
        
        # 绘制预测结果图
        prediction_plot_path = os.path.join(plot_dir, f'{stock_code}_prediction.png')
        # 确保metadata变量已定义
        df = None
        if 'metadata' in locals() and metadata:
            df = metadata.get('df', None)
        elif isinstance(result, tuple) and len(result) >= 3:
            # 尝试从result获取metadata
            metadata_candidate = result[2] if isinstance(result[2], dict) else None
            if metadata_candidate:
                df = metadata_candidate.get('df', None)
        
        if df is not None and 'close' in df:
            plot_prediction_result(stock_code, df['close'].values, predicted_price, prediction_plot_path)
        else:
            print("警告: 无法获取收盘价数据，跳过图表绘制")
        
        # 返回预测数据和图表路径
        prediction_data = {
            'type': 'baseline',
            'predicted_price': predicted_price,
            'latest_close': latest_close,
            'price_change_percent': price_change_percent,
            'plot_path': prediction_plot_path
        }
        
        return True, prediction_data
        
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

# 主函数，用于示例
def main():
    # 示例参数
    stock_code = '000001.SZ'  # 平安银行
    model_type = 'baseline'   # 模型类型
    sequence_length = 60      # 序列长度
    prediction_days = 1       # 预测天数
    
    # 执行预测
    success, result = main_predict(stock_code, model_type, sequence_length, prediction_days)
    
    if success and result:
        print("\n预测成功完成!")
        print(f"预测数据: {result}")
    else:
        print("预测失败")

if __name__ == "__main__":
    main()