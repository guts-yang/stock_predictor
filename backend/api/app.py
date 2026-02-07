# 导入所需库
from flask import Flask, render_template, request, jsonify, send_file, make_response, send_from_directory, redirect, url_for
import os
import sys
import platform
import subprocess
import psutil
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import io
import base64
import urllib
import shutil
from datetime import datetime, timedelta
import time
import webbrowser

# 添加项目根目录到Python路径（支持新的模块结构）
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# 确保目录存在
def ensure_directories():
    """确保必要的目录存在"""
    directories = ['data', MODELS_DIR, PLOTS_DIR, RESULTS_DIR, 'logs', 'static', 'templates']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# 添加股票数据模块导入，用于获取股票名称
from backend.core.stock_data import get_stock_name

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入图像处理相关库
from PIL import Image
import base64
from io import BytesIO
import uuid
import json

# 导入项目模块
from backend.core.stock_data import fetch_stock_data, save_stock_data, get_latest_stock_data, get_device
from backend.core.config import DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_LAYERS, DEFAULT_SEQUENCE_LENGTH, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, DEFAULT_PATIENCE, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, TUSHARE_TOKEN, MAX_BATCH_PREDICTION, SYSTEM_VERSION
from backend.services.train_stock_model import train_stock_model
from backend.services.predict_stock import main_predict, batch_predict_stocks, load_trained_model, predict_stock_price, get_default_model_path, plot_kline_prediction
from backend.services.stock_selector import StockSelector
from backend.services.financial_visualization import create_financial_visualizer

# 创建Flask应用
app = Flask(__name__,
            template_folder='../../frontend/templates',
            static_folder='../../frontend/static')
app.config['JSON_SORT_KEYS'] = False  # 保持JSON响应中键的顺序
app.config['JSON_AS_ASCII'] = False    # 确保中文正常显示
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 设置最大内容长度为16MB

# 配置CORS支持
from flask_cors import CORS
CORS(app, 
     resources={r"/api/*": {"origins": ["http://127.0.0.1:5000", "http://localhost:5000", "http://127.0.0.1", "http://localhost"]}},
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "X-Requested-With", "Accept", "Authorization", "Cache-Control"],
     supports_credentials=True)

# 配置日志
import logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'app.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
app.logger = logging.getLogger('stock_predictor')

# 计算目录大小
def get_directory_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size

# 将字节转换为可读格式
def format_size(size_bytes):
    """格式化文件大小显示，添加错误处理和日志记录"""
    try:
        if size_bytes is None or size_bytes < 0:
            return '0 B'
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        size = float(size_bytes)
        unit_index = 0
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        return "{0:.2f} {1}".format(size, units[unit_index])
    except Exception as e:
        # 添加日志记录
        app.logger.warning(f"格式化文件大小时出错: {str(e)}")
        return "未知大小"

# 设置模板文件夹
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

# 确保目录存在
os.makedirs(template_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# 首页路由
@app.route('/')
def index():
    return render_template('index.html')

# 智能选股页面路由
@app.route('/stock_selector')
def stock_selector_page():
    return render_template('stock_selector.html')

# 获取股票数据路由
@app.route('/api/fetch_data', methods=['POST'])
def fetch_data():
    """获取股票数据API - 增强版，添加参数验证和错误处理"""
    try:
        # 获取并验证输入参数
        data = request.json
        if not data:
            app.logger.warning('请求数据为空')
            return jsonify({'success': False, 'message': '请求数据不能为空'}), 400
            
        stock_code = data.get('stock_code')
        if not stock_code:
            app.logger.warning('股票代码为空')
            return jsonify({'success': False, 'message': '股票代码不能为空'}), 400
        
        start_date = data.get('start_date', DEFAULT_START_DATE)
        end_date = data.get('end_date', DEFAULT_END_DATE)
        save_dir = data.get('save_dir', 'data')
        
        app.logger.info(f'正在获取股票数据: {stock_code}, 日期范围: {start_date} - {end_date}')
        
        # 调用数据获取函数
        try:
            df = fetch_stock_data(stock_code, start_date, end_date)
        except Exception as e:
            app.logger.error(f'调用fetch_stock_data出错: {e}', exc_info=True)
            return jsonify({'success': False, 'message': f'数据获取失败: {str(e)}'})
        
        if df is None or df.empty:
            app.logger.warning(f'无法获取股票数据: {stock_code}, 可能股票代码无效或日期范围没有交易数据')
            return jsonify({'success': False, 'message': '无法获取股票数据，请检查股票代码和日期范围'})
        
        # 保存数据
        try:
            success = save_stock_data(stock_code, start_date, end_date, save_dir)
            if not success:
                app.logger.warning(f'数据保存失败: {stock_code}')
                # 即使保存失败也返回数据，因为获取已经成功
                # 准备数据用于前端显示
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                stock_data = {
                    'dates': df['trade_date'].dt.strftime('%Y-%m-%d').tolist(),
                    'close_prices': [float(x) for x in df['close'].tolist()],
                    'open_prices': [float(x) for x in df['open'].tolist()],
                    'high_prices': [float(x) for x in df['high'].tolist()],
                    'low_prices': [float(x) for x in df['low'].tolist()],
                    'volume': [float(x) for x in df['vol'].tolist()],
                    'ma5': [float(x) for x in df['ma5'].tolist()] if 'ma5' in df.columns else [],
                    'ma10': [float(x) for x in df['ma10'].tolist()] if 'ma10' in df.columns else []
                }
                return jsonify({'success': True, 'data': stock_data, 'message': '数据获取成功但保存失败'})
        except Exception as e:
            app.logger.error(f'保存股票数据出错: {e}')
            # 即使保存出错也返回数据
        
        # 准备数据用于前端显示，确保数据类型安全
        try:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            stock_data = {
                'dates': df['trade_date'].dt.strftime('%Y-%m-%d').tolist() if not df.empty else [],
                'close_prices': [float(x) for x in df['close'].tolist()] if 'close' in df.columns else [],
                'open_prices': [float(x) for x in df['open'].tolist()] if 'open' in df.columns else [],
                'high_prices': [float(x) for x in df['high'].tolist()] if 'high' in df.columns else [],
                'low_prices': [float(x) for x in df['low'].tolist()] if 'low' in df.columns else [],
                'volume': [float(x) for x in df['vol'].tolist()] if 'vol' in df.columns else [],
                'ma5': [float(x) for x in df['ma5'].tolist()] if 'ma5' in df.columns else [],
                'ma10': [float(x) for x in df['ma10'].tolist()] if 'ma10' in df.columns else []
            }
            
            app.logger.info(f'股票数据获取成功: {stock_code}, 数据点数: {len(df)}')
            return jsonify({'success': True, 'data': stock_data, 'message': '数据获取成功'})
        except Exception as e:
            app.logger.error(f'数据格式转换出错: {e}')
            return jsonify({'success': False, 'message': '数据处理失败，请稍后重试'})
    except ValueError as e:
        app.logger.error(f'数据格式错误: {e}')
        return jsonify({'success': False, 'message': f'数据格式错误: {str(e)}'}), 400
    except Exception as e:
        app.logger.error(f'获取股票数据API整体出错: {e}', exc_info=True)
        return jsonify({'success': False, 'message': '服务器内部错误，请稍后再试'}), 500

# 训练模型路由
@app.route('/api/train_model', methods=['POST'])
def train_model():
    """训练模型API - 增强版，添加完整的参数验证和错误处理"""
    try:
        # 获取并验证请求数据
        data = request.json
        if not data:
            app.logger.warning('请求数据为空')
            return jsonify({'success': False, 'message': '请求数据不能为空'}), 400
        
        # 验证必要参数
        stock_code = data.get('stock_code')
        if not stock_code:
            app.logger.warning('股票代码为空')
            return jsonify({'success': False, 'message': '股票代码不能为空'}), 400
        
        # 验证股票代码格式
        if not (stock_code.endswith('.SZ') or stock_code.endswith('.SH')):
            app.logger.warning(f'股票代码格式错误: {stock_code}')
            return jsonify({'success': False, 'message': '股票代码格式错误，请使用.SZ或.SH后缀'}), 400
        
        # 获取并验证日期参数
        start_date = data.get('start_date', DEFAULT_START_DATE)
        end_date = data.get('end_date', DEFAULT_END_DATE)
        
        # 验证模型类型
        model_type = data.get('model_type', 'baseline')
        valid_model_types = ['lstm', 'baseline', 'gru', 'kline_lstm']
        if model_type not in valid_model_types:
            app.logger.warning(f'无效的模型类型: {model_type}')
            return jsonify({'success': False, 'message': f'无效的模型类型，支持的类型: {valid_model_types}'}), 400
        
        # 获取模型参数并进行类型和范围验证
        try:
            hidden_size = int(data.get('hidden_size', DEFAULT_HIDDEN_SIZE))
            num_layers = int(data.get('num_layers', DEFAULT_NUM_LAYERS))
            sequence_length = int(data.get('sequence_length', DEFAULT_SEQUENCE_LENGTH))
            batch_size = int(data.get('batch_size', DEFAULT_BATCH_SIZE))
            learning_rate = float(data.get('learning_rate', DEFAULT_LEARNING_RATE))
            epochs = int(data.get('epochs', DEFAULT_EPOCHS))
            patience = int(data.get('patience', DEFAULT_PATIENCE))
            
            # 参数边界检查
            if hidden_size <= 0 or hidden_size > 1024:
                raise ValueError("隐藏层大小必须在1-1024之间")
            if num_layers <= 0 or num_layers > 10:
                raise ValueError("LSTM层数必须在1-10之间")
            if sequence_length <= 0 or sequence_length > 365:
                raise ValueError("序列长度必须在1-365之间")
            if batch_size <= 0 or batch_size > 1024:
                raise ValueError("批次大小必须在1-1024之间")
            if learning_rate <= 0 or learning_rate > 1.0:
                raise ValueError("学习率必须在0-1之间")
            if epochs <= 0 or epochs > 1000:
                raise ValueError("训练轮数必须在1-1000之间")
            if patience <= 0 or patience > 100:
                raise ValueError("早停耐心值必须在1-100之间")
        except ValueError as e:
            app.logger.warning(f'模型参数错误: {str(e)}')
            return jsonify({'success': False, 'message': f'模型参数错误: {str(e)}'}), 400
        except Exception as e:
            app.logger.warning(f'参数解析错误: {str(e)}')
            return jsonify({'success': False, 'message': f'参数解析错误: {str(e)}'}), 400
        
        app.logger.info(f"开始训练模型: 股票代码={stock_code}, 模型类型={model_type}, 参数=[隐藏层={hidden_size}, 层数={num_layers}, 序列长度={sequence_length}, 批次={batch_size}, 学习率={learning_rate}, 轮数={epochs}, 早停={patience}]")
        
        # 获取设备
        try:
            device = get_device()
            app.logger.info(f'使用设备: {device}')
        except Exception as e:
            app.logger.error(f'获取设备信息失败: {str(e)}')
            return jsonify({'success': False, 'message': f'获取设备信息失败: {str(e)}'}), 500
        
        # 调用训练函数
        try:
            model_path = train_stock_model(
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                model_type=model_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                sequence_length=sequence_length,
                batch_size=batch_size,
                learning_rate=learning_rate,
                epochs=epochs,
                patience=patience,
                device=device
            )
            app.logger.info(f"模型训练完成，保存路径: {model_path}")
        except Exception as e:
            error_str = str(e)
            app.logger.error(f"模型训练过程出错: {e}", exc_info=True)
            
            # 根据错误类型提供更具体的错误信息
            if 'Token' in error_str or 'token' in error_str:
                detailed_msg = f'训练模型失败: {error_str}，请检查Tushare API Token配置是否正确'
            elif '权限' in error_str:
                detailed_msg = f'训练模型失败: {error_str}，请升级Tushare账户权限'
            elif '数据' in error_str or 'data' in error_str:
                detailed_msg = f'训练模型失败: {error_str}，请检查股票代码是否正确或日期范围是否合理'
            elif '400' in error_str:
                detailed_msg = f'训练模型失败: {error_str}，请求参数错误，请检查输入参数是否符合要求'
            elif '网络' in error_str or '连接' in error_str or 'timeout' in error_str:
                detailed_msg = f'训练模型失败: {error_str}，网络连接问题，请检查网络或稍后重试'
            else:
                detailed_msg = f'训练模型失败: {error_str}'
                
            return jsonify({'success': False, 'message': detailed_msg}), 500
        
        if model_path:
            app.logger.info(f'模型训练成功: {model_path}')
            return jsonify({'success': True, 'model_path': model_path, 'message': '模型训练成功'})
        else:
            app.logger.warning(f'模型训练失败: {stock_code}，模型路径未返回')
            return jsonify({'success': False, 'message': '模型训练失败，请检查：1) 数据是否充分 2) 模型参数设置是否合理 3) 网络连接是否正常 4) Tushare API是否可用'}), 500
    except Exception as e:
        app.logger.error(f"训练模型API出错: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'服务器内部错误: {str(e)}'}), 500

# 预测单只股票路由
@app.route('/api/predict_single', methods=['POST'])
def predict_single():
    """预测单只股票价格 - 增强版，添加完整的参数验证和错误处理"""
    try:
        # 获取并验证请求数据
        data = request.json
        if not data:
            app.logger.warning('请求数据为空')
            return jsonify({'success': False, 'message': '请求数据不能为空'}), 400
        
        # 验证必要参数
        stock_code = data.get('stock_code')
        if not stock_code:
            app.logger.warning('股票代码为空')
            return jsonify({'success': False, 'message': '股票代码不能为空'}), 400
        
        # 验证股票代码格式
        if not (stock_code.endswith('.SZ') or stock_code.endswith('.SH')):
            app.logger.warning(f'股票代码格式错误: {stock_code}')
            return jsonify({'success': False, 'message': '股票代码格式错误，请使用.SZ或.SH后缀'}), 400
        
        # 验证模型类型
        model_type = data.get('model_type', 'baseline')
        valid_model_types = ['lstm', 'baseline', 'gru', 'kline_lstm']
        if model_type not in valid_model_types:
            app.logger.warning(f'无效的模型类型: {model_type}')
            return jsonify({'success': False, 'message': f'无效的模型类型，支持的类型: {valid_model_types}'}), 400
        
        # 获取可选参数
        model_path = data.get('model_path')
        
        # 设置预测天数
        if model_type == 'kline_lstm':
            try:
                prediction_days = int(data.get('prediction_days', 5))
                if prediction_days <= 0 or prediction_days > 30:
                    app.logger.warning(f'预测天数超出范围: {prediction_days}')
                    return jsonify({'success': False, 'message': '预测天数必须在1-30之间'}), 400
            except ValueError:
                app.logger.warning('预测天数必须是整数')
                return jsonify({'success': False, 'message': '预测天数必须是整数'}), 400
        else:
            prediction_days = None
        
        app.logger.info(f'开始预测单只股票: 股票代码={stock_code}, 模型类型={model_type}, 预测天数={prediction_days}')
        
        # 进行预测
        try:
            # 构建预期的模型文件路径进行预检查
            model_file = None
            if model_path:
                # 如果指定了模型目录，构建完整路径
                model_file = os.path.join(model_path, f"{stock_code}_{model_type}_best.pth")
            else:
                # 使用默认模型目录
                default_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', f"{stock_code}_{model_type}_best.pth")
                model_file = default_model_path
                
            # 检查模型文件是否存在
            if not os.path.exists(model_file):
                app.logger.warning(f'模型文件不存在: {model_file}')
                return jsonify({'success': False, 'message': f'预测失败：模型文件不存在，请确保已训练模型，文件路径：{model_file}'}), 500
            
            # 修复参数顺序：stock_code, model_type, sequence_length, prediction_days, model_dir
            # model_path实际上是model_dir参数
            success, result = main_predict(stock_code, model_type, DEFAULT_SEQUENCE_LENGTH, prediction_days, model_path)
        except Exception as e:
            app.logger.error(f'预测执行过程出错: {e}', exc_info=True)
            error_str = str(e)
            # 根据错误类型给出更具体的提示
            if 'model' in error_str.lower() or 'load' in error_str.lower():
                return jsonify({'success': False, 'message': f'模型加载失败: {error_str}，请检查模型文件是否正确'}), 500
            elif 'data' in error_str.lower():
                return jsonify({'success': False, 'message': f'数据获取失败: {error_str}，请检查股票代码或网络连接'}), 500
            elif 'connection' in error_str.lower() or 'timeout' in error_str.lower():
                return jsonify({'success': False, 'message': f'网络连接问题: {error_str}，请检查网络设置'}), 500
            else:
                return jsonify({'success': False, 'message': f'预测执行失败: {error_str}'}), 500
        
        if success:
            if model_type == 'kline_lstm':
                # K线模型预测结果处理
                try:
                    app.logger.info(f'K线预测成功: {stock_code}, 预测未来{prediction_days}天')
                    
                    # 格式化预测数据以便前端展示
                    formatted_predictions = []
                    for pred in result.get('predictions', []):
                        formatted_predictions.append({
                            'open': round(float(pred[0]), 2),
                            'high': round(float(pred[1]), 2),
                            'low': round(float(pred[2]), 2),
                            'close': round(float(pred[3]), 2),
                            'volume': round(float(pred[4]), 2)
                        })
                    
                    # 返回K线预测结果和图表路径
                    return jsonify({
                        'success': True,
                        'stock_code': stock_code,
                        'model_type': 'kline_lstm',
                        'predictions': formatted_predictions,
                        'latest_kline': {
                            'close': result['metadata']['latest_close'] if 'metadata' in result and 'latest_close' in result['metadata'] else None
                        },
                        'prediction_days': prediction_days,
                        'message': f'K线预测成功，预测未来{prediction_days}天'
                    })
                except Exception as e:
                    app.logger.error(f'处理K线预测结果时出错: {e}')
                    return jsonify({'success': False, 'message': f'处理预测结果失败: {str(e)}'}), 500
            else:
                # 基线模型预测结果处理
                try:
                    app.logger.info(f'股票预测成功: {stock_code}, 预测价格: {result['predicted_price']}, 涨跌幅: {result['price_change_percent']:.2f}%')
                    return jsonify({
                        'success': True,
                        'stock_code': stock_code,
                        'model_type': 'baseline',
                        'latest_close': result.get('latest_close'),
                        'predicted_price': result.get('predicted_price'),
                        'price_change_percent': result.get('price_change_percent'),
                        'chart_path': result.get('plot_path'),
                        'message': '预测成功'
                    })
                except Exception as e:
                    app.logger.error(f'处理基线模型预测结果时出错: {e}')
                    return jsonify({'success': False, 'message': f'处理预测结果失败: {str(e)}'}), 500
        else:
            # 预测失败处理
            app.logger.warning(f'股票预测失败: {stock_code}, 结果: {result}')
            
            # 检查是否是模型文件不存在的问题
            model_file = f"{stock_code}_{model_type}_best.pth"
            default_model_path = os.path.join(MODELS_DIR, model_file)
            
            if not os.path.exists(default_model_path) and model_path is None:
                # 模型文件不存在的情况
                error_msg = f'预测失败：未找到训练好的模型文件 "{model_file}"。请先使用相应的股票代码和模型类型训练模型。'
                error_msg += f'\n模型应该保存在: {default_model_path}'
            elif isinstance(result, dict) and 'error' in result:
                # 处理返回的错误字典
                error_msg = result['error']
                # 根据错误类型添加更具体的提示
                if 'Token' in error_msg or 'token' in error_msg:
                    error_msg += '，请检查config.py中的Tushare API Token配置'
                elif '权限' in error_msg:
                    error_msg += '，请升级Tushare账户权限以获取更多数据访问权限'
                elif '数据' in error_msg:
                    error_msg += '，请检查股票代码是否正确且有足够的历史数据'
                elif '连接' in error_msg or 'timeout' in error_msg:
                    error_msg += '，请检查网络连接状态和防火墙设置'
            else:
                # 综合检查提示
                error_msg = '预测失败，请按以下步骤排查：\n'
                error_msg += '1. 确认模型已训练完成并保存至正确路径\n'
                error_msg += f'   - 期望模型文件：{model_file}\n'
                error_msg += f'   - 期望路径：{default_model_path}\n'
                error_msg += '2. 验证股票代码格式是否使用.SZ或.SH后缀\n'
                error_msg += '3. 检查Tushare API Token在config.py中的配置\n'
                error_msg += '4. 确认网络连接正常，无防火墙拦截\n'
                error_msg += '5. 验证股票是否有足够的历史交易数据'
            
            return jsonify({'success': False, 'message': error_msg}), 400
    
    except Exception as e:
        app.logger.error(f'预测单只股票API出错: {e}', exc_info=True)
        return jsonify({'success': False, 'message': f'服务器内部错误: {str(e)}'}), 500

# 批量预测路由
@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    """批量预测股票价格 - 增强版，添加完整的参数验证和错误处理"""
    try:
        # 获取并验证请求数据
        data = request.json
        if not data:
            app.logger.warning('请求数据为空')
            return jsonify({'success': False, 'message': '请求数据不能为空'}), 400
        
        # 验证股票代码列表
        stock_codes = data.get('stock_codes', [])
        if not isinstance(stock_codes, list):
            app.logger.warning('股票代码必须是列表格式')
            return jsonify({'success': False, 'message': '股票代码必须是列表格式'}), 400
        
        if not stock_codes:
            app.logger.warning('股票代码列表为空')
            return jsonify({'success': False, 'message': '请至少选择一只股票'}), 400
        
        # 限制批量预测数量
        if len(stock_codes) > MAX_BATCH_PREDICTION:
            app.logger.warning(f'批量预测数量超过限制: {len(stock_codes)} > {MAX_BATCH_PREDICTION}')
            return jsonify({'success': False, 'message': f'批量预测数量已限制为{MAX_BATCH_PREDICTION}只股票'}), 400
        
        # 验证模型类型
        model_type = data.get('model_type', 'baseline')
        valid_model_types = ['lstm', 'baseline', 'gru', 'kline_lstm']
        if model_type not in valid_model_types:
            app.logger.warning(f'无效的模型类型: {model_type}')
            return jsonify({'success': False, 'message': f'无效的模型类型，支持的类型: {valid_model_types}'}), 400
        
        # 验证所有股票代码格式
        invalid_codes = []
        for code in stock_codes:
            if not isinstance(code, str) or not (code.endswith('.SZ') or code.endswith('.SH')):
                invalid_codes.append(code)
        
        if invalid_codes:
            app.logger.warning(f'发现无效的股票代码: {invalid_codes}')
            return jsonify({'success': False, 'message': f'无效的股票代码格式: {invalid_codes}，请使用.SZ或.SH后缀'}), 400
        
        app.logger.info(f'开始批量预测: {len(stock_codes)}只股票, 模型类型: {model_type}')
        
        # 使用第一个股票代码的模型进行批量预测
        reference_stock = stock_codes[0]
        
        # 获取并检查模型是否存在
        try:
            model_path = get_default_model_path(reference_stock, model_type)
            if not model_path or not os.path.exists(model_path):
                app.logger.warning(f'未找到股票 {reference_stock} 的模型: {model_path}')
                return jsonify({'success': False, 'message': f'未找到股票 {reference_stock} 的模型，请先训练模型'}), 404
        except Exception as e:
            app.logger.error(f'查找模型文件时出错: {str(e)}')
            return jsonify({'success': False, 'message': f'查找模型文件时出错: {str(e)}'}), 500
        
        # 获取设备
        try:
            device = get_device()
            app.logger.info(f'使用设备: {device}')
        except Exception as e:
            app.logger.error(f'获取设备信息失败: {str(e)}')
            return jsonify({'success': False, 'message': f'获取设备信息失败: {str(e)}'}), 500
        
        # 进行批量预测
        try:
            app.logger.info(f'开始执行批量预测: 模型路径={model_path}, 股票数量={len(stock_codes)}')
            results = batch_predict_stocks(model_path, stock_codes, device, model_type)
            
            # 分析预测结果
            if results and isinstance(results, list):
                success_count = sum(1 for item in results if isinstance(item, dict) and 'error' not in item)
                fail_count = len(results) - success_count
                
                app.logger.info(f'批量预测完成: 总共{len(results)}只, 成功{success_count}只, 失败{fail_count}只')
                
                # 保存结果到CSV
                try:
                    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_DIR)
                    os.makedirs(result_dir, exist_ok=True)
                    
                    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                    result_file = os.path.join(result_dir, f'batch_prediction_{current_time}.csv')
                    
                    # 过滤出有效的预测结果进行保存
                    valid_results = [item for item in results if isinstance(item, dict) and 'error' not in item]
                    if valid_results:
                        df = pd.DataFrame(valid_results)
                        df.to_csv(result_file, index=False, encoding='utf-8-sig')
                        app.logger.info(f'预测结果已保存到: {result_file}')
                    else:
                        app.logger.warning('没有有效的预测结果可以保存到CSV文件')
                except Exception as e:
                    app.logger.error(f'保存预测结果到CSV时出错: {e}')
                    # 即使保存失败，仍然返回预测结果
                    result_file = None
                
                # 对成功的结果按涨跌幅排序
                sorted_results = []
                failed_results = []
                
                for item in results:
                    if isinstance(item, dict):
                        if 'error' in item:
                            failed_results.append(item)
                        else:
                            sorted_results.append(item)
                
                # 按涨跌幅降序排序
                sorted_results.sort(key=lambda x: x.get('price_change_percent', 0), reverse=True)
                
                # 合并排序后的成功结果和失败结果
                final_results = sorted_results + failed_results
                
                return jsonify({
                    'success': True,
                    'results': final_results,
                    'summary': {
                        'total': len(results),
                        'success': success_count,
                        'failed': fail_count
                    },
                    'file_path': result_file,
                    'message': f'批量预测完成，成功{success_count}只，失败{fail_count}只'
                })
            else:
                app.logger.warning('批量预测返回空结果或格式错误')
                return jsonify({'success': False, 'message': '批量预测失败，未返回有效结果'}), 500
                
        except Exception as e:
            app.logger.error(f'批量预测执行过程出错: {e}', exc_info=True)
            return jsonify({'success': False, 'message': f'批量预测执行失败: {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f'批量预测路由出错: {e}', exc_info=True)
        return jsonify({'success': False, 'message': f'服务器内部错误: {str(e)}'}), 500

# 图像绘画功能API路由
@app.route('/api/painting/list', methods=['GET'])
def list_paintings():
    """获取绘画作品列表"""
    try:
        app.logger.info('开始获取绘画作品列表')
        
        # 获取绘画目录路径
        painting_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'paintings')
        
        if not os.path.exists(painting_dir):
            return jsonify({
                'success': True,
                'paintings': [],
                'message': '暂无绘画作品'
            })
        
        # 获取所有画布配置文件
        config_files = [f for f in os.listdir(painting_dir) if f.endswith('_config.json')]
        
        paintings = []
        for config_file in config_files:
            try:
                config_path = os.path.join(painting_dir, config_file)
                
                # 读取画布配置
                with open(config_path, 'r', encoding='utf-8') as f:
                    import json
                    canvas_config = json.load(f)
                
                # 获取文件统计信息
                file_stats = os.stat(config_path)
                last_modified = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                painting_info = {
                    'canvas_id': canvas_config.get('id'),
                    'width': canvas_config.get('width'),
                    'height': canvas_config.get('height'),
                    'background_color': canvas_config.get('background_color'),
                    'created_at': canvas_config.get('created_at'),
                    'last_modified': last_modified,
                    'stroke_count': len(canvas_config.get('strokes', [])),
                    'shape_count': len(canvas_config.get('shapes', [])),
                    'config_file': config_file
                }
                
                paintings.append(painting_info)
                
            except Exception as e:
                app.logger.warning(f'读取画布配置文件失败: {config_file}, 错误: {str(e)}')
                continue
        
        # 按创建时间倒序排列
        paintings.sort(key=lambda x: x['created_at'], reverse=True)
        
        app.logger.info(f'获取绘画作品列表成功，共 {len(paintings)} 个作品')
        
        return jsonify({
            'success': True,
            'paintings': paintings,
            'message': f'获取绘画作品列表成功，共 {len(paintings)} 个作品'
        })
        
    except Exception as e:
        app.logger.error(f'获取绘画作品列表失败: {e}')
        return jsonify({'success': False, 'message': f'获取作品列表失败: {str(e)}'}), 500

@app.route('/api/painting/create_canvas', methods=['POST'])
def create_painting_canvas():
    """创建绘画画布"""
    try:
        data = request.json
        if not data:
            app.logger.warning('绘画请求数据为空')
            return jsonify({'success': False, 'message': '请求数据不能为空'}), 400
        
        # 获取画布参数
        canvas_width = data.get('width', 800)
        canvas_height = data.get('height', 600)
        background_color = data.get('background_color', '#ffffff')
        
        # 参数验证
        if not isinstance(canvas_width, int) or not isinstance(canvas_height, int):
            return jsonify({'success': False, 'message': '画布尺寸必须是整数'}), 400
        
        if canvas_width < 100 or canvas_height < 100 or canvas_width > 2000 or canvas_height > 2000:
            return jsonify({'success': False, 'message': '画布尺寸必须在100-2000像素之间'}), 400
        
        # 生成唯一画布ID
        canvas_id = f"canvas_{int(time.time())}_{os.getpid()}"
        
        # 创建绘画目录
        painting_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'paintings')
        os.makedirs(painting_dir, exist_ok=True)
        
        # 生成画布配置
        canvas_config = {
            'id': canvas_id,
            'width': canvas_width,
            'height': canvas_height,
            'background_color': background_color,
            'created_at': datetime.now().isoformat(),
            'strokes': [],
            'shapes': []
        }
        
        # 保存画布配置
        config_path = os.path.join(painting_dir, f"{canvas_id}_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(canvas_config, f, ensure_ascii=False, indent=2)
        
        app.logger.info(f'创建绘画画布成功: {canvas_id}')
        return jsonify({
            'success': True,
            'canvas_id': canvas_id,
            'config': canvas_config,
            'message': '画布创建成功'
        })
        
    except Exception as e:
        app.logger.error(f'创建绘画画布失败: {e}')
        return jsonify({'success': False, 'message': f'创建画布失败: {str(e)}'}), 500

@app.route('/api/painting/save_stroke', methods=['POST'])
def save_painting_stroke():
    """保存绘画笔触"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'message': '请求数据不能为空'}), 400
        
        canvas_id = data.get('canvas_id')
        if not canvas_id:
            return jsonify({'success': False, 'message': '画布ID不能为空'}), 400
        
        # 获取笔触数据
        stroke_data = {
            'id': f"stroke_{int(time.time())}_{len(data.get('points', []))}",
            'points': data.get('points', []),
            'color': data.get('color', '#000000'),
            'width': data.get('width', 2),
            'opacity': data.get('opacity', 1.0),
            'tool': data.get('tool', 'brush')
        }
        
        # 更新画布配置
        painting_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'paintings')
        config_path = os.path.join(painting_dir, f"{canvas_id}_config.json")
        
        if not os.path.exists(config_path):
            return jsonify({'success': False, 'message': '画布不存在'}), 404
        
        # 读取现有配置
        with open(config_path, 'r', encoding='utf-8') as f:
            import json
            canvas_config = json.load(f)
        
        # 添加笔触
        canvas_config['strokes'].append(stroke_data)
        canvas_config['updated_at'] = datetime.now().isoformat()
        
        # 保存更新后的配置
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(canvas_config, f, ensure_ascii=False, indent=2)
        
        app.logger.info(f'保存绘画笔触成功: {canvas_id}')
        return jsonify({
            'success': True,
            'stroke_id': stroke_data['id'],
            'message': '笔触保存成功'
        })
        
    except Exception as e:
        app.logger.error(f'保存绘画笔触失败: {e}')
        return jsonify({'success': False, 'message': f'保存笔触失败: {str(e)}'}), 500

@app.route('/api/painting/export_canvas', methods=['POST'])
def export_painting_canvas():
    """导出绘画作品为图片"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'message': '请求数据不能为空'}), 400
        
        canvas_id = data.get('canvas_id')
        export_format = data.get('format', 'png')
        
        if not canvas_id:
            return jsonify({'success': False, 'message': '画布ID不能为空'}), 400
        
        if export_format not in ['png', 'jpg', 'svg']:
            return jsonify({'success': False, 'message': '不支持的导出格式'}), 400
        
        # 读取画布配置
        painting_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'paintings')
        config_path = os.path.join(painting_dir, f"{canvas_id}_config.json")
        
        if not os.path.exists(config_path):
            return jsonify({'success': False, 'message': '画布不存在'}), 404
        
        # 生成导出文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_filename = f"{canvas_id}_{timestamp}.{export_format}"
        export_path = os.path.join(painting_dir, export_filename)
        
        # 这里可以集成实际的图像生成逻辑
        # 目前返回模拟的成功响应
        app.logger.info(f'导出绘画作品成功: {export_filename}')
        
        return jsonify({
            'success': True,
            'file_name': export_filename,
            'file_url': f'/static/paintings/{export_filename}',
            'message': f'绘画作品导出成功'
        })
        
    except Exception as e:
        app.logger.error(f'导出绘画作品失败: {e}')
        return jsonify({'success': False, 'message': f'导出失败: {str(e)}'}), 500

@app.route('/api/painting/save', methods=['POST'])
def save_painting():
    """保存绘画作品"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'message': '请求数据不能为空'}), 400
        
        canvas_data = data.get('canvas_data')
        if not canvas_data:
            return jsonify({'success': False, 'message': '画布数据不能为空'}), 400
        
        # 生成画作ID
        painting_id = f"painting_{int(time.time())}_{os.getpid()}"
        
        # 保存画作数据
        painting_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'paintings')
        os.makedirs(painting_dir, exist_ok=True)
        
        # 保存base64图像数据
        if canvas_data.startswith('data:image/'):
            # 提取base64数据
            header, data_part = canvas_data.split(',', 1)
            image_data = base64.b64decode(data_part)
            
            # 保存为PNG文件
            image_filename = f"{painting_id}.png"
            image_path = os.path.join(painting_dir, image_filename)
            
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            # 创建元数据
            metadata = {
                'id': painting_id,
                'filename': image_filename,
                'created_at': datetime.now().isoformat(),
                'thumbnail': canvas_data
            }
            
            metadata_path = os.path.join(painting_dir, f"{painting_id}_meta.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            app.logger.info(f'保存绘画作品成功: {painting_id}')
            
            return jsonify({
                'success': True,
                'painting_id': painting_id,
                'message': '绘画作品保存成功'
            })
        else:
            return jsonify({'success': False, 'message': '无效的图像数据格式'}), 400
            
    except Exception as e:
        app.logger.error(f'保存绘画作品失败: {e}')
        return jsonify({'success': False, 'message': f'保存失败: {str(e)}'}), 500

@app.route('/api/painting/export', methods=['POST'])
def export_painting():
    """导出绘画作品（下载）"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'message': '请求数据不能为空'}), 400
        
        canvas_data = data.get('canvas_data')
        filename = data.get('filename', f'painting_{int(time.time())}.png')
        
        if not canvas_data:
            return jsonify({'success': False, 'message': '画布数据不能为空'}), 400
        
        # 验证文件名安全性
        if '..' in filename or '\\' in filename or '/' in filename:
            return jsonify({'success': False, 'message': '文件名包含非法字符'}), 400
        
        # 确保文件扩展名
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            filename += '.png'
        
        # 生成下载链接（这里可以返回base64数据让前端下载）
        download_data = {
            'success': True,
            'filename': filename,
            'data': canvas_data,
            'message': '导出成功'
        }
        
        return jsonify(download_data)
        
    except Exception as e:
        app.logger.error(f'导出绘画作品失败: {e}')
        return jsonify({'success': False, 'message': f'导出失败: {str(e)}'}), 500

@app.route('/api/painting/load/<painting_id>', methods=['GET'])
def load_painting(painting_id):
    """加载绘画作品"""
    try:
        painting_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'paintings')
        
        # 检查元数据文件
        metadata_path = os.path.join(painting_dir, f"{painting_id}_meta.json")
        if not os.path.exists(metadata_path):
            return jsonify({'success': False, 'message': '画作不存在'}), 404
        
        # 读取元数据
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return jsonify({
            'success': True,
            'canvas_data': metadata.get('thumbnail'),
            'filename': metadata.get('filename'),
            'created_at': metadata.get('created_at'),
            'message': '画作加载成功'
        })
        
    except Exception as e:
        app.logger.error(f'加载绘画作品失败: {e}')
        return jsonify({'success': False, 'message': f'加载失败: {str(e)}'}), 500

@app.route('/api/painting/delete/<painting_id>', methods=['DELETE'])
def delete_painting(painting_id):
    """删除绘画作品"""
    try:
        painting_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'paintings')
        
        # 删除图像文件
        image_filename = f"{painting_id}.png"
        image_path = os.path.join(painting_dir, image_filename)
        
        # 删除元数据文件
        metadata_path = os.path.join(painting_dir, f"{painting_id}_meta.json")
        
        deleted_files = []
        
        # 删除图像文件
        if os.path.exists(image_path):
            os.remove(image_path)
            deleted_files.append(image_filename)
        
        # 删除元数据文件
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            deleted_files.append(f"{painting_id}_meta.json")
        
        if not deleted_files:
            return jsonify({'success': False, 'message': '画作不存在'}), 404
        
        app.logger.info(f'删除绘画作品成功: {painting_id}')
        
        return jsonify({
            'success': True,
            'message': '画作删除成功',
            'deleted_files': deleted_files
        })
        
    except Exception as e:
        app.logger.error(f'删除绘画作品失败: {e}')
        return jsonify({'success': False, 'message': f'删除失败: {str(e)}'}), 500

# 专业金融数据可视化路由
@app.route('/api/financial_visualization', methods=['POST'])
def financial_visualization():
    """专业金融数据可视化API - 生成K线图、技术指标等专业图表"""
    try:
        # 获取并验证请求数据
        data = request.json
        if not data:
            app.logger.warning('金融可视化请求数据为空')
            return jsonify({'success': False, 'message': '请求数据不能为空'}), 400
        
        # 验证必要参数
        stock_code = data.get('stock_code')
        if not stock_code:
            app.logger.warning('金融可视化：股票代码为空')
            return jsonify({'success': False, 'message': '股票代码不能为空'}), 400
        
        # 验证股票代码格式
        if not (stock_code.endswith('.SZ') or stock_code.endswith('.SH')):
            app.logger.warning(f'金融可视化：股票代码格式错误: {stock_code}')
            return jsonify({'success': False, 'message': '股票代码格式错误，请使用.SZ或.SH后缀'}), 400
        
        # 获取可选参数
        start_date = data.get('start_date', DEFAULT_START_DATE)
        end_date = data.get('end_date', DEFAULT_END_DATE)
        chart_types = data.get('chart_types', ['kline', 'technical', 'volume'])  # 默认生成所有图表
        
        app.logger.info(f'开始生成专业金融可视化图表: {stock_code}, 图表类型: {chart_types}')
        
        # 获取股票数据
        try:
            df = fetch_stock_data(stock_code, start_date, end_date)
            if df is None or df.empty:
                app.logger.warning(f'无法获取股票数据: {stock_code}')
                return jsonify({'success': False, 'message': '无法获取股票数据，请检查股票代码和日期范围'}), 500
        except Exception as e:
            app.logger.error(f'获取股票数据失败: {e}')
            return jsonify({'success': False, 'message': f'获取股票数据失败: {str(e)}'}), 500
        
        # 获取股票名称
        try:
            stock_name = get_stock_name(stock_code)
        except Exception as e:
            app.logger.warning(f'获取股票名称失败: {e}')
            stock_name = ""
        
        # 创建金融可视化器
        try:
            viz = create_financial_visualizer()
        except Exception as e:
            app.logger.error(f'创建金融可视化器失败: {e}')
            return jsonify({'success': False, 'message': f'创建可视化工具失败: {str(e)}'}), 500
        
        # 生成专业金融图表
        try:
            charts = viz.generate_professional_report(df, stock_code, stock_name)
            
            if not charts:
                app.logger.error('生成金融图表失败')
                return jsonify({'success': False, 'message': '生成金融图表失败'}), 500
            
            # 根据请求过滤图表类型
            filtered_charts = {}
            for chart_type in chart_types:
                if chart_type in charts:
                    filtered_charts[chart_type] = charts[chart_type]
            
            # 获取摘要信息
            summary = charts.get('summary', {})
            
            app.logger.info(f'金融可视化图表生成成功: {stock_code}, 图表数量: {len(filtered_charts)}')
            
            return jsonify({
                'success': True,
                'charts': filtered_charts,
                'summary': summary,
                'message': '专业金融图表生成成功'
            })
            
        except Exception as e:
            app.logger.error(f'生成金融图表时出错: {e}')
            return jsonify({'success': False, 'message': f'生成金融图表失败: {str(e)}'}), 500
    
    except Exception as e:
        app.logger.error(f'金融可视化API出错: {e}', exc_info=True)
        return jsonify({'success': False, 'message': f'服务器内部错误: {str(e)}'}), 500

# 获取已训练模型列表路由
@app.route('/api/get_models', methods=['GET'])
def get_models():
    """获取已训练模型列表 - 增强版，添加完整的错误处理和日志记录"""
    try:
        app.logger.info('开始获取模型列表')
        
        # 获取模型目录路径
        try:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODELS_DIR)
            app.logger.debug(f'模型目录路径: {model_dir}')
        except Exception as e:
            app.logger.error(f'构建模型目录路径时出错: {e}')
            return jsonify({'success': False, 'message': f'构建模型目录路径失败: {str(e)}'}), 500
        
        # 检查并创建模型目录
        if not os.path.exists(model_dir):
            try:
                os.makedirs(model_dir, exist_ok=True)
                app.logger.info(f'模型目录不存在，已创建: {model_dir}')
                return jsonify({'success': True, 'models': [], 'message': '模型目录已创建，但暂无模型文件'})
            except Exception as e:
                app.logger.error(f'创建模型目录时出错: {e}')
                return jsonify({'success': False, 'message': f'创建模型目录失败: {str(e)}'}), 500
        
        # 获取所有模型文件
        try:
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            app.logger.info(f'找到 {len(model_files)} 个模型文件')
        except Exception as e:
            app.logger.error(f'读取模型目录时出错: {e}')
            return jsonify({'success': False, 'message': f'读取模型目录失败: {str(e)}'}), 500
        
        # 解析模型信息
        models = []
        success_count = 0
        fail_count = 0
        
        for model_file in model_files:
            try:
                model_path = os.path.join(model_dir, model_file)
                
                # 加载模型检查点
                try:
                    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                except Exception as load_error:
                    app.logger.warning(f'无法加载模型文件 {model_file}: {str(load_error)}')
                    fail_count += 1
                    continue
                
                # 获取文件信息
                try:
                    file_stats = os.stat(model_path)
                    last_modified = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    file_size = f"{file_stats.st_size / 1024:.2f} KB"
                except Exception as stat_error:
                    app.logger.warning(f'获取文件信息失败 {model_file}: {str(stat_error)}')
                    last_modified = '未知'
                    file_size = '未知'
                
                # 构建模型信息
                model_info = {
                    'filename': model_file,
                    'stock_code': str(checkpoint.get('stock_code', 'unknown')),
                    'model_type': str(checkpoint.get('model_type', 'unknown')),
                    'hidden_size': int(checkpoint.get('hidden_size', 64)),
                    'num_layers': int(checkpoint.get('num_layers', 2)),
                    'sequence_length': int(checkpoint.get('sequence_length', 5)),
                    'file_path': model_path,
                    'last_modified': last_modified,
                    'file_size': file_size
                }
                
                models.append(model_info)
                success_count += 1
                
            except Exception as e:
                app.logger.warning(f'处理模型文件 {model_file} 时出错: {str(e)}')
                fail_count += 1
                # 忽略无法处理的模型文件，继续处理下一个
                continue
        
        # 按修改时间倒序排列
        models.sort(key=lambda x: x['last_modified'], reverse=True)
        
        app.logger.info(f'获取模型列表完成: 成功解析 {success_count} 个模型，跳过 {fail_count} 个模型')
        
        return jsonify({
            'success': True,
            'models': models,
            'summary': {
                'total_files': len(model_files),
                'successfully_parsed': success_count,
                'skipped': fail_count
            },
            'message': f'获取模型列表成功，共 {len(models)} 个有效模型'
        })
        
    except Exception as e:
        app.logger.error(f'获取模型列表路由出错: {e}', exc_info=True)
        return jsonify({'success': False, 'message': f'服务器内部错误: {str(e)}'}), 500

# 删除模型路由
@app.route('/api/delete_model', methods=['POST'])
def delete_model():
    """删除模型文件 - 增强版，添加完整的参数验证和错误处理"""
    try:
        app.logger.info('开始处理模型删除请求')
        
        # 获取并验证请求数据
        data = request.json
        if not data:
            app.logger.warning('删除模型请求数据为空')
            return jsonify({'success': False, 'message': '请求数据不能为空'}), 400
        
        # 验证模型文件名参数
        model_filename = data.get('filename')
        if not model_filename:
            app.logger.warning('模型文件名参数为空')
            return jsonify({'success': False, 'message': '请提供模型文件名'}), 400
        
        # 验证文件名格式安全性
        if not isinstance(model_filename, str):
            app.logger.warning(f'模型文件名参数类型错误: {type(model_filename)}')
            return jsonify({'success': False, 'message': '模型文件名必须是字符串'}), 400
        
        # 安全检查：防止路径遍历攻击
        if '..' in model_filename or '\\' in model_filename or '/ ' in model_filename:
            app.logger.warning(f'检测到潜在的路径遍历攻击: {model_filename}')
            return jsonify({'success': False, 'message': '模型文件名包含非法字符'}), 400
        
        # 验证文件扩展名
        if not model_filename.endswith('.pth'):
            app.logger.warning(f'无效的模型文件扩展名: {model_filename}')
            return jsonify({'success': False, 'message': '只能删除.pth格式的模型文件'}), 400
        
        app.logger.info(f'准备删除模型文件: {model_filename}')
        
        # 构建模型文件路径
        try:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODELS_DIR)
            model_path = os.path.join(model_dir, model_filename)
            app.logger.debug(f'模型文件完整路径: {model_path}')
        except Exception as e:
            app.logger.error(f'构建模型文件路径时出错: {e}')
            return jsonify({'success': False, 'message': f'构建模型文件路径失败: {str(e)}'}), 500
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            app.logger.warning(f'模型文件不存在: {model_path}')
            return jsonify({'success': False, 'message': '模型文件不存在'}), 404
        
        # 检查是否为文件而不是目录
        if not os.path.isfile(model_path):
            app.logger.warning(f'指定路径不是文件: {model_path}')
            return jsonify({'success': False, 'message': '指定路径不是有效的文件'}), 400
        
        # 删除模型文件
        try:
            os.remove(model_path)
            app.logger.info(f'模型文件已成功删除: {model_filename}')
            
            # 返回成功响应
            return jsonify({
                'success': True,
                'message': '模型删除成功',
                'deleted_file': model_filename
            })
        except PermissionError:
            app.logger.error(f'删除模型文件时权限不足: {model_path}')
            return jsonify({'success': False, 'message': '删除失败，权限不足'}), 403
        except Exception as e:
            app.logger.error(f'删除模型文件时出错: {e}', exc_info=True)
            return jsonify({'success': False, 'message': f'删除模型文件失败: {str(e)}'}), 500
            
    except Exception as e:
        app.logger.error(f'删除模型路由出错: {e}', exc_info=True)
        return jsonify({'success': False, 'message': f'服务器内部错误: {str(e)}'}), 500

# 生成图表路由
@app.route('/api/generate_chart', methods=['POST'])
def generate_chart():
    """生成图表路由 - 增强版，添加完整的参数验证、错误处理和日志记录"""
    try:
        app.logger.info('开始处理图表生成请求')
        
        # 获取并验证请求数据
        data = request.json
        if not data:
            app.logger.warning('图表生成请求数据为空')
            return jsonify({'success': False, 'message': '请求数据不能为空'}), 400
        
        # 验证图表类型参数
        chart_type = data.get('chart_type')
        if not chart_type:
            app.logger.warning('图表类型参数为空')
            return jsonify({'success': False, 'message': '请提供图表类型'}), 400
        
        # 验证图表类型有效性
        valid_chart_types = ['stock_price', 'prediction_result', 'kline_chart']
        if chart_type not in valid_chart_types:
            app.logger.warning(f'无效的图表类型: {chart_type}')
            return jsonify({'success': False, 'message': f'图表类型必须是以下之一: {', '.join(valid_chart_types)}'}), 400
        
        fig = None
        
        # 根据图表类型创建相应图表
        if chart_type == 'stock_price':
            app.logger.info('开始生成股票价格走势图')
            
            # 验证必要的数据字段
            dates = data.get('dates', [])
            close_prices = data.get('close_prices', [])
            
            if not isinstance(dates, list) or not isinstance(close_prices, list):
                app.logger.warning('股票价格图表数据类型错误')
                return jsonify({'success': False, 'message': 'dates和close_prices必须是列表类型'}), 400
            
            if len(dates) == 0 or len(close_prices) == 0:
                app.logger.warning('股票价格图表数据为空')
                return jsonify({'success': False, 'message': 'dates和close_prices不能为空列表'}), 400
            
            if len(dates) != len(close_prices):
                app.logger.warning(f'股票价格图表数据长度不匹配: dates={len(dates)}, close_prices={len(close_prices)}')
                return jsonify({'success': False, 'message': 'dates和close_prices长度必须一致'}), 400
            
            # 创建图表
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # 绘制收盘价线
                ax.plot(dates, close_prices, label='收盘价', color='#165DFF', linewidth=2)
                
                # 绘制均线（如果提供）
                ma5 = data.get('ma5', [])
                if ma5 and isinstance(ma5, list) and len(ma5) == len(dates):
                    ax.plot(dates, ma5, label='5日均线', color='#52C41A', linewidth=1.5, linestyle='--')
                
                ma10 = data.get('ma10', [])
                if ma10 and isinstance(ma10, list) and len(ma10) == len(dates):
                    ax.plot(dates, ma10, label='10日均线', color='#FAAD14', linewidth=1.5, linestyle='--')
                
                # 设置图表属性
                ax.set_title('股票价格走势图')
                ax.set_xlabel('日期')
                ax.set_ylabel('价格')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                
                # 自动调整x轴标签
                if len(dates) > 30:
                    step = max(1, len(dates) // 30)
                    plt.xticks(range(0, len(dates), step), dates[::step], rotation=45)
                else:
                    plt.xticks(rotation=45)
                
                plt.tight_layout()
                app.logger.info('股票价格走势图生成成功')
            except Exception as e:
                app.logger.error(f'生成股票价格图表时出错: {e}', exc_info=True)
                if fig:
                    plt.close(fig)
                return jsonify({'success': False, 'message': f'生成股票价格图表失败: {str(e)}'}), 500
                
        elif chart_type == 'prediction_result':
            app.logger.info('开始生成预测结果图表')
            
            # 验证必要的数据字段
            stock_code = data.get('stock_code')
            latest_close = data.get('latest_close')
            predicted_price = data.get('predicted_price')
            
            if not stock_code:
                app.logger.warning('预测结果图表缺少股票代码')
                return jsonify({'success': False, 'message': '请提供股票代码'}), 400
            
            if not isinstance(latest_close, (int, float)) or not isinstance(predicted_price, (int, float)):
                app.logger.warning('预测结果图表价格数据类型错误')
                return jsonify({'success': False, 'message': '最新收盘价和预测收盘价必须是数字类型'}), 400
            
            # 创建图表
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # 获取股票名称
                try:
                    stock_name = get_stock_name(stock_code)
                except Exception as e:
                    app.logger.warning(f'获取股票名称失败，使用代码代替: {e}')
                    stock_name = stock_code
                
                # 绘制柱状图
                categories = ['最新收盘价', '预测收盘价']
                values = [latest_close, predicted_price]
                colors = ['#94a3b8', '#52C41A' if predicted_price > latest_close else '#F5222D']
                
                bars = ax.bar(categories, values, color=colors)
                ax.set_title(f'股票{stock_name}({stock_code})价格预测结果')
                ax.set_ylabel('价格')
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
                            ha='center', va='bottom', fontsize=12)
                
                # 计算预测变化百分比
                try:
                    change_percent = ((predicted_price - latest_close) / latest_close) * 100
                    change_text = f'预测{'上涨' if change_percent > 0 else '下跌'}: {abs(change_percent):.2f}%'
                    ax.text(0.5, 0.95, change_text, ha='center', va='center', transform=ax.transAxes,
                            fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
                except ZeroDivisionError:
                    app.logger.warning('最新收盘价为0，无法计算变化百分比')
                
                # 添加网格线
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                app.logger.info('预测结果图表生成成功')
            except Exception as e:
                app.logger.error(f'生成预测结果图表时出错: {e}', exc_info=True)
                if fig:
                    plt.close(fig)
                return jsonify({'success': False, 'message': f'生成预测结果图表失败: {str(e)}'}), 500
                
        elif chart_type == 'kline_chart':
            app.logger.info('开始生成K线预测图')
            
            # 验证必要的数据字段
            stock_code = data.get('stock_code')
            history_data = data.get('history_data', {})
            
            if not stock_code:
                app.logger.warning('K线图缺少股票代码')
                return jsonify({'success': False, 'message': '请提供股票代码'}), 400
            
            if not history_data or not isinstance(history_data, (dict, list)):
                app.logger.warning('K线图历史数据无效')
                return jsonify({'success': False, 'message': '请提供有效的历史数据'}), 400
            
            # 创建图表
            try:
                import matplotlib.dates as mdates
                from datetime import datetime, timedelta
                
                # 获取股票名称
                try:
                    stock_name = get_stock_name(stock_code)
                except Exception as e:
                    app.logger.warning(f'获取股票名称失败，使用代码代替: {e}')
                    stock_name = stock_code
                
                # 处理历史数据
                try:
                    df = pd.DataFrame(history_data)
                    if df.empty:
                        raise ValueError('历史数据为空')
                    
                    # 验证必要的列
                    required_columns = ['trade_date', 'open', 'close', 'high', 'low', 'vol']
                    for col in required_columns:
                        if col not in df.columns:
                            raise ValueError(f'历史数据缺少必要列: {col}')
                    
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                except Exception as e:
                    app.logger.error(f'处理历史数据时出错: {e}')
                    return jsonify({'success': False, 'message': f'历史数据格式错误: {str(e)}'}), 400
                
                # 获取预测数据
                predictions = data.get('predictions', [])
                if predictions and not isinstance(predictions, list):
                    app.logger.warning('预测数据类型错误')
                    return jsonify({'success': False, 'message': '预测数据必须是列表类型'}), 400
                
                # 创建K线图
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
                fig.suptitle(f'股票{stock_name}({stock_code}) K线预测图', fontsize=16, fontweight='bold')
                
                # 绘制历史K线
                try:
                    for i, row in df.iterrows():
                        # 判断涨跌
                        if row['close'] >= row['open']:
                            color = 'red'  # 涨
                        else:
                            color = 'green'  # 跌
                        
                        # 绘制蜡烛线
                        ax1.plot([row['trade_date'], row['trade_date']], [row['low'], row['high']], color=color, linewidth=1.5)
                        ax1.plot([row['trade_date'], row['trade_date']], [row['open'], row['close']], color=color, linewidth=6)
                    
                    # 绘制移动平均线（如果有）
                    if 'ma5' in df.columns:
                        ax1.plot(df['trade_date'], df['ma5'], color='blue', label='MA5', linewidth=1.2)
                    if 'ma10' in df.columns:
                        ax1.plot(df['trade_date'], df['ma10'], color='orange', label='MA10', linewidth=1.2)
                except Exception as e:
                    app.logger.error(f'绘制历史K线时出错: {e}')
                    raise Exception(f'绘制K线失败: {str(e)}')
                
                # 绘制预测K线（如果有）
                if predictions:
                    app.logger.info(f'包含{len(predictions)}天预测数据')
                    try:
                        # 生成预测日期
                        last_date = df['trade_date'].iloc[-1]
                        pred_dates = [last_date + timedelta(days=i+1) for i in range(len(predictions))]
                        
                        # 验证预测数据格式
                        required_pred_keys = ['open', 'high', 'low', 'close', 'volume']
                        for i, pred in enumerate(predictions):
                            if not isinstance(pred, dict):
                                raise ValueError(f'预测数据第{i+1}条格式错误')
                            for key in required_pred_keys:
                                if key not in pred:
                                    raise ValueError(f'预测数据第{i+1}条缺少必要字段: {key}')
                        
                        # 绘制预测K线
                        for i, (pred, date) in enumerate(zip(predictions, pred_dates)):
                            open_price, high, low, close, volume = pred['open'], pred['high'], pred['low'], pred['close'], pred.get('volume', 0)
                            
                            # 判断涨跌
                            if close >= open_price:
                                color = 'red'  # 涨
                            else:
                                color = 'green'  # 跌
                            
                            # 绘制预测蜡烛线（使用虚线）
                            ax1.plot([date, date], [low, high], color=color, linewidth=1.5, linestyle='--', alpha=0.8)
                            ax1.plot([date, date], [open_price, close], color=color, linewidth=6, alpha=0.8)
                    except Exception as e:
                        app.logger.error(f'绘制预测K线时出错: {e}')
                        # 继续执行，不影响历史K线的显示
                
                # 设置主图属性
                ax1.set_ylabel('价格', fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.legend(loc='upper left')
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax1.tick_params(axis='x', rotation=45)
                
                # 绘制成交量
                try:
                    # 历史成交量
                    for i, row in df.iterrows():
                        if row['close'] >= row['open']:
                            color = 'red'
                        else:
                            color = 'green'
                        ax2.bar(row['trade_date'], row['vol'], color=color, alpha=0.7)
                    
                    # 预测成交量
                    if predictions:
                        for pred, date in zip(predictions, pred_dates):
                            volume = pred.get('volume', 0)
                            if pred['close'] >= pred['open']:
                                color = 'red'
                            else:
                                color = 'green'
                            ax2.bar(date, volume, color=color, alpha=0.7, linestyle='--', edgecolor='black', linewidth=0.5)
                except Exception as e:
                    app.logger.error(f'绘制成交量时出错: {e}')
                    # 继续执行，不影响K线的显示
                
                # 设置成交量图属性
                ax2.set_ylabel('成交量', fontsize=12)
                ax2.set_xlabel('日期', fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax2.tick_params(axis='x', rotation=45)
                
                # 调整布局
                plt.tight_layout(rect=[0, 0, 1, 0.97])
                app.logger.info('K线预测图生成成功')
            except Exception as e:
                app.logger.error(f'生成K线图时出错: {e}', exc_info=True)
                if fig:
                    plt.close(fig)
                return jsonify({'success': False, 'message': f'生成K线图失败: {str(e)}'}), 500
        
        # 将图表转换为base64
        try:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            app.logger.debug(f'图表Base64编码长度: {len(image_base64)}')
        except Exception as e:
            app.logger.error(f'图表编码为Base64时出错: {e}', exc_info=True)
            return jsonify({'success': False, 'message': f'图表转换失败: {str(e)}'}), 500
        finally:
            # 确保关闭图表，防止内存泄漏
            if fig:
                plt.close(fig)
        
        app.logger.info(f'图表生成成功: {chart_type}')
        return jsonify({
            'success': True,
            'image_data': image_base64,
            'chart_type': chart_type,
            'message': '图表生成成功'
        })
        
    except Exception as e:
        app.logger.error(f'生成图表路由出错: {e}', exc_info=True)
        # 确保在异常情况下关闭图表
        try:
            plt.close('all')
        except:
            pass
        return jsonify({'success': False, 'message': f'服务器内部错误: {str(e)}'}), 500

# 下载预测结果路由
@app.route('/api/download_result', methods=['GET'])
def download_result():
    """下载预测结果文件 - 增强版，添加完整的参数验证、错误处理和日志记录"""
    try:
        app.logger.info('开始处理文件下载请求')
        
        # 获取并验证文件路径参数
        file_path = request.args.get('file_path')
        if not file_path:
            app.logger.warning('文件下载请求缺少file_path参数')
            return jsonify({'success': False, 'message': '请提供文件路径参数'}), 400
        
        # 验证文件路径安全性（防止路径遍历攻击）
        if not isinstance(file_path, str):
            app.logger.warning(f'文件路径参数类型错误: {type(file_path)}')
            return jsonify({'success': False, 'message': '文件路径必须是字符串类型'}), 400
        
        # 规范化文件路径，防止路径遍历攻击
        try:
            # 获取应用根目录
            app_root = os.path.dirname(os.path.abspath(__file__))
            # 确保文件路径在results目录下
            results_dir = os.path.join(app_root, 'results')
            
            # 构建安全的文件路径
            if os.path.isabs(file_path):
                # 如果是绝对路径，检查是否在results目录下
                if not os.path.commonpath([file_path, results_dir]) == results_dir:
                    app.logger.warning(f'检测到潜在的路径遍历攻击: {file_path}')
                    return jsonify({'success': False, 'message': '非法的文件路径'}), 403
            else:
                # 如果是相对路径，构建安全的绝对路径
                file_path = os.path.join(results_dir, file_path)
            
            # 规范化路径
            file_path = os.path.normpath(file_path)
            app.logger.debug(f'安全构建的文件路径: {file_path}')
        except Exception as e:
            app.logger.error(f'构建安全文件路径时出错: {e}')
            return jsonify({'success': False, 'message': '文件路径解析失败'}), 500
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            app.logger.warning(f'请求下载的文件不存在: {file_path}')
            return jsonify({'success': False, 'message': '文件不存在'}), 404
        
        # 检查是否为文件而不是目录
        if not os.path.isfile(file_path):
            app.logger.warning(f'指定路径不是文件: {file_path}')
            return jsonify({'success': False, 'message': '指定路径不是有效的文件'}), 400
        
        # 检查文件扩展名（只允许CSV文件）
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext != '.csv':
            app.logger.warning(f'不支持的文件类型: {file_ext}')
            return jsonify({'success': False, 'message': '只支持下载CSV文件'}), 400
        
        # 检查文件大小限制（例如限制为100MB）
        max_size = 100 * 1024 * 1024  # 100MB
        try:
            file_size = os.path.getsize(file_path)
            if file_size > max_size:
                app.logger.warning(f'文件大小超过限制: {format_size(file_size)} > {format_size(max_size)}')
                return jsonify({'success': False, 'message': f'文件大小超过限制，请尝试下载更小的文件'}), 413
        except OSError as e:
            app.logger.error(f'获取文件大小时出错: {e}')
            return jsonify({'success': False, 'message': '获取文件信息失败'}), 500
        
        # 获取文件名并设置响应头
        filename = os.path.basename(file_path)
        app.logger.info(f'准备下载文件: {filename}, 大小: {format_size(file_size)}')
        
        try:
            # 创建响应
            response = make_response(send_file(file_path, as_attachment=True))
            
            # 设置正确的Content-Type
            response.headers['Content-Type'] = 'text/csv; charset=utf-8'
            
            # 设置Content-Disposition，支持中文文件名
            encoded_filename = urllib.parse.quote(filename)
            response.headers['Content-Disposition'] = f'attachment; filename="{filename}"; filename*=UTF-8''{encoded_filename}'
            
            # 设置Content-Length
            response.headers['Content-Length'] = str(file_size)
            
            app.logger.info(f'文件下载成功: {filename}')
            return response
        except PermissionError:
            app.logger.error(f'读取文件时权限不足: {file_path}')
            return jsonify({'success': False, 'message': '读取文件失败，权限不足'}), 403
        except FileNotFoundError:
            app.logger.error(f'文件在下载过程中被删除: {file_path}')
            return jsonify({'success': False, 'message': '文件不存在或已被删除'}), 404
        except Exception as e:
            app.logger.error(f'发送文件时出错: {e}', exc_info=True)
            return jsonify({'success': False, 'message': f'文件下载失败: {str(e)}'}), 500
            
    except Exception as e:
        app.logger.error(f'下载文件路由出错: {e}', exc_info=True)
        return jsonify({'success': False, 'message': f'服务器内部错误: {str(e)}'}), 500

# 获取K线实时预览
@app.route('/api/get_kline_preview', methods=['POST'])
def get_kline_preview():
    """获取股票K线实时预览 - 增强版，添加完整的参数验证、错误处理和日志记录"""
    try:
        app.logger.info('开始处理K线预览请求')
        
        # 获取并验证请求数据
        data = request.json
        if not data:
            app.logger.warning('K线预览请求数据为空')
            return jsonify({'success': False, 'message': '请求数据不能为空'}), 400
        
        # 验证股票代码参数
        stock_code = data.get('stock_code')
        if not stock_code:
            app.logger.warning('K线预览请求缺少股票代码')
            return jsonify({'success': False, 'message': '请提供股票代码'}), 400
        
        if not isinstance(stock_code, str):
            app.logger.warning(f'股票代码参数类型错误: {type(stock_code)}')
            return jsonify({'success': False, 'message': '股票代码必须是字符串类型'}), 400
        
        # 验证天数参数
        days = data.get('days', 30)
        try:
            days = int(days)
            # 限制天数范围
            if days <= 0:
                app.logger.warning(f'无效的天数参数: {days}')
                return jsonify({'success': False, 'message': '天数必须为正整数'}), 400
            if days > 365:
                app.logger.warning(f'天数参数过大: {days}')
                return jsonify({'success': False, 'message': '天数不能超过365天'}), 400
        except (ValueError, TypeError):
            app.logger.warning(f'天数参数格式错误: {days}')
            return jsonify({'success': False, 'message': '天数必须是有效的整数'}), 400
        
        app.logger.info(f'准备生成股票K线预览: {stock_code}, 天数: {days}')
        
        # 获取最新股票数据
        try:
            _, _, metadata = get_latest_stock_data(stock_code, days, model_type='kline_lstm')
            if not metadata or 'df' not in metadata:
                raise ValueError('获取股票数据失败，返回数据格式错误')
        except Exception as e:
            app.logger.error(f'获取股票数据时出错: {e}', exc_info=True)
            return jsonify({'success': False, 'message': f'获取股票数据失败: {str(e)}'}), 500
        
        # 处理数据
        try:
            df = metadata['df'].copy()
            if df.empty:
                app.logger.warning(f'股票{stock_code}没有可用数据')
                return jsonify({'success': False, 'message': '该股票暂无可用数据'}), 404
            
            # 验证必要的列
            required_columns = ['trade_date', 'open', 'close', 'high', 'low', 'vol']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f'股票数据缺少必要列: {col}')
            
            # 转换日期列
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            # 只取最近days天
            recent_df = df.tail(days).copy()
            app.logger.debug(f'获取到{len(recent_df)}天的股票数据')
        except Exception as e:
            app.logger.error(f'处理股票数据时出错: {e}', exc_info=True)
            return jsonify({'success': False, 'message': f'处理股票数据失败: {str(e)}'}), 500
        
        fig = None
        
        # 创建图表
        try:
            # 确保matplotlib.dates可用
            import matplotlib.dates as mdates
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # 获取股票名称
            try:
                stock_name = get_stock_name(stock_code)
            except Exception as e:
                app.logger.warning(f'获取股票名称失败，使用代码代替: {e}')
                stock_name = stock_code
            
            fig.suptitle(f'股票{stock_name}({stock_code}) 最近{days}天K线图', fontsize=14, fontweight='bold')
            
            # 绘制历史K线
            try:
                for i, row in recent_df.iterrows():
                    # 判断涨跌
                    if row['close'] >= row['open']:
                        color = 'red'  # 涨
                    else:
                        color = 'green'  # 跌
                    
                    # 绘制蜡烛线
                    ax1.plot([row['trade_date'], row['trade_date']], [row['low'], row['high']], color=color, linewidth=1.5)
                    ax1.plot([row['trade_date'], row['trade_date']], [row['open'], row['close']], color=color, linewidth=6)
                
                # 绘制移动平均线（如果有）
                if 'ma5' in recent_df.columns:
                    ax1.plot(recent_df['trade_date'], recent_df['ma5'], color='blue', label='MA5', linewidth=1.2)
                if 'ma10' in recent_df.columns:
                    ax1.plot(recent_df['trade_date'], recent_df['ma10'], color='orange', label='MA10', linewidth=1.2)
            except Exception as e:
                app.logger.error(f'绘制K线时出错: {e}')
                raise Exception(f'绘制K线失败: {str(e)}')
            
            # 设置主图属性
            ax1.set_ylabel('价格', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.tick_params(axis='x', rotation=45)
            
            # 绘制成交量
            try:
                for i, row in recent_df.iterrows():
                    if row['close'] >= row['open']:
                        color = 'red'
                    else:
                        color = 'green'
                    ax2.bar(row['trade_date'], row['vol'], color=color, alpha=0.7)
            except Exception as e:
                app.logger.error(f'绘制成交量时出错: {e}')
                # 继续执行，不影响K线的显示
            
            # 设置成交量图属性
            ax2.set_ylabel('成交量', fontsize=10)
            ax2.set_xlabel('日期', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.tick_params(axis='x', rotation=45)
            
            # 调整布局
            plt.tight_layout(rect=[0, 0, 1, 0.97])
        except Exception as e:
            app.logger.error(f'创建图表时出错: {e}', exc_info=True)
            if fig:
                plt.close(fig)
            return jsonify({'success': False, 'message': f'生成K线图失败: {str(e)}'}), 500
        
        # 将图表转换为base64编码
        try:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            app.logger.debug(f'K线图Base64编码长度: {len(image_base64)}')
        except Exception as e:
            app.logger.error(f'图表编码为Base64时出错: {e}', exc_info=True)
            return jsonify({'success': False, 'message': f'图表转换失败: {str(e)}'}), 500
        finally:
            # 确保关闭图表，防止内存泄漏
            if fig:
                plt.close(fig)
        
        app.logger.info(f'K线预览生成成功: {stock_code}')
        return jsonify({
            'success': True,
            'image_data': image_base64,
            'stock_code': stock_code,
            'days': days,
            'data_count': len(recent_df),
            'message': 'K线预览生成成功'
        })
        
    except Exception as e:
        app.logger.error(f'K线预览路由出错: {e}', exc_info=True)
        # 确保在异常情况下关闭图表
        try:
            plt.close('all')
        except:
            pass
        return jsonify({'success': False, 'message': f'服务器内部错误: {str(e)}'}), 500

# 获取系统状态路由
@app.route('/api/system_status', methods=['GET'])
def system_status():
    """获取系统状态 - 增强版，提供全面的系统信息、资源使用情况和服务状态监控"""
    try:
        app.logger.info('开始获取系统状态')
        
        # 初始化结果字典
        status_result = {
            'success': True,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'healthy'
        }
        
        # 1. 检查GPU和设备信息
        try:
            has_gpu = torch.cuda.is_available()
            device = 'cuda' if has_gpu else 'cpu'
            
            gpu_info = {
                'available': has_gpu,
                'count': torch.cuda.device_count() if has_gpu else 0,
                'current_device': torch.cuda.current_device() if has_gpu else None,
                'device_name': torch.cuda.get_device_name(0) if has_gpu else None
            }
            
            status_result['device'] = device
            status_result['gpu_info'] = gpu_info
            app.logger.debug(f'设备信息: {device}, GPU可用: {has_gpu}')
        except Exception as e:
            app.logger.error(f'获取设备信息失败: {e}', exc_info=True)
            status_result['device'] = 'unknown'
            status_result['gpu_info'] = {'error': str(e), 'available': False}
        
        # 2. 检查Tushare连接
        try:
            tushare_connected = False
            tushare_details = {
                'connected': False,
                'token_configured': bool(TUSHARE_TOKEN),
                'error': None
            }
            
            if TUSHARE_TOKEN:
                try:
                    from stock_data import pro
                    if pro:
                        test = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name', limit=5)
                        tushare_connected = len(test) > 0
                        tushare_details['connected'] = True
                        tushare_details['sample_count'] = len(test)
                except Exception as te:
                    tushare_details['error'] = str(te)
                    app.logger.warning(f'Tushare连接测试失败: {str(te)}')
            else:
                tushare_details['error'] = 'API Token未配置'
                app.logger.warning('Tushare API Token未配置')
            
            status_result['tushare_connected'] = tushare_connected
            status_result['tushare_details'] = tushare_details
        except Exception as e:
            app.logger.error(f'检查Tushare连接失败: {e}', exc_info=True)
            status_result['tushare_connected'] = False
            status_result['tushare_details'] = {'error': str(e)}
        
        # 3. 检查和创建必要的目录
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            directories = {
                'data': os.path.join(base_dir, 'data'),
                'models': os.path.join(base_dir, MODELS_DIR),
                'results': os.path.join(base_dir, RESULTS_DIR),
                'plots': os.path.join(base_dir, PLOTS_DIR),
                'logs': os.path.join(base_dir, 'logs')
            }
            
            # 创建缺失的目录
            for dir_name, dir_path in directories.items():
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    app.logger.debug(f'确认目录存在: {dir_path}')
                except Exception as e:
                    app.logger.error(f'创建目录失败 {dir_path}: {e}')
            
            # 获取目录统计信息
            dir_stats = {}
            for dir_name, dir_path in directories.items():
                if os.path.exists(dir_path):
                    try:
                        if dir_name == 'models':
                            file_count = len([f for f in os.listdir(dir_path) if f.endswith('.pth')])
                        elif dir_name == 'data':
                            file_count = len([f for f in os.listdir(dir_path) if f.endswith('.csv')])
                        else:
                            file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
                        
                        dir_stats[dir_name] = {
                            'size': get_directory_size(dir_path),
                            'file_count': file_count,
                            'path': dir_path,
                            'accessible': True
                        }
                    except Exception as e:
                        dir_stats[dir_name] = {
                            'error': str(e),
                            'accessible': False
                        }
                        app.logger.error(f'获取目录统计失败 {dir_name}: {e}')
                else:
                    dir_stats[dir_name] = {
                        'error': '目录不存在',
                        'accessible': False
                    }
            
            status_result['directories'] = dir_stats
            status_result['model_count'] = dir_stats['models']['file_count'] if dir_stats['models']['accessible'] else 0
            status_result['data_count'] = dir_stats['data']['file_count'] if dir_stats['data']['accessible'] else 0
        except Exception as e:
            app.logger.error(f'检查目录时出错: {e}', exc_info=True)
            status_result['directories'] = {'error': str(e)}
            status_result['model_count'] = 0
            status_result['data_count'] = 0
        
        # 4. 获取Python和系统版本信息
        try:
            python_version = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
            python_details = {
                'version': python_version,
                'version_info': {
                    'major': sys.version_info.major,
                    'minor': sys.version_info.minor,
                    'micro': sys.version_info.micro,
                    'releaselevel': sys.version_info.releaselevel,
                    'serial': sys.version_info.serial
                }
            }
            
            # 获取操作系统信息
            import platform
            os_info = {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'platform': platform.platform()
            }
            
            status_result['python_info'] = python_details
            status_result['os_info'] = os_info
            status_result['system_version'] = SYSTEM_VERSION
        except Exception as e:
            app.logger.error(f'获取版本信息失败: {e}', exc_info=True)
            status_result['python_info'] = {'error': str(e)}
            status_result['os_info'] = {'error': str(e)}
        
        # 5. 获取存储使用情况
        try:
            # 获取当前磁盘信息
            disk_usage = psutil.disk_usage('.')
            used = format_size(disk_usage.used)
            total = format_size(disk_usage.total)
            usage_percentage = int(disk_usage.percent)
            
            # 检查磁盘空间是否充足
            space_warning = usage_percentage >= 90
            if space_warning:
                app.logger.warning(f'磁盘空间不足: {usage_percentage}% 已使用')
            
            storage_info = {
                'used': used,
                'total': total,
                'free': format_size(disk_usage.free),
                'usage_percentage': usage_percentage,
                'warning': space_warning,
                'bytes_used': disk_usage.used,
                'bytes_total': disk_usage.total,
                'bytes_free': disk_usage.free
            }
            
            status_result['storage_info'] = storage_info
        except Exception as e:
            app.logger.error(f"获取存储信息失败: {e}", exc_info=True)
            status_result['storage_info'] = {'error': str(e)}
        
        # 6. 获取内存使用情况
        try:
            memory = psutil.virtual_memory()
            memory_info = {
                'total': format_size(memory.total),
                'available': format_size(memory.available),
                'used': format_size(memory.used),
                'percent': memory.percent,
                'bytes_total': memory.total,
                'bytes_available': memory.available,
                'bytes_used': memory.used
            }
            
            status_result['memory_info'] = memory_info
        except Exception as e:
            app.logger.error(f"获取内存信息失败: {e}", exc_info=True)
            status_result['memory_info'] = {'error': str(e)}
        
        # 7. 获取CPU使用情况
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count(logical=True)
            
            cpu_info = {
                'percent': cpu_percent,
                'count': cpu_count,
                'load_avg': None
            }
            
            # 在类Unix系统上获取负载平均值
            try:
                if hasattr(os, 'getloadavg'):
                    cpu_info['load_avg'] = os.getloadavg()
            except:
                pass
            
            status_result['cpu_info'] = cpu_info
        except Exception as e:
            app.logger.error(f"获取CPU信息失败: {e}", exc_info=True)
            status_result['cpu_info'] = {'error': str(e)}
        
        # 8. 检查关键功能和依赖
        try:
            # 检查必要的第三方库
            required_libraries = {
                'torch': torch.__version__ if 'torch' in sys.modules else None,
                'pandas': pd.__version__ if 'pandas' in sys.modules else None,
                'numpy': np.__version__ if 'np' in sys.modules else None,
                'matplotlib': plt.matplotlib.__version__ if 'plt' in sys.modules else None,
                'psutil': psutil.__version__ if 'psutil' in sys.modules else None
            }
            
            # 检查选股器初始化状态
            selector_status = {
                'initialized': stock_selector is not None,
                'type': type(stock_selector).__name__ if stock_selector is not None else None
            }
            
            # 功能可用性检测
            features = {
                'kline_prediction': True,
                'multi_feature_prediction': True,
                'technical_indicators': True,
                'smart_selection': True,
                'data_export': True,
                'visualization': True
            }
            
            status_result['required_libraries'] = required_libraries
            status_result['selector_status'] = selector_status
            status_result['features'] = features
        except Exception as e:
            app.logger.error(f"检查功能可用性失败: {e}", exc_info=True)
            status_result['features'] = {'error': str(e)}
        
        # 9. 检查应用运行时间
        try:
            if hasattr(app, 'start_time'):
                uptime = datetime.now() - app.start_time
                status_result['uptime'] = str(uptime)
                status_result['start_time'] = app.start_time.strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass
        
        # 10. 构建汇总信息
        summary = {
            'status': 'healthy',
            'warnings': [],
            'errors': []
        }
        
        # 检查关键指标，生成警告
        if 'storage_info' in status_result and 'warning' in status_result['storage_info'] and status_result['storage_info']['warning']:
            summary['warnings'].append(f"磁盘空间不足: {status_result['storage_info']['usage_percentage']}%")
        
        if not status_result.get('tushare_connected', False):
            summary['warnings'].append("Tushare API连接失败，可能影响数据获取")
        
        # 如果有错误，更新状态
        if any('error' in section for section in status_result.values() if isinstance(section, dict)):
            summary['status'] = 'warning'
        
        status_result['summary'] = summary
        
        app.logger.info(f'获取系统状态成功 - 状态: {summary["status"]}')
        return jsonify(status_result)
    except Exception as e:
        app.logger.error(f'获取系统状态时发生严重错误: {e}', exc_info=True)
        return jsonify({
            'success': False,
            'message': f'系统状态获取失败: {str(e)}',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error_type': type(e).__name__
        }), 500

# 清理缓存路由
@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    """清理系统缓存 - 增强版，支持选择性清理、时间过滤和安全控制"""
    try:
        app.logger.info('开始处理缓存清理请求')
        
        # 获取并验证请求参数
        data = request.json or {}
        
        # 验证参数
        cache_types = data.get('cache_types', ['plots', 'results'])
        if not isinstance(cache_types, list):
            app.logger.warning('cache_types参数必须是列表类型')
            return jsonify({'success': False, 'message': 'cache_types参数必须是列表类型'}), 400
        
        # 支持的缓存类型
        valid_cache_types = ['plots', 'results']
        for cache_type in cache_types:
            if cache_type not in valid_cache_types:
                app.logger.warning(f'无效的缓存类型: {cache_type}')
                return jsonify({'success': False, 'message': f'无效的缓存类型: {cache_type}。支持的类型: {valid_cache_types}'}), 400
        
        # 时间过滤参数 (可选)
        days_old = data.get('days_old')
        if days_old is not None:
            try:
                days_old = int(days_old)
                if days_old < 0:
                    app.logger.warning(f'无效的days_old参数: {days_old}')
                    return jsonify({'success': False, 'message': 'days_old参数必须是非负整数'}), 400
            except (ValueError, TypeError):
                app.logger.warning(f'days_old参数格式错误: {days_old}')
                return jsonify({'success': False, 'message': 'days_old参数必须是有效的整数'}), 400
        
        # 是否强制删除 (即使有错误也继续)
        force_delete = data.get('force_delete', True)
        
        # 记录开始时间
        start_time = time.time()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 初始化统计信息
        stats = {
            'total': {
                'files_removed': 0,
                'files_skipped': 0,
                'files_failed': 0,
                'total_size': 0,
                'errors': []
            }
        }
        
        # 为每种缓存类型初始化统计
        for cache_type in valid_cache_types:
            stats[cache_type] = {
                'files_removed': 0,
                'files_skipped': 0,
                'files_failed': 0,
                'total_size': 0,
                'errors': [],
                'processed': cache_type in cache_types
            }
        
        # 获取截止时间（如果设置了days_old）
        cutoff_time = None
        if days_old is not None:
            cutoff_time = datetime.now() - timedelta(days=days_old)
            app.logger.info(f'将只清理 {days_old} 天前的文件')
        
        # 定义清理函数
        def clean_directory(directory_path, cache_type_name):
            """清理指定目录下的文件，支持时间过滤"""
            if not os.path.exists(directory_path):
                app.logger.info(f'目录不存在，跳过清理: {directory_path}')
                return
            
            if not os.path.isdir(directory_path):
                app.logger.warning(f'路径不是目录: {directory_path}')
                error_msg = f'路径不是目录: {directory_path}'
                stats['total']['errors'].append(error_msg)
                stats[cache_type_name]['errors'].append(error_msg)
                return
            
            try:
                files = os.listdir(directory_path)
                app.logger.debug(f'开始清理目录: {directory_path}, 包含 {len(files)} 个文件/目录')
                
                for filename in files:
                    file_path = os.path.join(directory_path, filename)
                    
                    # 跳过子目录
                    if not os.path.isfile(file_path):
                        app.logger.debug(f'跳过非文件: {file_path}')
                        continue
                    
                    # 时间过滤检查
                    if cutoff_time:
                        try:
                            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                            if file_mtime > cutoff_time:
                                app.logger.debug(f'文件太新，跳过: {file_path} ({file_mtime})')
                                stats['total']['files_skipped'] += 1
                                stats[cache_type_name]['files_skipped'] += 1
                                continue
                        except Exception as e:
                            app.logger.warning(f'无法获取文件修改时间 {file_path}: {e}')
                            if not force_delete:
                                raise
                    
                    # 删除文件
                    try:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        
                        # 更新统计
                        stats['total']['files_removed'] += 1
                        stats[cache_type_name]['files_removed'] += 1
                        stats['total']['total_size'] += file_size
                        stats[cache_type_name]['total_size'] += file_size
                        
                        app.logger.debug(f'成功删除文件: {file_path}, 大小: {file_size} 字节')
                    except Exception as e:
                        error_msg = f'删除文件失败 {file_path}: {str(e)}'
                        app.logger.warning(error_msg)
                        
                        stats['total']['files_failed'] += 1
                        stats[cache_type_name]['files_failed'] += 1
                        stats['total']['errors'].append(error_msg)
                        stats[cache_type_name]['errors'].append(error_msg)
                        
                        if not force_delete:
                            raise
            except Exception as e:
                error_msg = f'清理目录时出错 {directory_path}: {str(e)}'
                app.logger.error(error_msg, exc_info=True)
                
                stats['total']['errors'].append(error_msg)
                stats[cache_type_name]['errors'].append(error_msg)
                
                if not force_delete:
                    raise
        
        # 执行清理
        cleanup_map = {
            'plots': os.path.join(base_dir, PLOTS_DIR),
            'results': os.path.join(base_dir, RESULTS_DIR)
        }
        
        for cache_type in cache_types:
            app.logger.info(f'开始清理 {cache_type} 缓存')
            clean_directory(cleanup_map[cache_type], cache_type)
        
        # 计算总清理信息
        elapsed_time = time.time() - start_time
        total_size = stats['total']['total_size']
        formatted_size = format_size(total_size)
        
        # 检查清理是否成功
        success = stats['total']['files_failed'] == 0 or force_delete
        
        # 构建详细的统计响应
        response_stats = {
            'total': {
                'files_removed': stats['total']['files_removed'],
                'files_skipped': stats['total']['files_skipped'],
                'files_failed': stats['total']['files_failed'],
                'space_freed': formatted_size,
                'bytes_freed': total_size,
                'time_taken': f'{elapsed_time:.2f} 秒',
                'has_errors': len(stats['total']['errors']) > 0
            }
        }
        
        # 添加每种缓存类型的详细统计
        for cache_type in valid_cache_types:
            if stats[cache_type]['processed']:
                response_stats[cache_type] = {
                    'files_removed': stats[cache_type]['files_removed'],
                    'files_skipped': stats[cache_type]['files_skipped'],
                    'files_failed': stats[cache_type]['files_failed'],
                    'space_freed': format_size(stats[cache_type]['total_size']),
                    'bytes_freed': stats[cache_type]['total_size'],
                    'error_count': len(stats[cache_type]['errors'])
                }
        
        # 生成消息
        if stats['total']['files_removed'] > 0:
            message = f'缓存清理完成: 成功删除 {stats["total"]["files_removed"]} 个文件, 释放空间 {formatted_size}'
        else:
            message = '没有可删除的缓存文件'
        
        if stats['total']['files_failed'] > 0:
            message += f', 但有 {stats["total"]["files_failed"]} 个文件删除失败'
        
        app.logger.info(f'缓存清理操作完成: {message}, 耗时 {elapsed_time:.2f} 秒')
        
        return jsonify({
            'success': success,
            'message': message,
            'stats': response_stats,
            'parameters': {
                'cache_types': cache_types,
                'days_old': days_old,
                'force_delete': force_delete
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'has_warnings': stats['total']['files_failed'] > 0
        })
    except Exception as e:
        app.logger.error(f'清理缓存时发生严重错误: {e}', exc_info=True)
        return jsonify({
            'success': False,
            'message': f'缓存清理失败: {str(e)}',
            'error_type': type(e).__name__,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

# 辅助函数：获取目录大小
def get_directory_size(path):
    total_size = 0
    if os.path.exists(path):
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
    # 转换为MB
    return f"{total_size / (1024 * 1024):.2f} MB"

# 错误处理
@app.errorhandler(404)
def page_not_found(error):
    app.logger.warning('请求的资源不存在')
    return jsonify({'success': False, 'message': '请求的资源不存在'}), 404

@app.errorhandler(500)
def internal_server_error(error):
    app.logger.error(f'服务器内部错误: {str(error)}')
    return jsonify({'success': False, 'message': '服务器内部错误，请稍后再试'}), 500

# 创建选股器实例（全局变量）
stock_selector = None

# 启动前准备
def startup_preparation():
    app.logger.info('开始启动前准备工作')
    
    # 确保必要的目录存在
    dirs_to_create = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
        MODELS_DIR,
        RESULTS_DIR,
        PLOTS_DIR,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    ]
    
    for dir_path in dirs_to_create:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            app.logger.info(f'创建目录: {dir_path}')
    
    # 检查Tushare配置
    if not TUSHARE_TOKEN:
        app.logger.warning('Tushare API Token未配置，请在config.py中设置')
    
    # 检查是否有GPU可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    app.logger.info(f'使用设备: {device}')
    
    # 初始化选股器
    global stock_selector
    stock_selector = StockSelector()
    app.logger.info('选股器初始化完成')
    
    app.logger.info('启动前准备工作完成')

# 获取股票池路由
@app.route('/api/get_stock_pool', methods=['GET'])
def get_stock_pool():
    """获取股票池 - 增强版，包含完整的参数验证、错误处理和详细日志"""
    try:
        app.logger.info('开始处理股票池获取请求')
        
        # 获取并验证请求参数
        industry = request.args.get('industry')
        market = request.args.get('market', 'A')
        
        # 参数验证
        if market is not None:
            valid_markets = ['A', 'SH', 'SZ', 'BJ', 'HK', 'US']
            if market.upper() not in valid_markets:
                app.logger.warning(f'无效的市场参数: {market}, 有效的市场: {valid_markets}')
                return jsonify({
                    'success': False, 
                    'message': f'无效的市场参数，有效的市场: {valid_markets}',
                    'error_type': 'InvalidParameter',
                    'valid_markets': valid_markets
                }), 400
            market = market.upper()
        
        # 验证industry参数格式（可选）
        if industry is not None and not isinstance(industry, str):
            app.logger.warning(f'行业参数必须是字符串类型: {type(industry)}')
            return jsonify({
                'success': False,
                'message': '行业参数必须是字符串类型',
                'error_type': 'InvalidParameterType'
            }), 400
        
        # 记录请求参数
        app.logger.info(f'获取股票池参数: 市场={market}, 行业={industry}')
        
        # 检查选股器是否初始化
        if stock_selector is None:
            app.logger.error('选股器未初始化')
            return jsonify({
                'success': False,
                'message': '选股器未初始化，请稍后再试',
                'error_type': 'ServiceUnavailable'
            }), 503
        
        # 获取股票池 - 添加超时处理
        try:
            pool = stock_selector.fetch_stock_pool(market=market, industry=industry)
        except TimeoutError as e:
            app.logger.error(f'获取股票池超时: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': '获取股票池超时，请稍后再试',
                'error_type': 'TimeoutError'
            }), 504
        except Exception as e:
            app.logger.error(f'获取股票池过程中发生错误: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': f'获取股票池失败: {str(e)}',
                'error_type': 'DataFetchError'
            }), 500
        
        # 验证返回结果
        if pool is None:
            app.logger.warning('股票池获取结果为空')
            return jsonify({
                'success': False,
                'message': '未找到符合条件的股票',
                'error_type': 'NoDataFound',
                'parameters': {'market': market, 'industry': industry}
            }), 404
        
        # 验证数据类型
        if not hasattr(pool, 'to_dict'):
            app.logger.error(f'股票池返回数据格式错误: {type(pool)}')
            return jsonify({
                'success': False,
                'message': '股票池数据格式错误',
                'error_type': 'InvalidDataFormat'
            }), 500
        
        # 转换为字典列表
        try:
            result = pool.to_dict('records')
        except Exception as e:
            app.logger.error(f'数据转换失败: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': '数据处理失败',
                'error_type': 'DataProcessingError'
            }), 500
        
        # 统计结果数量
        count = len(result)
        
        # 构建详细响应
        response = {
            'success': True,
            'data': result,
            'count': count,
            'parameters': {
                'market': market,
                'industry': industry
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': f'成功获取{count}只股票'
        }
        
        app.logger.info(f'获取股票池成功: 市场={market}, 行业={industry}, 数量={count}只')
        return jsonify(response), 200
        
    except Exception as e:
        app.logger.error(f'获取股票池时发生严重错误: {e}', exc_info=True)
        return jsonify({
            'success': False,
            'message': f'服务器处理股票池请求失败: {str(e)}',
            'error_type': type(e).__name__,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

# 股票分析路由
@app.route('/api/analyze_stocks', methods=['POST'])
def analyze_stocks():
    """股票分析路由 - 增强版，包含完整的参数验证、错误处理和详细日志"""
    try:
        app.logger.info('开始处理股票分析请求')
        
        # 获取并验证请求数据
        data = request.json
        if data is None:
            app.logger.warning('请求体为空，需要提供JSON数据')
            return jsonify({
                'success': False,
                'message': '请求体必须包含JSON数据',
                'error_type': 'EmptyRequestBody'
            }), 400
        
        # 参数获取
        stock_codes = data.get('stock_codes', [])
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        limit = data.get('limit', 50)
        
        # 参数验证 - stock_codes必须是列表
        if not isinstance(stock_codes, list):
            app.logger.warning(f'stock_codes参数必须是列表类型: {type(stock_codes)}')
            return jsonify({
                'success': False,
                'message': 'stock_codes参数必须是列表类型',
                'error_type': 'InvalidParameterType'
            }), 400
        
        # 验证stock_codes中的每个元素都是字符串
        for code in stock_codes:
            if not isinstance(code, str):
                app.logger.warning(f'股票代码必须是字符串类型: {type(code)}')
                return jsonify({
                    'success': False,
                    'message': '股票代码必须是字符串类型',
                    'error_type': 'InvalidParameterType'
                }), 400
        
        # 参数验证 - limit必须是正整数
        try:
            limit = int(limit)
            if limit <= 0 or limit > 1000:
                app.logger.warning(f'limit参数超出有效范围: {limit}')
                return jsonify({
                    'success': False,
                    'message': 'limit参数必须在1到1000之间',
                    'error_type': 'InvalidParameterRange'
                }), 400
        except (ValueError, TypeError):
            app.logger.warning(f'limit参数必须是整数: {limit}')
            return jsonify({
                'success': False,
                'message': 'limit参数必须是有效的整数',
                'error_type': 'InvalidParameterType'
            }), 400
        
        # 参数验证 - 日期格式检查
        def validate_date(date_str):
            if date_str is None:
                return True
            try:
                # 检查日期格式 (YYYY-MM-DD)
                import re
                if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                    return False
                # 验证日期有效性
                datetime.strptime(date_str, '%Y-%m-%d')
                return True
            except:
                return False
        
        if start_date is not None and not validate_date(start_date):
            app.logger.warning(f'无效的start_date格式: {start_date}')
            return jsonify({
                'success': False,
                'message': 'start_date参数必须是有效的日期格式 YYYY-MM-DD',
                'error_type': 'InvalidDateFormat'
            }), 400
        
        if end_date is not None and not validate_date(end_date):
            app.logger.warning(f'无效的end_date格式: {end_date}')
            return jsonify({
                'success': False,
                'message': 'end_date参数必须是有效的日期格式 YYYY-MM-DD',
                'error_type': 'InvalidDateFormat'
            }), 400
        
        # 验证日期范围
        if start_date and end_date:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            if start > end:
                app.logger.warning(f'开始日期不能晚于结束日期: {start_date} > {end_date}')
                return jsonify({
                    'success': False,
                    'message': '开始日期不能晚于结束日期',
                    'error_type': 'InvalidDateRange'
                }), 400
        
        # 记录请求参数
        app.logger.info(f'股票分析参数: 股票数量={len(stock_codes)}, 限制={limit}, 日期范围={start_date} 到 {end_date}')
        
        # 检查选股器是否初始化
        if stock_selector is None:
            app.logger.error('选股器未初始化')
            return jsonify({
                'success': False,
                'message': '选股器未初始化，请稍后再试',
                'error_type': 'ServiceUnavailable'
            }), 503
        
        # 构建stock_pool参数
        try:
            stock_pool = None
            if stock_codes:
                stock_pool = pd.DataFrame([{'ts_code': code} for code in stock_codes])
        except Exception as e:
            app.logger.error(f'构建股票池数据失败: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': '构建股票池数据失败',
                'error_type': 'DataProcessingError'
            }), 400
        
        # 执行分析 - 添加超时处理
        try:
            results = stock_selector.analyze_stock_pool(
                stock_pool=stock_pool,
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
        except TimeoutError as e:
            app.logger.error(f'股票分析超时: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': '股票分析超时，请稍后再试',
                'error_type': 'TimeoutError'
            }), 504
        except Exception as e:
            app.logger.error(f'股票分析过程中发生错误: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': f'股票分析失败: {str(e)}',
                'error_type': 'AnalysisError'
            }), 500
        
        # 验证返回结果
        if results is None:
            app.logger.warning('股票分析结果为空')
            return jsonify({
                'success': False,
                'message': '股票分析未找到有效结果',
                'error_type': 'NoDataFound',
                'parameters': {'stock_codes_count': len(stock_codes), 'limit': limit}
            }), 404
        
        # 验证数据类型
        if not hasattr(results, 'to_dict'):
            app.logger.error(f'股票分析返回数据格式错误: {type(results)}')
            return jsonify({
                'success': False,
                'message': '股票分析数据格式错误',
                'error_type': 'InvalidDataFormat'
            }), 500
        
        # 转换为字典列表
        try:
            result_data = results.to_dict('records')
        except Exception as e:
            app.logger.error(f'数据转换失败: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': '数据处理失败',
                'error_type': 'DataProcessingError'
            }), 500
        
        # 统计结果数量
        count = len(result_data)
        
        # 构建详细响应
        response = {
            'success': True,
            'data': result_data,
            'count': count,
            'parameters': {
                'stock_codes_count': len(stock_codes),
                'start_date': start_date,
                'end_date': end_date,
                'limit': limit
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': f'成功分析{count}只股票'
        }
        
        app.logger.info(f'股票分析成功: 分析了{count}只股票, 耗时: {datetime.now().strftime("%H:%M:%S")}')
        return jsonify(response), 200
        
    except Exception as e:
        app.logger.error(f'股票分析时发生严重错误: {e}', exc_info=True)
        return jsonify({
            'success': False,
            'message': f'服务器处理股票分析请求失败: {str(e)}',
            'error_type': type(e).__name__,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

# 股票评分路由
@app.route('/api/score_stocks', methods=['POST'])
def score_stocks():
    """股票评分路由 - 增强版，包含完整的参数验证、错误处理和详细日志"""
    try:
        app.logger.info('开始处理股票评分请求')
        
        # 获取并验证请求数据
        data = request.json
        if data is None:
            app.logger.warning('请求体为空，需要提供JSON数据')
            return jsonify({
                'success': False,
                'message': '请求体必须包含JSON数据',
                'error_type': 'EmptyRequestBody'
            }), 400
        
        # 获取weights参数
        weights = data.get('weights')
        
        # 参数验证 - weights必须是字典或None
        if weights is not None and not isinstance(weights, dict):
            app.logger.warning(f'weights参数必须是字典类型: {type(weights)}')
            return jsonify({
                'success': False,
                'message': 'weights参数必须是字典类型',
                'error_type': 'InvalidParameterType'
            }), 400
        
        # 验证weights字典中的值必须是数字
        if weights is not None:
            for key, value in weights.items():
                if not isinstance(value, (int, float)):
                    app.logger.warning(f'weights值必须是数字类型: 键={key}, 值={value}, 类型={type(value)}')
                    return jsonify({
                        'success': False,
                        'message': f'weights值必须是数字类型，键"{key}"的值类型错误',
                        'error_type': 'InvalidParameterType'
                    }), 400
        
        # 记录请求参数
        app.logger.info(f'股票评分参数: weights={weights}')
        
        # 检查选股器是否初始化
        if stock_selector is None:
            app.logger.error('选股器未初始化')
            return jsonify({
                'success': False,
                'message': '选股器未初始化，请稍后再试',
                'error_type': 'ServiceUnavailable'
            }), 503
        
        # 执行评分 - 添加超时处理
        try:
            scored_results = stock_selector.score_stocks(weights=weights)
        except TimeoutError as e:
            app.logger.error(f'股票评分超时: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': '股票评分超时，请稍后再试',
                'error_type': 'TimeoutError'
            }), 504
        except Exception as e:
            app.logger.error(f'股票评分过程中发生错误: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': f'股票评分失败: {str(e)}',
                'error_type': 'ScoringError'
            }), 500
        
        # 验证返回结果
        if scored_results is None:
            app.logger.warning('股票评分结果为空')
            return jsonify({
                'success': False,
                'message': '股票评分未找到有效结果，请先执行股票分析',
                'error_type': 'NoDataFound',
                'required_steps': ['analyze_stocks']
            }), 404
        
        # 验证数据类型
        if not hasattr(scored_results, 'to_dict'):
            app.logger.error(f'股票评分返回数据格式错误: {type(scored_results)}')
            return jsonify({
                'success': False,
                'message': '股票评分数据格式错误',
                'error_type': 'InvalidDataFormat'
            }), 500
        
        # 转换为字典列表
        try:
            result_data = scored_results.to_dict('records')
        except Exception as e:
            app.logger.error(f'数据转换失败: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': '数据处理失败',
                'error_type': 'DataProcessingError'
            }), 500
        
        # 统计结果数量
        count = len(result_data)
        
        # 构建详细响应
        response = {
            'success': True,
            'data': result_data,
            'count': count,
            'parameters': {
                'weights': weights
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': f'成功为{count}只股票评分'
        }
        
        # 添加评分统计信息
        if result_data:
            try:
                scores = [float(item.get('score', 0)) for item in result_data if 'score' in item]
                if scores:
                    response['statistics'] = {
                        'avg_score': sum(scores) / len(scores),
                        'max_score': max(scores),
                        'min_score': min(scores),
                        'score_count': len(scores)
                    }
            except Exception as e:
                app.logger.warning(f'计算评分统计信息时出错: {e}')
        
        app.logger.info(f'股票评分成功: 评分了{count}只股票, 权重设置: {weights is not None}')
        return jsonify(response), 200
        
    except Exception as e:
        app.logger.error(f'股票评分时发生严重错误: {e}', exc_info=True)
        return jsonify({
            'success': False,
            'message': f'服务器处理股票评分请求失败: {str(e)}',
            'error_type': type(e).__name__,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

# 选择股票路由
@app.route('/api/select_stocks', methods=['POST'])
def select_stocks():
    """选择股票路由 - 增强版，包含完整的参数验证、错误处理和详细日志"""
    try:
        app.logger.info('开始处理选股请求')
        
        # 获取并验证请求数据
        data = request.json
        if data is None:
            app.logger.warning('请求体为空，需要提供JSON数据')
            return jsonify({
                'success': False,
                'message': '请求体必须包含JSON数据',
                'error_type': 'EmptyRequestBody'
            }), 400
        
        # 获取参数
        top_n = data.get('top_n', 10)
        min_score = data.get('min_score')
        
        # 参数验证 - top_n必须是正整数
        try:
            top_n = int(top_n)
            if top_n <= 0 or top_n > 100:
                app.logger.warning(f'top_n参数超出有效范围: {top_n}')
                return jsonify({
                    'success': False,
                    'message': 'top_n参数必须在1到100之间',
                    'error_type': 'InvalidParameterRange'
                }), 400
        except (ValueError, TypeError):
            app.logger.warning(f'top_n参数必须是整数: {top_n}')
            return jsonify({
                'success': False,
                'message': 'top_n参数必须是有效的整数',
                'error_type': 'InvalidParameterType'
            }), 400
        
        # 参数验证 - min_score必须是数字或None
        if min_score is not None:
            try:
                min_score = float(min_score)
                if min_score < 0 or min_score > 100:
                    app.logger.warning(f'min_score参数超出有效范围: {min_score}')
                    return jsonify({
                        'success': False,
                        'message': 'min_score参数必须在0到100之间',
                        'error_type': 'InvalidParameterRange'
                    }), 400
            except (ValueError, TypeError):
                app.logger.warning(f'min_score参数必须是数字: {min_score}')
                return jsonify({
                    'success': False,
                    'message': 'min_score参数必须是有效的数字',
                    'error_type': 'InvalidParameterType'
                }), 400
        
        # 记录请求参数
        app.logger.info(f'选股参数: 数量={top_n}, 最低分数={min_score}')
        
        # 检查选股器是否初始化
        if stock_selector is None:
            app.logger.error('选股器未初始化')
            return jsonify({
                'success': False,
                'message': '选股器未初始化，请稍后再试',
                'error_type': 'ServiceUnavailable'
            }), 503
        
        # 执行选股 - 添加超时处理
        try:
            selected = stock_selector.select_stocks(top_n=top_n, min_score=min_score)
        except TimeoutError as e:
            app.logger.error(f'选股超时: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': '选股超时，请稍后再试',
                'error_type': 'TimeoutError'
            }), 504
        except Exception as e:
            app.logger.error(f'选股过程中发生错误: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': f'选股失败: {str(e)}',
                'error_type': 'SelectionError'
            }), 500
        
        # 验证返回结果
        if selected is None:
            app.logger.warning('选股结果为空')
            return jsonify({
                'success': False,
                'message': '选股未找到有效结果，请先执行股票分析和评分',
                'error_type': 'NoDataFound',
                'required_steps': ['analyze_stocks', 'score_stocks']
            }), 404
        
        # 验证数据类型
        if not hasattr(selected, 'to_dict'):
            app.logger.error(f'选股返回数据格式错误: {type(selected)}')
            return jsonify({
                'success': False,
                'message': '选股数据格式错误',
                'error_type': 'InvalidDataFormat'
            }), 500
        
        # 转换为字典列表
        try:
            result_data = selected.to_dict('records')
        except Exception as e:
            app.logger.error(f'数据转换失败: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': '数据处理失败',
                'error_type': 'DataProcessingError'
            }), 500
        
        # 统计结果数量
        count = len(result_data)
        
        # 构建详细响应
        response = {
            'success': True,
            'data': result_data,
            'count': count,
            'parameters': {
                'top_n': top_n,
                'min_score': min_score
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': f'成功选择{count}只股票'
        }
        
        # 添加选股统计信息
        if result_data:
            try:
                scores = [float(item.get('score', 0)) for item in result_data if 'score' in item]
                if scores:
                    response['statistics'] = {
                        'avg_score': sum(scores) / len(scores),
                        'max_score': max(scores),
                        'min_score': min(scores),
                        'score_count': len(scores)
                    }
            except Exception as e:
                app.logger.warning(f'计算选股统计信息时出错: {e}')
        
        app.logger.info(f'选股成功: 选择了{count}只股票, 按top_n={top_n}和min_score={min_score}过滤')
        return jsonify(response), 200
        
    except Exception as e:
        app.logger.error(f'选股时发生严重错误: {e}', exc_info=True)
        return jsonify({
            'success': False,
            'message': f'服务器处理选股请求失败: {str(e)}',
            'error_type': type(e).__name__,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

# 导出选股结果路由
@app.route('/api/export_selection', methods=['POST'])
def export_selection():
    """导出选股结果路由 - 增强版，包含完整的参数验证、错误处理和详细日志"""
    try:
        app.logger.info('开始处理导出选股结果请求')
        
        # 获取并验证请求数据
        data = request.json
        if data is None:
            app.logger.warning('请求体为空，需要提供JSON数据')
            return jsonify({
                'success': False,
                'message': '请求体必须包含JSON数据',
                'error_type': 'EmptyRequestBody'
            }), 400
        
        # 获取参数
        filename = data.get('filename')
        format = data.get('format', 'csv')
        
        # 参数验证 - format必须是支持的格式
        valid_formats = ['csv', 'excel', 'xlsx', 'json', 'pickle']
        if format.lower() not in valid_formats:
            app.logger.warning(f'无效的格式参数: {format}, 有效的格式: {valid_formats}')
            return jsonify({
                'success': False,
                'message': f'无效的格式参数，有效的格式: {valid_formats}',
                'error_type': 'InvalidParameter',
                'valid_formats': valid_formats
            }), 400
        
        format = format.lower()
        
        # 参数验证 - filename必须是字符串或None
        if filename is not None and not isinstance(filename, str):
            app.logger.warning(f'文件名参数必须是字符串类型: {type(filename)}')
            return jsonify({
                'success': False,
                'message': '文件名参数必须是字符串类型',
                'error_type': 'InvalidParameterType'
            }), 400
        
        # 验证文件名安全性（如果提供）
        if filename:
            # 检查文件名是否包含非法字符
            import re
            if not re.match(r'^[a-zA-Z0-9_\-\.]+$', filename):
                app.logger.warning(f'文件名包含非法字符: {filename}')
                return jsonify({
                    'success': False,
                    'message': '文件名只能包含字母、数字、下划线、连字符和点',
                    'error_type': 'InvalidFileName'
                }), 400
            
            # 检查文件名长度
            if len(filename) > 100:
                app.logger.warning(f'文件名过长: {len(filename)} 字符')
                return jsonify({
                    'success': False,
                    'message': '文件名长度不能超过100个字符',
                    'error_type': 'FileNameTooLong'
                }), 400
        
        # 记录请求参数
        app.logger.info(f'导出参数: 文件名={filename}, 格式={format}')
        
        # 检查选股器是否初始化
        if stock_selector is None:
            app.logger.error('选股器未初始化')
            return jsonify({
                'success': False,
                'message': '选股器未初始化，请稍后再试',
                'error_type': 'ServiceUnavailable'
            }), 503
        
        # 执行导出 - 添加超时处理
        try:
            filepath = stock_selector.export_results(filename=filename, format=format)
        except TimeoutError as e:
            app.logger.error(f'导出超时: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': '导出超时，请稍后再试',
                'error_type': 'TimeoutError'
            }), 504
        except PermissionError as e:
            app.logger.error(f'导出权限错误: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': '文件系统权限不足，无法创建导出文件',
                'error_type': 'PermissionError'
            }), 403
        except Exception as e:
            app.logger.error(f'导出过程中发生错误: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': f'导出失败: {str(e)}',
                'error_type': 'ExportError'
            }), 500
        
        # 验证返回结果
        if filepath is None:
            app.logger.warning('导出结果为空')
            return jsonify({
                'success': False,
                'message': '导出失败，请先执行选股',
                'error_type': 'NoDataToExport',
                'required_steps': ['select_stocks']
            }), 404
        
        # 验证文件是否实际创建
        if not os.path.exists(filepath):
            app.logger.error(f'导出文件不存在: {filepath}')
            return jsonify({
                'success': False,
                'message': '导出文件创建失败',
                'error_type': 'FileCreationError'
            }), 500
        
        # 验证文件大小
        try:
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                app.logger.warning(f'导出文件为空: {filepath}')
                return jsonify({
                    'success': False,
                    'message': '导出文件为空，请检查选股结果',
                    'error_type': 'EmptyFile'
                }), 400
        except Exception as e:
            app.logger.warning(f'无法获取文件大小: {e}')
        
        # 获取文件名
        try:
            basename = os.path.basename(filepath)
        except Exception as e:
            app.logger.error(f'无法获取文件名: {e}', exc_info=True)
            basename = 'exported_result'
        
        # 构建详细响应
        response = {
            'success': True,
            'file_path': filepath,
            'filename': basename,
            'format': format,
            'parameters': {
                'filename': filename,
                'format': format
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': f'成功导出选股结果到文件 {basename}'
        }
        
        # 添加文件信息
        try:
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                response['file_info'] = {
                    'size_bytes': file_size,
                    'size_human': format_size(file_size),
                    'creation_time': datetime.fromtimestamp(os.path.getctime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
                }
        except Exception as e:
            app.logger.warning(f'获取文件信息时出错: {e}')
        
        app.logger.info(f'导出成功: {filepath}, 格式: {format}, 大小: {format_size(file_size) if os.path.exists(filepath) else "未知"}')
        return jsonify(response), 200
        
    except Exception as e:
        app.logger.error(f'导出选股结果时发生严重错误: {e}', exc_info=True)
        return jsonify({
            'success': False,
            'message': f'服务器处理导出请求失败: {str(e)}',
            'error_type': type(e).__name__,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

# 可视化选股结果路由
@app.route('/api/visualize_selection', methods=['POST'])
def visualize_selection():
    """可视化选股结果路由 - 增强版，包含完整的参数验证、错误处理和详细日志"""
    try:
        app.logger.info('开始处理可视化选股结果请求')
        
        # 获取并验证请求数据
        data = request.json
        if data is None:
            app.logger.warning('请求体为空，需要提供JSON数据')
            return jsonify({
                'success': False,
                'message': '请求体必须包含JSON数据',
                'error_type': 'EmptyRequestBody'
            }), 400
        
        # 获取参数
        filename = data.get('filename')
        
        # 参数验证 - filename必须是字符串或None
        if filename is not None and not isinstance(filename, str):
            app.logger.warning(f'文件名参数必须是字符串类型: {type(filename)}')
            return jsonify({
                'success': False,
                'message': '文件名参数必须是字符串类型',
                'error_type': 'InvalidParameterType'
            }), 400
        
        # 验证文件名安全性（如果提供）
        if filename:
            # 检查文件名是否包含非法字符
            import re
            if not re.match(r'^[a-zA-Z0-9_\-\.]+$', filename):
                app.logger.warning(f'文件名包含非法字符: {filename}')
                return jsonify({
                    'success': False,
                    'message': '文件名只能包含字母、数字、下划线、连字符和点',
                    'error_type': 'InvalidFileName'
                }), 400
            
            # 检查文件名长度
            if len(filename) > 100:
                app.logger.warning(f'文件名过长: {len(filename)} 字符')
                return jsonify({
                    'success': False,
                    'message': '文件名长度不能超过100个字符',
                    'error_type': 'FileNameTooLong'
                }), 400
        
        # 记录请求参数
        app.logger.info(f'可视化参数: 文件名={filename}')
        
        # 检查选股器是否初始化
        if stock_selector is None:
            app.logger.error('选股器未初始化')
            return jsonify({
                'success': False,
                'message': '选股器未初始化，请稍后再试',
                'error_type': 'ServiceUnavailable'
            }), 503
        
        # 执行可视化 - 添加超时处理
        try:
            filepath = stock_selector.visualize_results(filename=filename)
        except TimeoutError as e:
            app.logger.error(f'可视化超时: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': '可视化超时，请稍后再试',
                'error_type': 'TimeoutError'
            }), 504
        except PermissionError as e:
            app.logger.error(f'可视化权限错误: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': '文件系统权限不足，无法创建可视化文件',
                'error_type': 'PermissionError'
            }), 403
        except MemoryError as e:
            app.logger.error(f'可视化内存错误: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': '可视化过程中内存不足，请减少数据量',
                'error_type': 'MemoryError'
            }), 507
        except Exception as e:
            app.logger.error(f'可视化过程中发生错误: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': f'可视化失败: {str(e)}',
                'error_type': 'VisualizationError'
            }), 500
        
        # 验证返回结果
        if filepath is None:
            app.logger.warning('可视化结果为空')
            return jsonify({
                'success': False,
                'message': '可视化失败，请先执行选股',
                'error_type': 'NoDataToVisualize',
                'required_steps': ['select_stocks']
            }), 404
        
        # 验证文件是否实际创建
        if not os.path.exists(filepath):
            app.logger.error(f'可视化文件不存在: {filepath}')
            return jsonify({
                'success': False,
                'message': '可视化文件创建失败',
                'error_type': 'FileCreationError'
            }), 500
        
        # 验证文件大小
        try:
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                app.logger.warning(f'可视化文件为空: {filepath}')
                return jsonify({
                    'success': False,
                    'message': '可视化文件为空，请检查选股结果',
                    'error_type': 'EmptyFile'
                }), 400
        except Exception as e:
            app.logger.warning(f'无法获取文件大小: {e}')
        
        # 获取相对路径供前端访问
        try:
            relative_path = os.path.basename(filepath)
            url = f'/plots/{relative_path}'
        except Exception as e:
            app.logger.error(f'无法生成文件路径: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'message': '文件路径生成失败',
                'error_type': 'PathGenerationError'
            }), 500
        
        # 构建详细响应
        response = {
            'success': True,
            'file_path': relative_path,
            'url': url,
            'parameters': {
                'filename': filename
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': f'成功生成选股结果可视化图表，可通过URL访问: {url}'
        }
        
        # 添加文件信息
        try:
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                response['file_info'] = {
                    'size_bytes': file_size,
                    'size_human': format_size(file_size),
                    'creation_time': datetime.fromtimestamp(os.path.getctime(filepath)).strftime('%Y-%m-%d %H:%M:%S'),
                    'extension': os.path.splitext(filepath)[1].lower() if filepath else ''
                }
        except Exception as e:
            app.logger.warning(f'获取文件信息时出错: {e}')
        
        app.logger.info(f'可视化成功: {filepath}, 大小: {format_size(file_size) if os.path.exists(filepath) else "未知"}, 可通过URL访问: {url}')
        return jsonify(response), 200
        
    except Exception as e:
        app.logger.error(f'可视化选股结果时发生严重错误: {e}', exc_info=True)
        return jsonify({
            'success': False,
            'message': f'服务器处理可视化请求失败: {str(e)}',
            'error_type': type(e).__name__,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

# 添加图表文件访问路由
@app.route('/plots/<path:filename>')
def serve_plots(filename):
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), PLOTS_DIR)
    return send_from_directory(plot_dir, filename)

# 结果文件访问路由
@app.route('/results/<path:filename>')
def serve_results(filename):
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_DIR)
    return send_from_directory(results_dir, filename)

if __name__ == '__main__':
    # 启动前准备
    startup_preparation()
    
    # 运行Flask应用
    app.logger.info('启动股票预测系统Web服务...')
    app.run(host='0.0.0.0', port=5000, debug=False)