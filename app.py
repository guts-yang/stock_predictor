# 导入所需库
from flask import Flask, render_template, request, jsonify, send_file, make_response, send_from_directory
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
import shutil
from datetime import datetime
import time
import webbrowser

# 添加股票数据模块导入，用于获取股票名称
from stock_data import get_stock_name

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from stock_data import fetch_stock_data, save_stock_data, get_latest_stock_data, get_device
from train_stock_model import train_stock_model
from predict_stock import main_predict, batch_predict_stocks, load_trained_model, predict_stock_price, get_default_model_path, plot_kline_prediction
from config import DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_LAYERS, DEFAULT_SEQUENCE_LENGTH, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, DEFAULT_PATIENCE, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, TUSHARE_TOKEN, MAX_BATCH_PREDICTION, SYSTEM_VERSION

# 创建Flask应用
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False  # 保持JSON响应中键的顺序
app.config['JSON_AS_ASCII'] = False    # 确保中文正常显示

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
    if size_bytes < 0:
        return '0 B'
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_bytes)
    unit_index = 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.2f} {units[unit_index]}"

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

# 获取股票数据路由
@app.route('/api/fetch_data', methods=['POST'])
def fetch_data():
    try:
        data = request.json
        stock_code = data.get('stock_code')
        start_date = data.get('start_date', DEFAULT_START_DATE)
        end_date = data.get('end_date', DEFAULT_END_DATE)
        save_dir = data.get('save_dir', 'data')
        
        if not stock_code:
            return jsonify({'success': False, 'message': '请提供股票代码'})
        
        app.logger.info(f'正在获取股票数据: {stock_code}, 日期范围: {start_date} - {end_date}')
        
        # 获取股票数据
        df = fetch_stock_data(stock_code, start_date, end_date)
        
        if df is None or df.empty:
            app.logger.warning(f'无法获取股票数据: {stock_code}')
            return jsonify({'success': False, 'message': '无法获取股票数据'})
        
        # 保存数据
        success = save_stock_data(stock_code, start_date, end_date, save_dir)
        
        if success:
            # 准备数据用于前端显示
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            stock_data = {
                'dates': df['trade_date'].dt.strftime('%Y-%m-%d').tolist(),
                'close_prices': df['close'].tolist(),
                'open_prices': df['open'].tolist(),
                'high_prices': df['high'].tolist(),
                'low_prices': df['low'].tolist(),
                'volume': df['vol'].tolist(),
                'ma5': df['ma5'].tolist() if 'ma5' in df.columns else [],
                'ma10': df['ma10'].tolist() if 'ma10' in df.columns else []
            }
            
            app.logger.info(f'股票数据获取成功: {stock_code}, 数据点数: {len(df)}')
            return jsonify({'success': True, 'data': stock_data, 'message': '数据获取成功'})
        else:
            app.logger.warning(f'数据保存失败: {stock_code}')
            return jsonify({'success': False, 'message': '数据保存失败'})
    except Exception as e:
        app.logger.error(f'获取股票数据时出错: {str(e)}')
        return jsonify({'success': False, 'message': str(e)})

# 训练模型路由
@app.route('/api/train_model', methods=['POST'])
def train_model():
    try:
        data = request.json
        stock_code = data.get('stock_code')
        start_date = data.get('start_date', DEFAULT_START_DATE)
        end_date = data.get('end_date', DEFAULT_END_DATE)
        model_type = data.get('model_type', 'baseline')
        hidden_size = data.get('hidden_size', DEFAULT_HIDDEN_SIZE)
        num_layers = data.get('num_layers', DEFAULT_NUM_LAYERS)
        sequence_length = data.get('sequence_length', DEFAULT_SEQUENCE_LENGTH)
        batch_size = data.get('batch_size', DEFAULT_BATCH_SIZE)
        learning_rate = data.get('learning_rate', DEFAULT_LEARNING_RATE)
        epochs = data.get('epochs', DEFAULT_EPOCHS)
        patience = data.get('patience', DEFAULT_PATIENCE)
        
        if not stock_code:
            return jsonify({'success': False, 'message': '请提供股票代码'})
        
        app.logger.info(f'开始训练模型: 股票={stock_code}, 模型类型={model_type}, 隐藏层大小={hidden_size}, 层数={num_layers}')
        
        # 获取设备
        device = get_device()
        app.logger.info(f'使用设备: {device}')
        
        # 调用训练函数
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
        
        if model_path:
            app.logger.info(f'模型训练成功: {model_path}')
            return jsonify({'success': True, 'model_path': model_path, 'message': '模型训练成功'})
        else:
            app.logger.warning(f'模型训练失败: {stock_code}')
            return jsonify({'success': False, 'message': '模型训练失败'})
    except Exception as e:
        app.logger.error(f'训练模型时出错: {str(e)}')
        return jsonify({'success': False, 'message': str(e)})

# 预测单只股票路由
@app.route('/api/predict_single', methods=['POST'])
def predict_single():
    try:
        data = request.json
        stock_code = data.get('stock_code')
        model_type = data.get('model_type', 'baseline')
        model_path = data.get('model_path')
        prediction_days = int(data.get('prediction_days', 5)) if model_type == 'kline_lstm' else None
        
        if not stock_code:
            return jsonify({'success': False, 'message': '请提供股票代码'})
        
        app.logger.info(f'开始预测单只股票: {stock_code}, 模型类型: {model_type}')
        
        # 进行预测
        success, result = main_predict(stock_code, model_path, model_type, prediction_days)
        
        if success:
            if model_type == 'kline_lstm':
                # K线模型预测结果
                app.logger.info(f'K线预测成功: {stock_code}, 预测未来{prediction_days}天')
                
                # 格式化预测数据以便前端展示
                formatted_predictions = []
                for pred in result['predictions']:
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
                        'close': result['metadata']['latest_close']
                    },
                    'prediction_days': prediction_days,
                    'message': f'K线预测成功，预测未来{prediction_days}天'
                })
            else:
                # 基线模型预测结果
                app.logger.info(f'股票预测成功: {stock_code}, 预测价格: {result['predicted_price']}, 涨跌幅: {result['price_change_percent']:.2f}%')
                return jsonify({
                    'success': True,
                    'stock_code': stock_code,
                    'model_type': 'baseline',
                    'latest_close': result['latest_close'],
                    'predicted_price': result['predicted_price'],
                    'price_change_percent': result['price_change_percent'],
                    'chart_path': result['plot_path'],
                    'message': '预测成功'
                })
        
        app.logger.warning(f'股票预测失败: {stock_code}')
        return jsonify({'success': False, 'message': '预测失败，请确保模型已训练'})
    except Exception as e:
        app.logger.error(f'预测股票时出错: {str(e)}')
        return jsonify({'success': False, 'message': str(e)})

# 批量预测路由
@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.json
        stock_codes = data.get('stock_codes', [])
        model_type = data.get('model_type', 'baseline')
        
        if not stock_codes:
            return jsonify({'success': False, 'message': '请至少选择一只股票'})
        
        # 限制批量预测数量
        if len(stock_codes) > MAX_BATCH_PREDICTION:
            app.logger.warning(f'批量预测数量超过限制: {len(stock_codes)} > {MAX_BATCH_PREDICTION}')
            return jsonify({'success': False, 'message': f'批量预测数量已限制为{MAX_BATCH_PREDICTION}只股票'})
        
        app.logger.info(f'开始批量预测: {len(stock_codes)}只股票, 模型类型: {model_type}')
        
        # 使用第一个股票代码的模型进行批量预测
        reference_stock = stock_codes[0]
        model_path = get_default_model_path(reference_stock, model_type)
        
        # 检查模型是否存在
        if not os.path.exists(model_path):
            app.logger.warning(f'未找到股票 {reference_stock} 的模型')
            return jsonify({'success': False, 'message': f'未找到股票 {reference_stock} 的模型，请先训练模型'})
        
        # 获取设备
        device = get_device()
        
        # 进行批量预测
        results = batch_predict_stocks(model_path, stock_codes, device, model_type)
        
        if results:
            # 保存结果到CSV
            result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_DIR)
            os.makedirs(result_dir, exist_ok=True)
            
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_file = os.path.join(result_dir, f'batch_prediction_{current_time}.csv')
            
            df = pd.DataFrame(results)
            df.to_csv(result_file, index=False, encoding='utf-8-sig')
            
            app.logger.info(f'批量预测成功: {len(results)}只股票, 结果文件: {result_file}')
            return jsonify({'success': True, 'results': results, 'file_path': result_file, 'message': '批量预测成功'})
        else:
            app.logger.warning('批量预测失败')
            return jsonify({'success': False, 'message': '批量预测失败'})
    except Exception as e:
        app.logger.error(f'批量预测时出错: {str(e)}')
        return jsonify({'success': False, 'message': str(e)})

# 获取已训练模型列表路由
@app.route('/api/get_models', methods=['GET'])
def get_models():
    try:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODELS_DIR)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            return jsonify({'success': True, 'models': []})
        
        # 获取所有模型文件
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        
        # 解析模型信息
        models = []
        for model_file in model_files:
            try:
                model_path = os.path.join(model_dir, model_file)
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                
                models.append({
                    'filename': model_file,
                    'stock_code': checkpoint.get('stock_code', 'unknown'),
                    'model_type': checkpoint.get('model_type', 'unknown'),
                    'hidden_size': checkpoint.get('hidden_size', 64),
                    'num_layers': checkpoint.get('num_layers', 2),
                    'sequence_length': checkpoint.get('sequence_length', 5),
                    'file_path': model_path,
                    'last_modified': datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S'),
                    'file_size': f"{os.path.getsize(model_path) / 1024:.2f} KB"
                })
            except Exception as e:
                app.logger.warning(f'无法加载模型文件 {model_file}: {str(e)}')
                # 忽略无法加载的模型文件
                continue
        
        # 按修改时间倒序排列
        models.sort(key=lambda x: x['last_modified'], reverse=True)
        
        app.logger.info(f'获取模型列表成功: {len(models)}个模型')
        return jsonify({'success': True, 'models': models})
    except Exception as e:
        app.logger.error(f'获取模型列表时出错: {str(e)}')
        return jsonify({'success': False, 'message': str(e)})

# 删除模型路由
@app.route('/api/delete_model', methods=['POST'])
def delete_model():
    try:
        data = request.json
        model_filename = data.get('filename')
        
        if not model_filename:
            return jsonify({'success': False, 'message': '请提供模型文件名'})
        
        # 构建模型文件路径
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODELS_DIR)
        model_path = os.path.join(model_dir, model_filename)
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            return jsonify({'success': False, 'message': '模型文件不存在'})
        
        # 删除模型文件
        os.remove(model_path)
        app.logger.info(f'模型文件已删除: {model_filename}')
        
        return jsonify({'success': True, 'message': '模型删除成功'})
    except Exception as e:
        app.logger.error(f'删除模型时出错: {str(e)}')
        return jsonify({'success': False, 'message': str(e)})

# 生成图表路由
@app.route('/api/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.json
        chart_type = data.get('chart_type')
        
        # 创建图表
        if chart_type == 'stock_price':
            # 股票价格图表
            fig, ax = plt.subplots(figsize=(12, 6))
            dates = data.get('dates', [])
            close_prices = data.get('close_prices', [])
            ma5 = data.get('ma5', [])
            ma10 = data.get('ma10', [])
            
            ax.plot(dates, close_prices, label='收盘价', color='#165DFF', linewidth=2)
            
            if ma5:
                ax.plot(dates, ma5, label='5日均线', color='#52C41A', linewidth=1.5, linestyle='--')
            
            if ma10:
                ax.plot(dates, ma10, label='10日均线', color='#FAAD14', linewidth=1.5, linestyle='--')
            
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
                
        elif chart_type == 'prediction_result':
            # 预测结果图表
            fig, ax = plt.subplots(figsize=(12, 6))
            stock_code = data.get('stock_code')
            latest_close = data.get('latest_close')
            predicted_price = data.get('predicted_price')
            
            # 获取股票名称
            stock_name = get_stock_name(stock_code)
            
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
            change_percent = ((predicted_price - latest_close) / latest_close) * 100
            change_text = f'预测{'上涨' if change_percent > 0 else '下跌'}: {abs(change_percent):.2f}%'
            ax.text(0.5, 0.95, change_text, ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            
            # 添加网格线
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
        elif chart_type == 'kline_chart':
            # K线图生成
            import matplotlib.dates as mdates
            from datetime import datetime, timedelta
            
            stock_code = data.get('stock_code')
            stock_name = get_stock_name(stock_code)
            
            # 获取历史数据
            history_data = data.get('history_data', {})
            df = pd.DataFrame(history_data)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            # 获取预测数据
            predictions = data.get('predictions', [])
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
            fig.suptitle(f'股票{stock_name}({stock_code}) K线预测图', fontsize=16, fontweight='bold')
            
            # 绘制历史K线
            for i, row in df.iterrows():
                # 判断涨跌
                if row['close'] >= row['open']:
                    color = 'red'  # 涨
                else:
                    color = 'green'  # 跌
                
                # 绘制蜡烛线
                ax1.plot([row['trade_date'], row['trade_date']], [row['low'], row['high']], color=color, linewidth=1.5)
                ax1.plot([row['trade_date'], row['trade_date']], [row['open'], row['close']], color=color, linewidth=6)
            
            # 绘制移动平均线
            ax1.plot(df['trade_date'], df['ma5'], color='blue', label='MA5', linewidth=1.2)
            ax1.plot(df['trade_date'], df['ma10'], color='orange', label='MA10', linewidth=1.2)
            
            # 如果有预测数据，绘制预测K线
            if predictions:
                # 生成预测日期
                last_date = df['trade_date'].iloc[-1]
                pred_dates = [last_date + timedelta(days=i+1) for i in range(len(predictions))]
                
                # 绘制预测K线（使用虚线和半透明效果）
                for i, (pred, date) in enumerate(zip(predictions, pred_dates)):
                    open_price, high, low, close, volume = pred['open'], pred['high'], pred['low'], pred['close'], pred['volume']
                    
                    # 判断涨跌
                    if close >= open_price:
                        color = 'red'  # 涨
                    else:
                        color = 'green'  # 跌
                    
                    # 绘制预测蜡烛线（使用虚线）
                    ax1.plot([date, date], [low, high], color=color, linewidth=1.5, linestyle='--', alpha=0.8)
                    ax1.plot([date, date], [open_price, close], color=color, linewidth=6, alpha=0.8)
            
            # 设置主图属性
            ax1.set_ylabel('价格', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.tick_params(axis='x', rotation=45)
            
            # 绘制成交量
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
                    volume = pred['volume']
                    if pred['close'] >= pred['open']:  # close >= open
                        color = 'red'
                    else:
                        color = 'green'
                    ax2.bar(date, volume, color=color, alpha=0.7, linestyle='--', edgecolor='black', linewidth=0.5)
            
            # 设置成交量图属性
            ax2.set_ylabel('成交量', fontsize=12)
            ax2.set_xlabel('日期', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.tick_params(axis='x', rotation=45)
            
            # 调整布局
            plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        plt.tight_layout()
        
        # 将图表转换为base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        app.logger.info(f'生成图表成功: {chart_type}')
        return jsonify({'success': True, 'image_data': image_base64})
    except Exception as e:
        app.logger.error(f'生成图表时出错: {str(e)}')
        return jsonify({'success': False, 'message': str(e)})

# 下载预测结果路由
@app.route('/api/download_result', methods=['GET'])
def download_result():
    try:
        file_path = request.args.get('file_path')
        
        if not file_path or not os.path.exists(file_path):
            app.logger.warning(f'请求下载的文件不存在: {file_path}')
            return jsonify({'success': False, 'message': '文件不存在'})
        
        # 设置中文文件名
        filename = os.path.basename(file_path)
        response = make_response(send_file(file_path, as_attachment=True))
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"; filename*=UTF-8''{filename}'
        
        app.logger.info(f'文件下载成功: {file_path}')
        return response
    except Exception as e:
        app.logger.error(f'下载文件时出错: {str(e)}')
        return jsonify({'success': False, 'message': str(e)})

# 获取K线实时预览
@app.route('/api/get_kline_preview', methods=['POST'])
def get_kline_preview():
    try:
        data = request.json
        stock_code = data.get('stock_code')
        days = int(data.get('days', 30))
        
        app.logger.info(f'正在生成股票K线预览: {stock_code}, 天数: {days}')
        
        # 获取最新股票数据
        _, _, metadata = get_latest_stock_data(stock_code, days, model_type='kline_lstm')
        df = metadata['df'].copy()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 只取最近days天
        recent_df = df.tail(days).copy()
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # 获取股票名称
        stock_name = get_stock_name(stock_code)
        fig.suptitle(f'股票{stock_name}({stock_code}) 最近{days}天K线图', fontsize=14, fontweight='bold')
        
        # 绘制历史K线
        for i, row in recent_df.iterrows():
            # 判断涨跌
            if row['close'] >= row['open']:
                color = 'red'  # 涨
            else:
                color = 'green'  # 跌
            
            # 绘制蜡烛线
            ax1.plot([row['trade_date'], row['trade_date']], [row['low'], row['high']], color=color, linewidth=1.5)
            ax1.plot([row['trade_date'], row['trade_date']], [row['open'], row['close']], color=color, linewidth=6)
        
        # 绘制移动平均线
        ax1.plot(recent_df['trade_date'], recent_df['ma5'], color='blue', label='MA5', linewidth=1.2)
        ax1.plot(recent_df['trade_date'], recent_df['ma10'], color='orange', label='MA10', linewidth=1.2)
        
        # 设置主图属性
        ax1.set_ylabel('价格', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
        ax1.tick_params(axis='x', rotation=45)
        
        # 绘制成交量
        for i, row in recent_df.iterrows():
            if row['close'] >= row['open']:
                color = 'red'
            else:
                color = 'green'
            ax2.bar(row['trade_date'], row['vol'], color=color, alpha=0.7)
        
        # 设置成交量图属性
        ax2.set_ylabel('成交量', fontsize=10)
        ax2.set_xlabel('日期', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
        ax2.tick_params(axis='x', rotation=45)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # 将图表转换为base64编码
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        app.logger.info(f'K线预览生成成功: {stock_code}')
        return jsonify({
            'success': True,
            'image_base64': image_base64
        })
    except Exception as e:
        app.logger.error(f'生成K线预览时出错: {str(e)}')
        return jsonify({'success': False, 'message': str(e)})

# 获取系统状态路由
@app.route('/api/system_status', methods=['GET'])
def system_status():
    try:
        # 检查GPU是否可用
        has_gpu = torch.cuda.is_available()
        device = 'cuda' if has_gpu else 'cpu'
        
        # 检查Tushare连接
        tushare_connected = False
        try:
            if TUSHARE_TOKEN:
                from stock_data import pro
                test = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name')
                tushare_connected = len(test) > 0
        except Exception as e:
            app.logger.warning(f'Tushare连接测试失败: {str(e)}')
            tushare_connected = False
        
        # 检查模型和数据目录
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODELS_DIR)
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # 获取目录统计信息
        model_count = len([f for f in os.listdir(model_dir) if f.endswith('.pth')])
        data_count = len([f for f in os.listdir(data_dir) if f.endswith('.csv')])
        
        # 获取各目录空间使用情况
        dir_sizes = {
            'data': get_directory_size(data_dir),
            'models': get_directory_size(model_dir),
            'results': get_directory_size(os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_DIR)),
            'plots': get_directory_size(os.path.join(os.path.dirname(os.path.abspath(__file__)), PLOTS_DIR))
        }
        
        # 获取Python版本信息
        python_version = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
        
        # 获取存储使用情况
        storage_info = {}
        try:
            # 获取当前磁盘信息
            disk_usage = psutil.disk_usage('.')
            used = format_size(disk_usage.used)
            total = format_size(disk_usage.total)
            usage_percentage = int(disk_usage.percent)
            
            storage_info = {
                'used': used,
                'total': total,
                'usage_percentage': usage_percentage
            }
        except Exception as e:
            app.logger.error(f"获取存储信息失败: {e}")
        
        app.logger.info('获取系统状态成功')
        
        return jsonify({
            'success': True,
            'has_gpu': has_gpu,
            'device': device,
            'tushare_connected': tushare_connected,
            'model_count': model_count,
            'data_count': data_count,
            'dir_sizes': dir_sizes,
            'system_version': SYSTEM_VERSION,
            'python_version': python_version,
            'storage_info': storage_info,
            'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features': {
                'kline_prediction': True,
                'multi_feature_prediction': True,
                'technical_indicators': True
            }
        })
    except Exception as e:
        app.logger.error(f'获取系统状态时出错: {str(e)}')
        return jsonify({'success': False, 'message': str(e)})

# 清理缓存路由
@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 统计信息
        stats = {
            'plots': {'files_removed': 0, 'total_size': 0},
            'results': {'files_removed': 0, 'total_size': 0}
        }
        
        # 清理图表缓存
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), PLOTS_DIR)
        if os.path.exists(plots_dir):
            for filename in os.listdir(plots_dir):
                file_path = os.path.join(plots_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        stats['plots']['files_removed'] += 1
                        stats['plots']['total_size'] += file_size
                    except Exception as e:
                        app.logger.warning(f'无法删除文件 {file_path}: {str(e)}')
        
        # 清理结果缓存
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_DIR)
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                file_path = os.path.join(results_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        stats['results']['files_removed'] += 1
                        stats['results']['total_size'] += file_size
                    except Exception as e:
                        app.logger.warning(f'无法删除文件 {file_path}: {str(e)}')
        
        # 计算总清理信息
        total_files = stats['plots']['files_removed'] + stats['results']['files_removed']
        total_size = stats['plots']['total_size'] + stats['results']['total_size']
        elapsed_time = time.time() - start_time
        
        # 格式化大小
        formatted_size = format_size(total_size)
        
        app.logger.info(f'缓存清理完成: 删除了 {total_files} 个文件, 释放空间 {formatted_size}, 耗时 {elapsed_time:.2f} 秒')
        
        return jsonify({
            'success': True, 
            'message': f'缓存清理完成: 删除了 {total_files} 个文件, 释放空间 {formatted_size}',
            'stats': {
                'total_files_removed': total_files,
                'total_space_freed': formatted_size,
                'plots_files_removed': stats['plots']['files_removed'],
                'plots_space_freed': format_size(stats['plots']['total_size']),
                'results_files_removed': stats['results']['files_removed'],
                'results_space_freed': format_size(stats['results']['total_size']),
                'time_taken': f'{elapsed_time:.2f} 秒'
            }
        })
    except Exception as e:
        app.logger.error(f'清理缓存时出错: {str(e)}')
        return jsonify({'success': False, 'message': str(e)})

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
    
    app.logger.info('启动前准备工作完成')

# 添加图表文件访问路由
@app.route('/plots/<path:filename>')
def serve_plots(filename):
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), PLOTS_DIR)
    return send_from_directory(plot_dir, filename)

if __name__ == '__main__':
    # 启动前准备
    startup_preparation()
    
    # 自动打开浏览器
    app.logger.info('尝试自动打开浏览器...')
    try:
        # 使用子线程在服务器启动后打开浏览器，避免阻塞
        import threading
        def open_browser():
            # 等待服务器启动
            time.sleep(2)
            url = 'http://localhost:5000/'
            app.logger.info(f'正在打开浏览器访问: {url}')
            webbrowser.open(url)
        
        # 启动子线程
        threading.Thread(target=open_browser, daemon=True).start()
    except Exception as e:
        app.logger.error(f'自动打开浏览器失败: {str(e)}')
        app.logger.info('请手动在浏览器中访问 http://localhost:5000/')
    
    # 运行Flask应用
    app.logger.info('启动股票预测系统Web服务...')
    app.run(host='0.0.0.0', port=5000, debug=False)