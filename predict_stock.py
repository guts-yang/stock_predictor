# 股票价格预测模块
import torch
import os
from stock_data import get_device, get_latest_stock_data
from stock_model import get_model
import pandas as pd
import matplotlib.pyplot as plt

# 导入配置文件
from config import DEFAULT_MODEL_TYPE, MODELS_DIR, RESULTS_DIR

# 设置中文字体显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 加载训练好的模型
def load_trained_model(model_path):
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件{model_path}不存在")
        return None, None
    
    # 加载模型参数
    checkpoint = torch.load(model_path, map_location=get_device())
    
    # 提取模型配置参数
    model_type = checkpoint['model_type']
    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    output_size = checkpoint['output_size']
    num_layers = checkpoint['num_layers']
    sequence_length = checkpoint.get('sequence_length', 5)
    
    # 创建模型实例
    model = get_model(model_type, input_size, hidden_size, output_size, num_layers)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, sequence_length

# 使用模型进行预测
def predict_stock_price(model, stock_code, device, sequence_length=5):
    # 获取最新的股票数据
    latest_sequence, latest_close = get_latest_stock_data(stock_code, sequence_length)
    
    if latest_sequence is None:
        print(f"无法获取股票{stock_code}的最新数据")
        return None
    
    # 将数据移至设备
    latest_sequence = latest_sequence.to(device)
    
    # 设置模型为评估模式
    model.eval()
    
    with torch.no_grad():
        # 进行预测
        prediction = model(latest_sequence)
        predicted_price = prediction.item()
    
    # 计算预测变化百分比
    price_change_percent = ((predicted_price - latest_close) / latest_close) * 100
    
    # 恢复模型为训练模式
    model.train()
    
    return predicted_price, latest_close, price_change_percent

# 批量预测多只股票
def batch_predict_stocks(model_path, stock_codes, device):
    # 加载模型
    model, sequence_length = load_trained_model(model_path)
    if model is None:
        return []
    
    # 将模型移至设备
    model.to(device)
    
    # 存储预测结果
    results = []
    
    # 遍历所有股票代码进行预测
    for stock_code in stock_codes:
        print(f"正在预测股票: {stock_code}")
        result = predict_stock_price(model, stock_code, device, sequence_length)
        
        if result is not None:
            predicted_price, latest_close, price_change_percent = result
            results.append({
                '股票代码': stock_code,
                '最新收盘价': latest_close,
                '预测收盘价': predicted_price,
                '预测涨跌幅(%)': price_change_percent
            })
    
    return results

# 绘制预测结果与实际价格对比图
def plot_prediction_result(stock_code, latest_close, predicted_price, save_path=None):
    from stock_data import get_stock_name
    
    plt.figure(figsize=(8, 5))
    
    # 获取股票名称
    stock_name = get_stock_name(stock_code)
    
    # 绘制实际价格和预测价格
    plt.bar(['最新收盘价', '预测收盘价'], [latest_close, predicted_price], color=['blue', 'orange'])
    
    # 添加数值标签
    for i, v in enumerate([latest_close, predicted_price]):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # 设置图表属性
    plt.title(f'股票{stock_name}({stock_code})价格预测')
    plt.ylabel('价格')
    plt.grid(True, axis='y')
    
    # 计算并显示预测变化
    change_percent = ((predicted_price - latest_close) / latest_close) * 100
    change_text = f'预测{'上涨' if change_percent > 0 else '下跌'}: {abs(change_percent):.2f}%'
    plt.figtext(0.5, 0.01, change_text, ha='center', fontsize=12)
    
    if save_path:
        plt.savefig(save_path)
        print(f"预测结果图已保存到: {save_path}")
    
    plt.close()

# 获取默认模型路径
def get_default_model_path(stock_code, model_type=DEFAULT_MODEL_TYPE):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, MODELS_DIR)
    model_path = os.path.join(model_dir, f'{stock_code}_{model_type}_best.pth')
    return model_path

# 主预测函数
def main_predict(stock_code, model_path=None, model_type=DEFAULT_MODEL_TYPE):
    # 获取设备
    device = get_device()
    print(f'使用设备: {device}')
    
    # 如果没有提供模型路径，使用默认路径
    if model_path is None:
        model_path = get_default_model_path(stock_code, model_type)
    
    # 加载模型
    model, sequence_length = load_trained_model(model_path)
    if model is None:
        print("无法加载模型，预测失败")
        return False
    
    # 将模型移至设备
    model.to(device)
    
    # 进行预测
    result = predict_stock_price(model, stock_code, device, sequence_length)
    
    if result is not None:
        predicted_price, latest_close, price_change_percent = result
        
        # 打印预测结果
        print(f"股票代码: {stock_code}")
        print(f"最新收盘价: {latest_close:.2f}")
        print(f"预测收盘价: {predicted_price:.2f}")
        print(f"预测{'上涨' if price_change_percent > 0 else '下跌'}: {abs(price_change_percent):.2f}%")
        
        # 创建图像保存目录
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # 绘制预测结果图
        plot_path = os.path.join(plot_dir, f'{stock_code}_prediction_result.png')
        plot_prediction_result(stock_code, latest_close, predicted_price, plot_path)
        
        return True
    
    return False

# 主函数，方便直接运行预测
def main():
    # 示例参数
    stock_code = '000001.SZ'  # 平安银行
    
    # 进行预测
    success = main_predict(stock_code)
    
    if success:
        print("预测完成！")
    else:
        print("预测失败！")

if __name__ == "__main__":
    main()