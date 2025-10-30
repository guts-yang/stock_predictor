# 股票预测模型训练模块
import torch
from torch import nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from stock_data import load_stock_data, get_device
from stock_model import get_model
import pandas as pd

# 导入配置文件
from config import DEFAULT_MODEL_TYPE, DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_LAYERS, DEFAULT_SEQUENCE_LENGTH, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, DEFAULT_PATIENCE, MODELS_DIR, TEST_SIZE, RANDOM_STATE, USE_GPU, EARLY_STOPPING_ENABLED, EARLY_STOPPING_MIN_DELTA, LR_SCHEDULER_ENABLED, LR_SCHEDULER_STEP_SIZE, LR_SCHEDULER_GAMMA, AUTO_SAVE_MODEL, AUTO_SAVE_INTERVAL
from stock_data import get_stock_name

# 设置中文字体显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 评估模型性能
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    actual_values = []
    predicted_values = []
    
    with torch.no_grad():
        for batch in data_loader:
            sequences = batch['sequence'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()
            
            # 保存实际值和预测值用于后续分析
            actual_values.extend(targets.cpu().numpy())
            predicted_values.extend(outputs.squeeze().cpu().numpy())
    
    model.train()
    return total_loss / len(data_loader), actual_values, predicted_values

# 绘制训练和测试损失曲线
def plot_losses(train_losses, test_losses, stock_code, save_path=None):
    # 获取股票名称
    stock_name = get_stock_name(stock_code)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.title(f'股票{stock_name}({stock_code})训练和测试损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"损失曲线已保存到: {save_path}")
    
    plt.close()

# 绘制实际值和预测值的对比图
def plot_predictions(actual_values, predicted_values, stock_code, save_path=None):
    # 获取股票名称
    stock_name = get_stock_name(stock_code)
    
    plt.figure(figsize=(10, 6))
    plt.plot(actual_values, label='实际股价')
    plt.plot(predicted_values, label='预测股价')
    plt.title(f'股票{stock_name}({stock_code})实际股价与预测股价对比')
    plt.xlabel('样本')
    plt.ylabel('股价')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"预测对比图已保存到: {save_path}")
    
    plt.close()

# 训练股票预测模型
def train_stock_model(stock_code, start_date, end_date, 
                     model_type=DEFAULT_MODEL_TYPE, 
                     hidden_size=DEFAULT_HIDDEN_SIZE, 
                     num_layers=DEFAULT_NUM_LAYERS, 
                     sequence_length=DEFAULT_SEQUENCE_LENGTH, 
                     batch_size=DEFAULT_BATCH_SIZE, 
                     learning_rate=DEFAULT_LEARNING_RATE, 
                     epochs=DEFAULT_EPOCHS, 
                     patience=DEFAULT_PATIENCE, 
                     device=None):
    # 如果没有提供设备，则自动获取
    if device is None:
        device = get_device()
    print(f'使用设备: {device}')
    
    # 加载股票数据
    train_loader, test_loader, stock_dataset = load_stock_data(
        stock_code, start_date, end_date, sequence_length, batch_size
    )
    
    # 获取输入特征数量
    input_size = len(stock_dataset.X.columns)
    output_size = 1
    
    # 创建模型
    model = get_model(model_type, input_size, hidden_size, output_size, num_layers)
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失函数，适合回归问题
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化器
    
    # 学习率调度器，当损失不再改善时降低学习率
    scheduler = None
    if LR_SCHEDULER_ENABLED:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_SCHEDULER_STEP_SIZE, gamma=LR_SCHEDULER_GAMMA)
    
    # 记录训练和测试损失
    train_losses = []
    test_losses = []
    
    # 最佳损失和早停计数器
    best_loss = float('inf')
    early_stop_counter = 0
    
    # 创建模型保存目录
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODELS_DIR)
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建图像保存目录
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 开始训练
    model.train()
    for epoch in range(epochs):
        total_train_loss = 0
        
        for batch in train_loader:
            sequences = batch['sequence'].to(device)
            targets = batch['target'].to(device)
            
            # 前向传播
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 评估模型在测试集上的性能
        avg_test_loss, actual_values, predicted_values = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(avg_test_loss)
        
        # 更新学习率
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_test_loss)
            else:
                scheduler.step()
        
        # 打印训练进度
        print(f'Epoch [{epoch+1}/{epochs}], 训练损失: {avg_train_loss:.4f}, 测试损失: {avg_test_loss:.4f}')
        
        # 早停机制
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            early_stop_counter = 0
            
            # 保存最佳模型
            model_path = os.path.join(model_dir, f'{stock_code}_{model_type}_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'stock_code': stock_code,
                'model_type': model_type,
                'input_size': input_size,
                'hidden_size': hidden_size,
                'output_size': output_size,
                'num_layers': num_layers,
                'sequence_length': sequence_length
            }, model_path)
            print(f"最佳模型已保存到: {model_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"早停机制触发，在第{epoch+1}轮停止训练")
                break
    
    # 绘制训练和测试损失曲线
    loss_plot_path = os.path.join(plot_dir, f'{stock_code}_{model_type}_loss.png')
    plot_losses(train_losses, test_losses, stock_code, loss_plot_path)
    
    # 绘制预测对比图
    pred_plot_path = os.path.join(plot_dir, f'{stock_code}_{model_type}_prediction.png')
    plot_predictions(actual_values, predicted_values, stock_code, pred_plot_path)
    
    print(f"训练完成！最佳测试损失: {best_loss:.4f}")
    
    return model_path

# 主函数，方便直接运行训练
def main():
    # 示例参数
    stock_code = '000001.SZ'  # 平安银行
    start_date = '20200101'
    end_date = '20230101'
    
    # 训练模型
    model_path = train_stock_model(
        stock_code=stock_code,
        start_date=start_date,
        end_date=end_date,
        model_type='lstm',
        hidden_size=64,
        num_layers=2,
        sequence_length=5,
        batch_size=32,
        learning_rate=0.001,
        epochs=100,
        patience=10
    )
    
    print(f"模型训练完成并保存到: {model_path}")

if __name__ == "__main__":
    main()