# 股票预测LSTM模型定义
import torch
import torch.nn as nn

# 导入配置文件
from config import DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_LAYERS

# 简化的基线LSTM模型
class BaselineLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=DEFAULT_HIDDEN_SIZE, output_size=1, num_layers=DEFAULT_NUM_LAYERS):
        super(BaselineLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 定义单个全连接层
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 获取批次大小
        batch_size = x.size(0)
        
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM层前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 通过全连接层
        out = self.fc(out)
        
        return out

# 获取模型函数，仅返回基线LSTM模型
def get_model(model_type, input_size, hidden_size=DEFAULT_HIDDEN_SIZE, output_size=1, num_layers=DEFAULT_NUM_LAYERS):
    # 忽略model_type参数，始终返回基线LSTM模型
    return BaselineLSTMModel(input_size, hidden_size, output_size, num_layers)