# 股票预测LSTM模型定义
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# 高级K线预测LSTM模型，支持多特征预测（开盘价、收盘价、最高价、最低价、成交量）
class AdvancedKLineLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=DEFAULT_HIDDEN_SIZE, output_size=5, num_layers=DEFAULT_NUM_LAYERS):
        super(AdvancedKLineLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size  # 5个输出：开盘价、收盘价、最高价、最低价、成交量
        
        # 多层LSTM网络
        self.lstm_layers = nn.ModuleList()
        
        # 第一层LSTM
        self.lstm_layers.append(nn.LSTM(input_size, hidden_size, 1, batch_first=True, dropout=0.1))
        
        # 添加更多LSTM层
        for _ in range(num_layers - 1):
            self.lstm_layers.append(nn.LSTM(hidden_size, hidden_size, 1, batch_first=True, dropout=0.1))
        
        # 注意力机制层
        self.attention = nn.Linear(hidden_size, 1)
        
        # 预测层，输出5个特征
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.bn1 = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        # 获取批次大小和序列长度
        batch_size, seq_len, _ = x.size()
        
        # 初始化隐藏状态和细胞状态
        h_prev = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c_prev = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        
        # 存储每一层的输出
        layer_outputs = []
        current_input = x
        
        # 多层LSTM处理
        for lstm in self.lstm_layers:
            lstm_output, (h_prev, c_prev) = lstm(current_input, (h_prev, c_prev))
            layer_outputs.append(lstm_output)
            current_input = lstm_output
        
        # 使用注意力机制
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        
        # 全连接层处理
        out = F.relu(self.bn1(self.fc1(context_vector)))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# 获取模型函数，支持不同模型类型
def get_model(model_type, input_size, hidden_size=DEFAULT_HIDDEN_SIZE, output_size=1, num_layers=DEFAULT_NUM_LAYERS):
    if model_type == 'kline_lstm':
        # K线预测模型，默认输出5个特征
        return AdvancedKLineLSTMModel(input_size, hidden_size, 5, num_layers)
    else:
        # 默认返回基线LSTM模型
        return BaselineLSTMModel(input_size, hidden_size, output_size, num_layers)