"""
LSTM 文本生成模型
"""

import torch
import torch.nn as nn


class LSTMGenerator(nn.Module):
    """LSTM 文本生成模型"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(LSTMGenerator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM 层
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):
        # 嵌入
        embeds = self.embedding(x)
        embeds = self.dropout(embeds)
        
        # LSTM
        if hidden is None:
            lstm_out, hidden = self.lstm(embeds)
        else:
            lstm_out, hidden = self.lstm(embeds, hidden)
        
        lstm_out = self.dropout(lstm_out)
        
        # 全连接
        output = self.fc(lstm_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device='cpu'):
        """初始化隐藏状态"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


def create_model(vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, dropout=0.2, device='cpu'):
    """创建模型"""
    model = LSTMGenerator(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数量: {total_params:,}")
    
    return model


if __name__ == '__main__':
    # 测试模型
    model = create_model(vocab_size=100)
    
    # 测试前向传播
    test_input = torch.randint(0, 100, (2, 50))  # batch_size=2, seq_len=50
    output, hidden = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
