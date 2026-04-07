import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 使用 LaTeX 公式对应的正弦/余弦位置编码实现
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # 注册为 buffer，不会作为模型参数被更新

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # 注意对齐 batch_first 维度
        x = x + self.pe[:x.size(1), 0, :].unsqueeze(0) 
        return x

class LightweightGridTransformer(nn.Module):
    def __init__(self, input_dim, num_classes=3, d_model=64, nhead=2, num_layers=2):
        super().__init__()
        # 1. 输入特征层 [cite: 50]：特征降维与嵌入
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. 编码器层 [cite: 50]：低头数、少层数实现轻量化 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,        # 降低注意力头数 (如改为 2)
            dim_feedforward=128,# 压缩前馈网络维度
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 输出分类层 [cite: 51]
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes) # 比如 0:正常, 1:DDoS, 2:FDIA
        )
        
    def forward(self, x, valid_mask):
        # x 形状: [batch, seq_len, input_dim]
        x_emb = self.embedding(x)
        x_emb = self.pos_encoder(x_emb) # 注入时序位置信息
        
        # 注意：valid_mask 取反 (True代表忽略该位置)
        src_key_padding_mask = ~valid_mask 
        
        out = self.transformer_encoder(x_emb, src_key_padding_mask=src_key_padding_mask)
        
        # 取时间序列的最后一个时间步的特征进行分类
        final_feature = out[:, -1, :] 
        logits = self.classifier(final_feature)
        
        return logits