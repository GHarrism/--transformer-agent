import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class GridDataFusionProcessor:
    """数据处理引擎：负责将异构数据清洗、对齐并转换为模型所需的特征张量"""
    def __init__(self, cic_path, scada_path, time_window='1S'):
        self.cic_path = cic_path
        self.scada_path = scada_path
        self.time_window = time_window
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()

    def load_and_fuse(self):
        print(f"正在加载异构数据并按 {self.time_window} 窗口对齐...")
        # 1. 加载数据
        cic_df = pd.read_parquet(self.cic_path)
        scada_df = pd.read_csv(self.scada_path)

        # (假设原始数据时间列名为 'Timestamp'，标签列名为 'Label')
        # 如果你的实际列名不同，请在此处修改
        cic_df['Timestamp'] = pd.to_datetime(cic_df['Timestamp'])
        scada_df['Timestamp'] = pd.to_datetime(scada_df['Timestamp'])

        cic_df.set_index('Timestamp', inplace=True)
        scada_df.set_index('Timestamp', inplace=True)

        # 2. 提取并保留标签 (按时间窗口取众数或直接取最后状态)
        # 这里简化处理：将流量标签和SCADA标签合并，实际工程中根据具体攻击类型映射
        if 'Label' in cic_df.columns:
            cic_labels = cic_df['Label'].resample(self.time_window).last()
            cic_df.drop(columns=['Label'], inplace=True)
        else:
            cic_labels = pd.Series('Normal', index=cic_df.index.floor(self.time_window))

        # 3. 核心：时间窗口特征重采样 (均值聚合)
        cic_resampled = cic_df.resample(self.time_window).mean(numeric_only=True)
        scada_resampled = scada_df.resample(self.time_window).mean(numeric_only=True)

        # 4. 数据融合 (外连接保留所有时间步)
        fused_df = pd.merge(
            cic_resampled, scada_resampled, 
            left_index=True, right_index=True, 
            how='outer', suffixes=('_net', '_phy')
        )
        
        # 5. 生成 Mask 并处理缺失值
        valid_mask = ~fused_df.isna().any(axis=1).values 
        fused_df.ffill(inplace=True).fillna(0, inplace=True) # 前向填充

        # 6. 特征标准化 (Min-Max)
        features = self.scaler.fit_transform(fused_df.values)
        
        # 7. 标签编码
        # 实际情况中需要综合 cic_labels 和 scada_labels，此处用 cic_labels 示意
        labels = self.label_encoder.fit_transform(cic_labels.reindex(fused_df.index).fillna('Normal').astype(str))

        print(f"特征张量形状: {features.shape}, 标签数量: {len(labels)}")
        return features, labels, valid_mask


class GridFusionDataset(Dataset):
    """PyTorch Dataset：负责将时序特征打包成滑动窗口格式喂给 Transformer"""
    def __init__(self, features, labels, valid_masks, window_size=10):
        self.window_size = window_size
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.masks = torch.BoolTensor(valid_masks)
        
    def __len__(self):
        return len(self.features) - self.window_size + 1

    def __getitem__(self, idx):
        # 截取长度为 window_size 的特征序列和掩码
        x_seq = self.features[idx : idx + self.window_size]
        mask_seq = self.masks[idx : idx + self.window_size]
        
        # 取窗口最后一个时刻的标签作为当前序列的分类目标
        y_label = self.labels[idx + self.window_size - 1] 
        
        return x_seq, mask_seq, y_label

# --- 本地测试代码 ---
if __name__ == "__main__":
    # 模拟运行测试 (请确保你的路径下有对应文件)
    # processor = GridDataFusionProcessor(
    #     cic_path='data/cic_ids2017/Benign-Monday-no-metadata.parquet',
    #     scada_path='data/power_system/scada_data.csv'
    # )
    # features, labels, valid_masks = processor.load_and_fuse()
    # dataset = GridFusionDataset(features, labels, valid_masks, window_size=10)
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # for x, mask, y in dataloader:
    #     print(f"Batch X: {x.shape}, Batch Mask: {mask.shape}, Batch Y: {y.shape}")
    #     break
    pass