import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import os

class HDF5SmartGridDataset(Dataset):
    """
    工业级 HDF5 懒加载数据集
    不把数据读进内存，只在模型需要时，按索引从硬盘抽水
    """
    def __init__(self, h5_path, indices=None):
        self.h5_path = h5_path
        self.indices = indices
        
        # 仅仅为了获取数据长度，不读取具体数据
        with h5py.File(self.h5_path, 'r') as f:
            if self.indices is None:
                self.length = len(f['labels'])
            else:
                self.length = len(self.indices)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 🌟 核心黑科技：每次抽水时瞬间打开数据库读取一小片，读完就关，绝对不占内存
        with h5py.File(self.h5_path, 'r') as f:
            actual_idx = self.indices[idx] if self.indices is not None else idx
            
            x = f['features'][actual_idx]
            y = f['labels'][actual_idx]
        
        # 转换为 PyTorch 张量
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # 为 Transformer 生成注意力掩码 (这里由于是固定窗口长度，全为 True)
        mask = torch.ones(x.shape[0], dtype=torch.bool)
        
        return x_tensor, mask, y_tensor

def get_dataloaders(h5_path='data/full_fused_data.h5', batch_size=128, test_size=0.2):
    print(">>> 正在初始化 HDF5 工业级数据加载器...")
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"找不到数据库文件 {h5_path}，请先运行 prepare_data.py！")
        
    # 1. 获取总样本数
    with h5py.File(h5_path, 'r') as f:
        total_samples = len(f['labels'])
        print(f"  - 数据库共包含 {total_samples} 条时序片段")
        
    # 2. 划分训练集和验证集索引 (打乱顺序)
    indices = np.arange(total_samples)
    train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=42)
    
    # 3. 实例化懒加载 Dataset
    train_dataset = HDF5SmartGridDataset(h5_path, indices=train_idx)
    val_dataset = HDF5SmartGridDataset(h5_path, indices=val_idx)
    
    # 4. 创建 DataLoader
    # 将这行代码中的 num_workers 设为你的 CPU 核心数（比如 4 或 8）
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"  - 训练集分块数 (Batches): {len(train_loader)}")
    print(f"  - 验证集分块数 (Batches): {len(val_loader)}")
    
    return train_loader, val_loader