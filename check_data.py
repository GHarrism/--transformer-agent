import h5py
import numpy as np
from collections import Counter

def data_health_check():
    print(">>> 启动 HDF5 数据体检...")
    try:
        with h5py.File('data/full_fused_data.h5', 'r') as f:
            # 1. 检查标签分布
            labels = f['labels'][:]
            print("\n📊 真实标签分布 (Label Distribution):")
            counts = Counter(labels)
            for k, v in counts.items():
                print(f"  - 类别 {k}: {v} 条")
                
            # 2. 检查数值爆炸
            features = f['features'][:]
            print("\n☢️ 特征健康状态 (Feature Health):")
            has_nan = np.isnan(features).any()
            has_inf = np.isinf(features).any()
            print(f"  - 包含空值 (NaN): {has_nan}")
            print(f"  - 包含无穷大 (Inf): {has_inf}")
            
    except Exception as e:
        print(f"❌ 读取错误: {e}")

if __name__ == "__main__":
    data_health_check()