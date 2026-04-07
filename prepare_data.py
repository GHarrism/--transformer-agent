import pandas as pd
import numpy as np
import h5py
import os
import glob

# ==========================================
# 全局配置
# ==========================================
CIC_DIR = 'data/cic_ids2017'
SCADA_DIR = 'data/power_system'
OUTPUT_FILE = 'data/full_fused_data.h5'
WINDOW_SIZE = 10
STRIDE = 10  # 🌟 10倍降采样步长

# 🌟 5分类硬核标签字典
LABEL_MAPPING = {
    'BENIGN': 0,               
    'DDoS': 1, 'DoS': 1, 'Botnet': 1, 'Infiltration': 1,
    'PortScan': 2,             
    'FTP-Patator': 3, 'SSH-Patator': 3, 'BruteForce': 3, 'Web Attack': 3,
    'FDIA': 4                  
}

def clean_and_map_labels(df_net, df_phy):
    # 1. 提取网络标签
    net_labels = np.zeros(len(df_net), dtype=int)
    if 'Label' in df_net.columns:
        for key, val in LABEL_MAPPING.items():
            net_labels[df_net['Label'].astype(str).str.contains(key, case=False, na=False)] = val
        df_net = df_net.drop(columns=['Label'])
    
    # 2. 提取并注入物理标签 (FDIA)
    phy_labels = np.zeros(len(df_phy), dtype=int)
    marker_col = 'marker' if 'marker' in df_phy.columns else ('Marker' if 'Marker' in df_phy.columns else None)
    if marker_col:
        phy_labels[df_phy[marker_col] == 1] = LABEL_MAPPING['FDIA']
        df_phy = df_phy.drop(columns=[marker_col])
        
    # 🌟 人工注入 2% 的 FDIA 物理攻击，确保类别 4 不缺席！
    if np.sum(phy_labels == LABEL_MAPPING['FDIA']) == 0:
        fdia_indices = np.random.choice(len(df_phy), size=int(len(df_phy) * 0.02), replace=False)
        phy_labels[fdia_indices] = LABEL_MAPPING['FDIA']
        df_phy.iloc[fdia_indices] = df_phy.iloc[fdia_indices] * 1.5

    # 3. 标签融合
    final_labels = np.where(phy_labels == 4, 4, net_labels)
    
    # 4. 填补清洗
    df_net = df_net.select_dtypes(include=[np.number]).fillna(0)
    df_phy = df_phy.select_dtypes(include=[np.number]).fillna(0)
    
    return df_net, df_phy, final_labels

def build_offline_dataset():
    print("🚀 开始构建全量离线 HDF5 数据库 (10倍降采样 + FDIA注入版)...")
    
    cic_files = sorted(glob.glob(os.path.join(CIC_DIR, '*.parquet')))
    scada_files = sorted(glob.glob(os.path.join(SCADA_DIR, '*.csv')))
    
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    with h5py.File(OUTPUT_FILE, 'w') as h5f:
        X_dataset, y_dataset = None, None
        current_idx = 0
        
        for i, (cic_path, scada_path) in enumerate(zip(cic_files, scada_files)):
            print(f"\n📂 正在处理数据块 {i+1}...")
            
            df_net = pd.read_parquet(cic_path)
            df_phy = pd.read_csv(scada_path)
            
            # 循环补齐，不截断网络攻击
            len_net = len(df_net)
            len_phy = len(df_phy)
            if len_phy < len_net:
                repeats = (len_net // len_phy) + 1
                df_phy = pd.concat([df_phy] * repeats, ignore_index=True).iloc[:len_net]
            else:
                df_phy = df_phy.iloc[:len_net]
                
            df_net, df_phy, labels = clean_and_map_labels(df_net, df_phy)
            
            fused_features = np.hstack([df_net.values, df_phy.values])
            
            # 🌟 强力抹杀 Inf 和 NaN
            fused_features = np.nan_to_num(fused_features, nan=0.0, posinf=1e5, neginf=-1e5)
            fused_features = np.clip(fused_features, a_min=-1e5, a_max=1e5) / 1000.0
            
            feature_dim = fused_features.shape[1]
            valid_starts = list(range(0, len(fused_features) - WINDOW_SIZE + 1, STRIDE))
            num_windows = len(valid_starts)
            
            if num_windows <= 0: continue
                
            X_chunk = np.zeros((num_windows, WINDOW_SIZE, feature_dim), dtype=np.float32)
            y_chunk = np.zeros(num_windows, dtype=np.int8)
            
            for idx, j in enumerate(valid_starts):
                X_chunk[idx] = fused_features[j : j + WINDOW_SIZE]
                y_chunk[idx] = labels[j + WINDOW_SIZE - 1]
            
            if X_dataset is None:
                X_dataset = h5f.create_dataset('features', data=X_chunk, maxshape=(None, WINDOW_SIZE, feature_dim), chunks=True, compression="gzip")
                y_dataset = h5f.create_dataset('labels', data=y_chunk, maxshape=(None,), chunks=True, compression="gzip")
                current_idx = num_windows
            else:
                X_dataset.resize((current_idx + num_windows, WINDOW_SIZE, feature_dim))
                y_dataset.resize((current_idx + num_windows,))
                X_dataset[current_idx : current_idx + num_windows] = X_chunk
                y_dataset[current_idx : current_idx + num_windows] = y_chunk
                current_idx += num_windows
                
            print(f"  ✅ 成功写入 {num_windows} 个窗口。累计容量: {current_idx}")
            
    print(f"\n🎉 降采样 + 物理攻击注入版重构完成！落盘至: {OUTPUT_FILE}")

if __name__ == "__main__":
    build_offline_dataset()