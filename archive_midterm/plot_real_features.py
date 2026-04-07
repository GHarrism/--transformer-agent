import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 设置中文字体与学术风格
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='SimHei')

def plot_real_feature_importance():
    print(">>> 阶段 1：穿透数据集，抓取 205 个真实物理/网络特征名称...")
    # 只读取极少部分数据（10行），光速获取最原始的列名
    cic_path = 'data/cic_ids2017/Botnet-Friday-no-metadata.parquet'
    scada_path = 'data/power_system/data1.csv'

    try:
        cic_df = pd.read_parquet(cic_path).head(10)
        scada_df = pd.read_csv(scada_path).head(10)
    except Exception as e:
        print(f"❌ 读取数据失败，请确保路径正确: {e}")
        return

    # 清洗列名并剔除标签（与 dataset.py 逻辑完全同步）
    cic_df.columns = cic_df.columns.str.strip()
    scada_df.columns = scada_df.columns.str.strip()
    if 'Label' in cic_df.columns: cic_df.drop(columns=['Label'], inplace=True)
    if 'marker' in scada_df.columns: scada_df.drop(columns=['marker'], inplace=True)
    cic_df = cic_df.select_dtypes(include=[np.number])
    scada_df = scada_df.select_dtypes(include=[np.number])

    # 模拟合并以获取最终的 205 个确切特征名
    merged_empty = pd.merge(
        pd.DataFrame(columns=cic_df.columns),
        pd.DataFrame(columns=scada_df.columns),
        left_index=True, right_index=True, how='outer', suffixes=('_net', '_phy')
    )
    original_feature_names = list(merged_empty.columns)
    num_features = len(original_feature_names) 

    print(f"✅ 成功提取 {num_features} 个真实特征名称！\n")

    print(">>> 阶段 2：加载 XGBoost，执行【时间步折叠与权重累加】算法...")
    model = xgb.XGBClassifier()
    model.load_model('saved_models/xgb_student.json')
    importances = model.feature_importances_ # 拿到 2050 维权重

    # 🌟 核心黑科技：将 2050 维的展平时序权重，按时间步取模，累加回 205 维的全局权重！
    aggregated_importances = np.zeros(num_features)
    for i, weight in enumerate(importances):
        feature_idx = i % num_features # 取模还原原始特征的真实位置
        aggregated_importances[feature_idx] += weight

    # 找出累加后全局排名前 10 的真实特征
    top_10_indices = np.argsort(aggregated_importances)[-10:][::-1]
    top_10_weights = aggregated_importances[top_10_indices]
    top_10_names = [original_feature_names[i] for i in top_10_indices]

    print("🏆 全局排名前 10 的真实跨域特征 (网络+物理)：")
    for name, weight in zip(top_10_names, top_10_weights):
        print(f" - {name}: {weight:.4f}")

    print("\n>>> 阶段 3：绘制真实学术图表...")
    plt.figure(figsize=(10, 6), dpi=150)
    
    # 将英文特征名缩短，防止图表太挤 (可选优化)
    short_names = [name[:30] + '...' if len(name)>30 else name for name in top_10_names]
    
    sns.barplot(x=top_10_weights, y=short_names, palette='magma')

    plt.xlabel('全局累加特征重要性权重 (Aggregated Gini Importance)', fontsize=12)
    plt.ylabel('多源跨域真实特征 (原始数据集列名)', fontsize=12)
    plt.title('真实模型全局特征重要性分析 (Top 10)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('Chart3_True_Global_Importance.png')
    print("✅ 完美！绝对真实的权重图表已生成: Chart3_True_Global_Importance.png")

if __name__ == "__main__":
    plot_real_feature_importance()