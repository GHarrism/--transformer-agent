import torch
import os
import glob
from dataset import get_dataloaders
from model import LightweightGridTransformer
from trainer import GridTrainer
from distillation import KD_XGBoost_Engine

def main():
    print("="*60)
    print("🚀 智能电网入侵检测系统 (5分类终极版) 离线训练流水线")
    print("="*60)

    input_dim = 205
    num_classes = 5

    # 阶段 1: 数据加载
    print("\n>>> 阶段 1: 连接 HDF5 工业级数据库")
    train_loader, val_loader = get_dataloaders('data/full_fused_data.h5', batch_size=1024)

    # 初始化教师模型
    transformer_model = LightweightGridTransformer(input_dim=input_dim, num_classes=num_classes)

    # 阶段 2: 跳过训练，直接读档！
    print("\n>>> 阶段 2: 训练 Transformer 教师模型 (Focal Loss 护航)")
    print("⏩ 触发极速模式：跳过长达1小时的训练，直接读取已保存的 Transformer 大脑...")
    
    # 自动寻找 saved_models 目录下刚生成的 .pth 文件并读取
    pth_files = glob.glob('saved_models/*.pth')
    if not pth_files:
        print("❌ 错误：在 saved_models 目录下没有找到 .pth 模型文件！请检查路径。")
        return
        
    model_file = pth_files[-1] # 取最新保存的权重
    
    # 🌟 这里修复了你截图里的变量名 bug，统一使用 transformer_model
    transformer_model.load_state_dict(torch.load(model_file))
    transformer_model.eval()
    print(f"✅ 成功加载权重: {model_file}")

    # 阶段 3: 知识蒸馏 (加入 SMOTE 魔法)
    print("\n>>> 阶段 3: 跨界知识蒸馏给边缘 XGBoost")
    distiller = KD_XGBoost_Engine(transformer_model, train_loader, val_loader, num_classes=num_classes)
    distiller.train_student()
    
    # 打印超级华丽的 5 分类验证报告
    xgb_student = distiller.evaluate_student()

    # 阶段 4: 保存轻量化模型
    print("\n>>> 阶段 4: 导出终极模型权重")
    os.makedirs('saved_models', exist_ok=True)
    xgb_student.save_model('saved_models/xgb_student_5class.json')
    print("✅ 全链路 5 分类离线训练完成！边缘多分类模型 (.json) 已保存，随时准备供大屏前端调用！")

if __name__ == "__main__":
    main()