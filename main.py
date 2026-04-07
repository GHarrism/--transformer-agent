import os
import torch
from dataset import get_dataloaders
from model import LightweightGridTransformer
from trainer import GridTrainer
from distillation import KD_XGBoost_Engine

def main():
    print("\n" + "="*60)
    print("🚀 智能电网入侵检测系统 (5分类终极版) 离线训练流水线")
    print("="*60)

    # ==========================================
    # 阶段 1：加载 HDF5 懒加载数据底座
    # ==========================================
    print("\n>>> 阶段 1: 连接 HDF5 工业级数据库")
    try:
        train_loader, val_loader = get_dataloaders('data/full_fused_data.h5', batch_size=1024)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("请确保你已经成功运行了 prepare_data.py！")
        return

    # ==========================================
    # 阶段 2：训练 Transformer 教师大模型
    # ==========================================
    print("\n>>> 阶段 2: 训练 Transformer 教师模型 (Focal Loss 护航)")
    input_dim = 205
    num_classes = 5 # 升级为 5 分类

    transformer_model = LightweightGridTransformer(input_dim=input_dim, num_classes=num_classes)
    trainer = GridTrainer(transformer_model, train_loader, val_loader, learning_rate=0.001)

    #数据量庞大，2轮即可学习
    trainer.fit(epochs=2)

    # ==========================================
    # 阶段 3：跨界知识蒸馏给边缘 XGBoost
    # ==========================================
    print("\n>>> 阶段 3: 跨界知识蒸馏 (Transformer -> XGBoost)")
    distiller = KD_XGBoost_Engine(transformer_model, train_loader, val_loader, num_classes=num_classes)
    distiller.train_student()
    
    # 打印超级华丽的 5 分类验证报告
    xgb_student = distiller.evaluate_student()

    # ==========================================
    # 阶段 4：保存终极模型权重
    # ==========================================
    print("\n>>> 阶段 4: 导出终极模型权重")
    os.makedirs('saved_models', exist_ok=True)
    
    # 使用新名字保存，避免覆盖中期的旧模型
    torch.save(transformer_model.state_dict(), 'saved_models/transformer_teacher_5class.pth')
    xgb_student.save_model('saved_models/xgb_student_5class.json')

    print("✅ 全链路 5 分类离线训练完成！多分类模型已保存在 saved_models/ 目录下。")

if __name__ == "__main__":
    main()