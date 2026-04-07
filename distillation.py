import torch
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler  # 🌟 换成绝对不会报错的随机克隆魔法！

class KD_XGBoost_Engine:
    def __init__(self, teacher_model, train_loader, val_loader, device='cuda', num_classes=5):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.teacher_model = teacher_model.to(self.device)
        self.teacher_model.eval() 
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        
        self.student_model = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=5, 
            learning_rate=0.1,
            tree_method='hist',
            objective='multi:softprob',
            num_class=self.num_classes,
            eval_metric='mlogloss'
        )

    def _extract_knowledge(self, dataloader, desc_text="提取知识"):
        X_flattened = []
        y_teacher_preds = []
        y_true_labels = []

        # 🌟 进度条保护
        pbar = tqdm(dataloader, desc=f"  {desc_text}", dynamic_ncols=True)
        
        with torch.no_grad():
            for batch_x, batch_mask, batch_y in pbar:
                batch_x = batch_x.to(self.device)
                logits = self.teacher_model(batch_x)
                _, preds_tensor = torch.max(logits, 1)
                
                X_flattened.append(batch_x.view(batch_x.size(0), -1).cpu().numpy())
                y_teacher_preds.append(preds_tensor.cpu().numpy())
                y_true_labels.append(batch_y.cpu().numpy())

        return np.vstack(X_flattened), np.concatenate(y_teacher_preds), np.concatenate(y_true_labels)

    def train_student(self):
        X_train, y_train_teacher, _ = self._extract_knowledge(self.train_loader, "提取训练集特征")
        X_val, y_val_teacher, y_val_true = self._extract_knowledge(self.val_loader, "提取验证集特征")

        # 🌟 XGBoost 防崩溃终极护盾：强行补齐缺失类别
        for c in range(self.num_classes):
            if c not in y_train_teacher:
                X_train = np.vstack([X_train, X_train[0]])
                y_train_teacher = np.append(y_train_teacher, c)

        # 清理异常值
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)

        print(f"  提取完毕！原始特征维度: {X_train.shape}")
        
        # ==========================================
        # 🪄 核心魔法：RandomOverSampler 随机克隆，绝对不会报错！
        # ==========================================
        print("  [魔法开启] 正在使用克隆技术动态扩充少数类 (PortScan, BruteForce, FDIA)...")
        ros = RandomOverSampler(random_state=42)
        X_train, y_train_teacher = ros.fit_resample(X_train, y_train_teacher)
        print(f"  [魔法完成] 扩充后特征维度暴增至: {X_train.shape}，各类别数量已绝对平衡！")

        print("  正在计算多分类动态惩罚权重...")
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_teacher)
        
        print("  正在用 Transformer 的多分类经验训练 XGBoost 学生模型...")
        self.student_model.fit(
            X_train, 
            y_train_teacher, 
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val_true)],
            verbose=False
        )
        print("  ✅ XGBoost 多分类跨界知识蒸馏完成！")

    def evaluate_student(self):
        _, _, y_val_true = self._extract_knowledge(self.val_loader, "最终评估提取")