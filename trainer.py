import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class GridTrainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=0.001, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = FocalLoss(gamma=2.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, epochs=10):
        print(f"  [引擎状态] 启动 5 分类核心训练循环，分配计算设备: {self.device}...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            # 🌟 进度条保护
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}", dynamic_ncols=True)
            
            for batch_x, batch_mask, batch_y in pbar:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                
            train_acc = correct / total
            print(f"  ✅ Epoch [{epoch+1}/{epochs}] 完成 | Train Focal Loss: {total_loss/len(self.train_loader):.4f} | Train Acc: {train_acc:.4f}")