import numpy as np
import torch
# 引入 LangChain 预留大模型交互接口
from langchain_core.prompts import PromptTemplate

class DetectionAgent:
    """
    前端斥候：部署于本地（边缘计算节点）
    行为逻辑：运行轻量级模型 (XGBoost) 进行毫秒级快速筛查
    """
    def __init__(self, xgb_student_model):
        self.model = xgb_student_model

    def monitor(self, fused_features_2d):
        print("\n[Detection Agent 🟢] 正在边缘端进行实时流量与物理特征筛查...")
        # XGBoost 极速推理
        pred = self.model.predict(fused_features_2d)
        
        # 假设 0 是正常，非 0 是攻击
        if pred[0] != 0: 
            print(f"⚠️ [Detection Agent] 警报 (Alert)：发现疑似异常！攻击代码 [{pred[0]}]，立即上报云端分析师！")
            return True, pred[0]
        return False, 0

class AnalysisAgent:
    """
    云端军师：部署于算力中心
    行为逻辑：接收前端告警后，调用 Transformer 深度分析，并利用 LangChain 生成研判报告
    """
    def __init__(self, transformer_teacher_model, device='cpu'):
        self.model = transformer_teacher_model
        self.device = device
        
        # 利用 LangChain 的 PromptTemplate 实现大语言模型接口预留
        self.report_prompt = PromptTemplate(
            input_variables=["threat_type", "confidence", "details"],
            template="【智能电网安全态势研判报告】\n- 威胁类型: {threat_type}\n- 模型置信度: {confidence}\n- 攻击特征细节: {details}\n- 决策建议: 请立即执行防御预案。"
        )

    def analyze(self, raw_seq_data, valid_mask):
        print("[Analysis Agent 🔵] 接收到告警，启动 Transformer 提取高维全局特征进行深度决策 (Decision)...")
        self.model.eval()
        with torch.no_grad():
            # Transformer 进行深度前向传播
            logits = self.model(raw_seq_data.to(self.device), valid_mask.to(self.device))
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            confidence, predicted_class = torch.max(probs, 0)
        
        # 威胁映射字典
        threat_mapping = {1: "DDoS 网络洪水攻击", 2: "FDIA 虚假数据注入攻击"}
        threat_type = threat_mapping.get(predicted_class.item(), "未知高危异常")
        
        # 使用 LangChain 模板格式化报告
        report = self.report_prompt.format(
            threat_type=threat_type,
            confidence=f"{confidence.item()*100:.2f}%",
            details="时序窗口内出现明显的流量激增与 SCADA 物理状态偏离"
        )
        print(f"📄 [Analysis Agent] 报告生成完毕：\n{report}")
        return report, threat_type

class ResponseAgent:
    """
    执行单元：连接物理设备与网络防火墙
    行为逻辑：根据研判报告，下发具体的缓解策略和响应动作
    """
    def __init__(self):
        # 这里可以预留与电网真实物理设备（如断路器、路由器）的控制接口
        pass

    def execute(self, report, threat_type):
        print(f"🛡️ [Response Agent 🔴] 收到分析指令，开始执行行动 (Action)...")
        if "DDoS" in threat_type:
            print(" >> 动作 1: 在边界路由器下发 ACL 防火墙规则，封堵恶意源 IP。")
            print(" >> 动作 2: 将网络业务无缝切换至备用通信链路。")
        elif "FDIA" in threat_type:
            print(" >> 动作 1: 逻辑隔离受影响的 SCADA 传感器节点，丢弃其测量值。")
            print(" >> 动作 2: 激活系统状态估计的鲁棒补偿算法。")
        else:
            print(" >> 动作 1: 持续高频监测，将异常特征录入本地威胁情报库。")
        print("[Response Agent] 应急响应执行完毕，系统风险已解除！\n")

# --- 测试整个协作闭环的函数 ---
def run_agent_workflow(xgb_model, transformer_model, sample_2d, sample_3d_seq, sample_mask):
    """模拟系统运行时的完整工作流"""
    detector = DetectionAgent(xgb_model)
    analyst = AnalysisAgent(transformer_model)
    responder = ResponseAgent()

    # 1. 前端监测 (Alert)
    is_threat, _ = detector.monitor(sample_2d)
    
    if is_threat:
        # 2. 云端决策 (Decision)
        report, threat_type = analyst.analyze(sample_3d_seq, sample_mask)
        # 3. 终端响应 (Action)
        responder.execute(report, threat_type)
    else:
        print("[系统状态] 正常运行，各模态数据流平稳如水...")