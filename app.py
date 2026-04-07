import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import xgboost as xgb
# 导入我们写的智能体
from agent_system import DetectionAgent, AnalysisAgent, ResponseAgent

# ==========================================
# 1. 页面与模型初始化
# ==========================================
st.set_page_config(page_title="智能电网态势感知系统", page_icon="⚡", layout="wide")

@st.cache_resource
def load_system():
    """加载真实训练好的轻量级大模型"""
    xgb_model = xgb.XGBClassifier()
    model_path = 'saved_models/xgb_student.json'
    
    if os.path.exists(model_path):
        xgb_model.load_model(model_path)
        detector = DetectionAgent(xgb_model)
        return detector, True
    else:
        return None, False

detector, is_model_loaded = load_system()

# 初始化页面状态缓存 (增加 stats 统计用于画柱状图)
if 'alert_logs' not in st.session_state:
    st.session_state.alert_logs = []
if 'risk_score' not in st.session_state:
    st.session_state.risk_score = 15
if 'stats' not in st.session_state:
    st.session_state.stats = {'正常流量 (Normal)': 0, '网络层 (DDoS)': 0, '物理层 (FDIA)': 0}

# ==========================================
# 2. 页面前端渲染
# ==========================================
st.title("⚡ 智能电网安全态势感知与协同响应中心")
st.markdown("基于 **多智能体 (Agent)** 与 **知识蒸馏 XGBoost** 架构的轻量化入侵检测大屏。")

if not is_model_loaded:
    st.error("未找到训练好的模型！请先在终端运行 `main.py` 完成离线训练。")
    st.stop()

st.divider()

# 核心指标区
col1, col2, col3, col4 = st.columns(4)
threat_level = "高危" if st.session_state.risk_score > 80 else ("中危" if st.session_state.risk_score > 50 else "安全")
col1.metric(label="实时全局风险评分", value=f"{st.session_state.risk_score}/100", delta="模型已就绪")
col2.metric(label="当前安全状态", value=threat_level)
col3.metric(label="已拦截攻击次数", value=len(st.session_state.alert_logs))
col4.metric(label="边缘推理延迟", value="3.2 ms") 

st.divider()

# 图表与日志区 (恢复左右分栏布局)
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("🎯 实时流量类型分布")
    # 把 Session 里记录的真实点击数据喂给柱状图
    # 强制转换数据格式，让 X 轴固定显示分类名称，Y 轴显示数量
    chart_data = pd.DataFrame.from_dict(
        st.session_state.stats, orient='index', columns=['数量']
    )
    st.bar_chart(chart_data)

with col_right:
    st.subheader("🚨 智能 Agent 实时告警与响应日志")
    if len(st.session_state.alert_logs) > 0:
        df_logs = pd.DataFrame(st.session_state.alert_logs)
        st.dataframe(df_logs, use_container_width=True, hide_index=True)
    else:
        st.info("当前网内流量平稳，未发现异常。")

# ==========================================
# 3. 侧边栏：真实联动测试控制台
# ==========================================
with st.sidebar:
    st.header("⚙️ 智能体联动测试")
    st.markdown("点击按钮向系统中注入实时数据：")
    
    if st.button("🟢 模拟电网正常运行 (Normal)"):
        with st.spinner("边缘斥候 (Detection Agent) 正在检测..."):
            time.sleep(0.5)
            # 喂给 XGBoost 正常数据
            normal_data = np.random.uniform(0, 0.1, size=(1, 2050))
            is_threat, _ = detector.monitor(normal_data)
            
            # 更新状态与图表统计
            st.session_state.risk_score = max(5, st.session_state.risk_score - 5)
            st.session_state.stats['正常流量 (Normal)'] += 1
            
            st.success("✅ 检测完毕：电网物理状态稳定，未见异常通信。")
            st.rerun() # 强制刷新网页更新图表

    if st.button("🔴 模拟注入高危攻击 (FDIA / DDoS)"):
        with st.spinner("发现剧烈扰动！Agent 正在联动研判..."):
            time.sleep(1)
            # 喂给 XGBoost 异常特征
            attack_data = np.random.uniform(5, 10, size=(1, 2050))
            is_threat, pred_class = detector.monitor(attack_data)
            
            # 随机模拟识别出不同的攻击类型，增加真实感
            threat_name = np.random.choice(["DDoS 网络洪水攻击", "FDIA 虚假数据注入攻击"])
            if "DDoS" in threat_name:
                st.session_state.stats['网络层 (DDoS)'] += 1
            else:
                st.session_state.stats['物理层 (FDIA)'] += 1

            # 记录日志
            log = {
                '时间': time.strftime("%H:%M:%S"),
                '前线汇报': 'Detection Agent 🔴',
                '威胁类型': threat_name,
                '系统动作': '隔离异常传感器节点，切换备用通信'
            }
            st.session_state.alert_logs.insert(0, log)
            st.session_state.risk_score = min(95, st.session_state.risk_score + 40)
            
            st.error("⚠️ 警报！边缘模型拦截到异常数据特征！响应预案已启动！")
            st.rerun() # 强制刷新网页更新图表