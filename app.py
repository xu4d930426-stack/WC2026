import numpy as np
import scipy.stats as stats
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 網頁基本設定與官方 48 隊名單 (A-L組)
# ==========================================
st.set_page_config(
    page_title="PrediGoal Pro+ Ultra - 2026 世界盃 AI 終極預測系統", 
    page_icon="🏆", 
    layout="wide"
)

WORLD_CUP_48_TEAMS = {
    "Group A": {"Mexico": "墨西哥", "South Africa": "南非", "South Korea": "南韓", "Czechia": "捷克"},
    "Group B": {"Canada": "加拿大", "Bosnia and Herzegovina": "波赫", "Qatar": "卡達", "Switzerland": "瑞士"},
    "Group C": {"Brazil": "巴西", "Morocco": "摩洛哥", "Haiti": "海地", "Scotland": "蘇格蘭"},
    "Group D": {"USA": "美國", "Paraguay": "巴拉圭", "Australia": "澳洲", "Türkiye": "土耳其"},
    "Group E": {"Germany": "德國", "Curaçao": "古拉索", "Ivory Coast": "象牙海岸", "Ecuador": "厄瓜多"},
    "Group F": {"Netherlands": "荷蘭", "Japan": "日本", "Sweden": "瑞典", "Tunisia": "突尼西亞"},
    "Group G": {"Belgium": "比利時", "Egypt": "埃及", "Iran": "伊朗", "New Zealand": "紐西蘭"},
    "Group H": {"Spain": "西班牙", "Cabo Verde": "維德角", "Saudi Arabia": "沙烏地阿拉伯", "Uruguay": "烏拉圭"},
    "Group I": {"France": "法國", "Senegal": "塞內加爾", "Iraq": "伊拉克", "Norway": "挪威"},
    "Group J": {"Argentina": "阿根廷", "Algeria": "阿爾及利亞", "Austria": "奧地利", "Jordan": "約旦"},
    "Group K": {"Portugal": "葡萄牙", "Congo DR": "剛果民主共和國", "Uzbekistan": "烏茲別克", "Colombia": "哥倫比亞"},
    "Group L": {"England": "英格蘭", "Croatia": "克羅埃西亞", "Ghana": "迦納", "Panama": "巴拿馬"}
}

TEAM_CH_TO_ENG = {}
TEAM_TO_GROUP = {}
ALL_CH_TEAMS = []
for grp, t_dict in WORLD_CUP_48_TEAMS.items():
    for eng, ch in t_dict.items():
        TEAM_CH_TO_ENG[ch] = eng
        TEAM_TO_GROUP[ch] = grp
        ALL_CH_TEAMS.append(ch)

ALL_CH_TEAMS.sort()
GLOBAL_AVG_GOALS = 1.35 

# 全球國家硬實力雙維度特徵：[進攻係數, 防守係數]
TEAM_ADVANCED_STATS = {
    "Argentina": [1.85, 0.75], "Brazil": [1.80, 0.80], "France": [1.80, 0.78], "England": [1.75, 0.82], 
    "Spain": [1.75, 0.80], "Germany": [1.65, 0.88], "Netherlands": [1.60, 0.85], "Portugal": [1.60, 0.86],
    "Uruguay": [1.50, 0.88], "Colombia": [1.45, 0.90], "Belgium": [1.45, 0.92], "Croatia": [1.40, 0.90],
    "Japan": [1.35, 0.92], "Morocco": [1.25, 0.82], "Switzerland": [1.30, 0.92], "USA": [1.25, 0.95], 
    "Mexico": [1.25, 0.98], "Ecuador": [1.20, 0.90], "South Korea": [1.20, 1.02], "Austria": [1.20, 1.00], 
    "Sweden": [1.20, 1.02], "Türkiye": [1.15, 1.05], "Senegal": [1.15, 1.00], "Norway": [1.25, 1.12], 
    "Canada": [1.15, 1.08], "Australia": [1.10, 1.05], "Iran": [1.05, 0.95], "Czechia": [1.10, 1.08], 
    "Egypt": [1.10, 1.04], "Algeria": [1.05, 1.08], "Tunisia": [0.95, 0.98], "Paraguay": [0.90, 0.95], 
    "Bosnia and Herzegovina": [1.00, 1.12], "Scotland": [0.95, 1.05], "Ghana": [1.00, 1.15], 
    "Ivory Coast": [1.00, 1.12], "Saudi Arabia": [0.95, 1.10], "Qatar": [0.95, 1.18], "Uzbekistan": [0.90, 1.10], 
    "Panama": [0.90, 1.15], "South Africa": [0.90, 1.14], "Iraq": [0.85, 1.12], "Cabo Verde": [0.85, 1.15], 
    "Jordan": [0.80, 1.18], "Congo DR": [0.80, 1.20], "New Zealand": [0.75, 1.22], "Haiti": [0.70, 1.25], 
    "Curaçao": [0.65, 1.30]
}

# ==========================================
# 2. 核心大數據演算法：卜瓦松 + 蒙地卡羅動態模擬
# ==========================================
def run_monte_carlo_simulation(lambda_a, lambda_b, num_simulations=10000, max_goals=6):
    sim_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for _ in range(num_simulations):
        goals_a = 0
        goals_b = 0
        for period in range(3):
            current_lambda_a = lambda_a / 3.0
            current_lambda_b = lambda_b / 3.0
            if goals_a > goals_b:
                current_lambda_b *= 1.12
                current_lambda_a *= 0.95
            elif goals_b > goals_a:
                current_lambda_a *= 1.12
                current_lambda_b *= 0.95
            goals_a += np.random.poisson(current_lambda_a)
            goals_b += np.random.poisson(current_lambda_b)
        goals_a = min(goals_a, max_goals)
        goals_b = min(goals_b, max_goals)
        sim_matrix[goals_a, goals_b] += 1
    sim_matrix /= num_simulations
    return sim_matrix

def calculate_match_probabilities(team_a_ch, team_b_ch, max_goals=6):
    eng_a = TEAM_CH_TO_ENG[team_a_ch]
    eng_b = TEAM_CH_TO_ENG[team_b_ch]
    
    # 複製參數，避免原地修改全局字典
    stats_a = list(TEAM_ADVANCED_STATS.get(eng_a, [1.0, 1.0]))
    stats_b = list(TEAM_ADVANCED_STATS.get(eng_b, [1.0, 1.0]))
    
    # -------------------------------------------------------------
    # 🌟 大賽動態環境調節器 (修正名氣虛高、強化戰術保守度與主場)
    # -------------------------------------------------------------
    # 1. 主場/地緣優勢校正 (主隊進攻增加，防守失球率下降)
    stats_a[0] *= 1.08  # 主隊進攻 +8%
    stats_a[1] *= 0.95  # 主隊防守變好 -5%失球
    
    # 2. 強隊開局冷卻抑制 (調低傳統豪門小組賽初期的進攻爆發，模擬保守踢法)
    if stats_a[0] > 1.5: stats_a[0] *= 0.86
    if stats_b[0] > 1.5: stats_b[0] *= 0.86
    
    # 3. 弱隊死守剛性補償 (強弱懸殊時，自動調強弱隊防守，模擬鐵桶陣)
    if (stats_a[0] - stats_b[0]) > 0.4:
        stats_b[1] *= 0.88  # 客隊防守失球率下降 12%
    elif (stats_b[0] - stats_a[0]) > 0.4:
        stats_a[1] *= 0.88  # 主隊防守失球率下降 12%
    # -------------------------------------------------------------
    
    # 計算主客隊個別的進球期望值純量
    lambda_a = GLOBAL_AVG_GOALS * stats_a[0] * stats_b[1]
    lambda_b = GLOBAL_AVG_GOALS * stats_b[0] * stats_a[1]
    
    # 大賽整體趨於謹慎，期望值全面微幅壓縮
    lambda_a *= 0.94
    lambda_b *= 0.94
    
    dynamic_matrix = run_monte_carlo_simulation(lambda_a, lambda_b, num_simulations=10000, max_goals=max_goals)
    
    # 嚴謹計算勝、平、負率
    win_p, draw_p, loss_p = 0.0, 0.0, 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if i > j:
                win_p += dynamic_matrix[i, j]
            elif i == j:
                draw_p += dynamic_matrix[i, j]
            else:
                loss_p += dynamic_matrix[i, j]
    
    total_goals_dist = {}
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            g = i + j
            total_goals_dist[g] = total_goals_dist.get(g, 0) + dynamic_matrix[i, j]
            
    p_under_1_5 = total_goals_dist.get(0, 0) + total_goals_dist.get(1, 0)
    p_over_1_5 = 1.0 - p_under_1_5
    p_under_2_5 = p_under_1_5 + total_goals_dist.get(2, 0)
    p_over_2_5 = 1.0 - p_under_2_5
    
    # 計算雙方進球 (BTTS)
    zero_lines_prob = np.sum(dynamic_matrix[0, :]) + np.sum(dynamic_matrix[:, 0]) - dynamic_matrix[0, 0]
    p_btts = max(0.0, 1.0 - zero_lines_prob)
    
    prob_zero_a = stats.poisson.pmf(0, lambda_a)
    prob_zero_b = stats.poisson.pmf(0, lambda_b)
    score_a = int(round(lambda_a))
    score_b = int(round(lambda_b))
    
    if prob_zero_a > 0.45: score_a = 0
    if prob_zero_b > 0.45: score_b = 0
    best_score = (score_a, score_b)
    
    correct_scores = []
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            correct_scores.append(((i, j), dynamic_matrix[i, j]))
    correct_scores.sort(key=lambda x: x[1], reverse=True)
    
    return {
        "lambda_a": lambda_a, "lambda_b": lambda_b,
        "win_p": win_p, "draw_p": draw_p, "loss_p": loss_p,
        "p_over_1_5": p_over_1_5, "p_over_2_5": p_over_2_5, "p_btts": p_btts,
        "xg_score": best_score, "top_scores": correct_scores[:5], "matrix": dynamic_matrix
    }

# ==========================================
# 3. Streamlit 網頁 UI 渲染
# ==========================================
st.markdown("### 🏆 PrediGoal Pro+ Ultra - 2026 世界盃 AI 終極預測系統")

mode = st.sidebar.radio("🔮 請選擇功能模組", ["單場精準大數據預測", "小組對戰交叉模擬"])

if mode == "單場精準大數據預測":
    st.header("🎯 國家隊強強對決預測")
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("請選擇 主隊 (Home)", ALL_CH_TEAMS, index=ALL_CH_TEAMS.index("土耳其"))
    with col2:
        team_b = st.selectbox("請选择 客隊 (Away)", ALL_CH_TEAMS, index=ALL_CH_TEAMS.index("澳洲"))
        
    if team_a != team_b:
        res = calculate_match_probabilities(team_a, team_b)
        
        # 勝平負大盤
        st.subheader("📊 獨贏盤 (勝平負) 概率預測 (動態模擬優化)")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric(f"🟢 {team_a} 獨贏勝率", f"{res['win_p']*100:.2f}%", f"AI預期進球(xG): {res['lambda_a']:.2f}")
        m_col2.metric("🟡 常規時間平局率", f"{res['draw_p']*100:.2f}%")
        m_col3.metric(f"🔵 {team_b} 獨贏勝率", f"{res['loss_p']*100:.2f}%", f"AI預期進球(xG): {res['lambda_b']:.2f}")
        
        # 指標對照
        st.write("---")
        st.subheader("💡 核心科學指標對照")
        x_col1, x_col2 = st.columns(2)
        with x_col1:
            st.info(f"🤖 **AI 綜合情形期望值比分 (xG Score)： {res['xg_score'][0]} - {res['xg_score'][1]}**")
        with x_col2:
            top_score_tuple = res['top_scores'][0][0]
            top_score_prob = res['top_scores'][0][1]
            st.success(f"🎲 **單一最高機率比分 (Mode Score)： {top_score_tuple[0]} - {top_score_tuple[1]} ({top_score_prob*100:.1f}%)**")

        # 🎯 大小球與 BTTS 智能指標
        st.write("---")
        st.subheader("🎯 大小球與 BTTS 智能指標")
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        with rec_col1:
            st.markdown("**【全場 1.5 球盤口】**")
            st.write(f"📈 大 1.5 機率: {res['p_over_1_5']*100:.1f}%")
            st.write(f"📉 小 1.5 機率: {(1.0 - res['p_over_1_5'])*100:.1f}%")
            if res['p_over_1_5'] > 0.65:
                st.caption("💡 建議：高機率有 2 球以上，適合大球配置")
            else:
                st.caption("💡 建議：交火可能較為沉悶")
                
        with rec_col2:
            st.markdown("**【全場 2.5 球盤口】**")
            st.write(f"📈 大 2.5 機率: {res['p_over_2_5']*100:.1f}%")
            st.write(f"📉 小 2.5 機率: {(1.0 - res['p_over_2_5'])*100:.1f}%")
            if res['p_over_2_5'] > 0.55:
                st.caption("💡 建議：兩隊球風開放，推薦大球")
            else:
                st.caption("💡 建議：防守嚴密，傾向小球")

        with rec_col3:
            st.markdown("**【雙方進球 BTTS】**")
            st.write(f"🤝 雙方皆進球機率: {res['p_btts']*100:.1f}%")
            st.write(f"🚫 至少一隊零封機率: {(1.0 - res['p_btts'])*100:.1f}%")
            if res['p_btts'] > 0.50:
                st.caption("💡 建議：兩隊皆有破門能力")
