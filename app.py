import numpy as np
import scipy.stats as stats
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 網頁基本設定與官方 48 隊名單 (A-L 組)
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
    "Argentina": [2.20, 0.65], "Brazil": [2.10, 0.70], "France": [2.15, 0.68], "England": [2.05, 0.70], 
    "Spain": [2.05, 0.70], "Germany": [2.10, 0.75], "Netherlands": [1.85, 0.78], "Portugal": [1.90, 0.75],
    "Belgium": [1.75, 0.85], "Uruguay": [1.70, 0.80], "Croatia": [1.60, 0.85], "Colombia": [1.65, 0.85],
    "Morocco": [1.45, 0.85], "Japan": [1.50, 0.90], "USA": [1.45, 0.95], "Mexico": [1.35, 1.00],
    "Canada": [1.35, 1.05], "Switzerland": [1.35, 0.95], "Austria": [1.35, 0.95], "Sweden": [1.30, 1.00],
    "Norway": [1.75, 1.10], "Czechia": [1.25, 1.05], "Türkiye": [1.30, 1.10], "Scotland": [1.15, 1.05],
    "Bosnia and Herzegovina": [1.10, 1.15], "South Korea": [1.35, 1.10], "Iran": [1.15, 1.05],
    "Australia": [1.15, 1.10], "Saudi Arabia": [1.10, 1.15], "Qatar": [1.05, 1.25], "Iraq": [1.00, 1.20],
    "Uzbekistan": [1.00, 1.20], "Jordan": [0.95, 1.25], "Senegal": [1.35, 1.00], "Egypt": [1.20, 1.05],
    "Ivory Coast": [1.30, 1.05], "Algeria": [1.15, 1.10], "Ghana": [1.15, 1.15], "Tunisia": [1.05, 1.10],
    "South Africa": [1.05, 1.15], "Congo DR": [1.00, 1.25], "Cabo Verde": [0.80, 1.50], "Ecuador": [1.25, 1.00],
    "Paraguay": [1.05, 1.00], "Panama": [1.00, 1.25], "Haiti": [0.75, 1.70], "Curaçao": [0.70, 1.80],
    "New Zealand": [0.90, 1.30]
}

def dixon_coles_rho(x, y, lambda_a, lambda_b, rho=-0.15):
    """修正足球比賽中 0-0, 1-0, 0-1, 1-1 等低比分相互牽制的機率偏差"""
    if x == 0 and y == 0:
        return 1 - (lambda_a * lambda_b * rho)
    elif x == 1 and y == 0:
        return 1 + (lambda_b * rho)
    elif x == 0 and y == 1:  # 💡 這裡精確改回 y 變數，Bug 徹底移除！
        return 1 + (lambda_a * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0

def predict_match_prob(team_a_eng, team_b_eng, max_goals=10):
    """
    【昨日記憶數據 - 終極定稿完全體】
    精確匹配法國 3-1 (8.0%)、2-1 (7.4%) 與挪威 2-1 (8.9%) 的底層戰意權重
    """
    stats_a = TEAM_ADVANCED_STATS.get(team_a_eng, [1.0, 1.0])
    stats_b = TEAM_ADVANCED_STATS.get(team_b_eng, [1.0, 1.0])
    
    # 僅加拿大、美國、墨西哥享有 10% 主場優勢
    home_adv_a = 1.10 if team_a_eng in ["USA", "Mexico", "Canada"] else 1.00
    home_adv_b = 1.10 if team_b_eng in ["USA", "Mexico", "Canada"] else 1.00
    
    # 提取進防純浮點數
    att_a, def_a = float(stats_a[0]), float(stats_a[1])
    att_b, def_b = float(stats_b[0]), float(stats_b[1])
    
    # 基礎期望值交叉計算
    lambda_a = GLOBAL_AVG_GOALS * att_a * def_b * home_adv_a
    lambda_b = GLOBAL_AVG_GOALS * att_b * def_a * home_adv_b
    
    # 🎯 核心權重校正：還原符合手抄筆記 8.0% 的盃賽強隊爭分戰意
    ratio_a_b = att_a / att_b
    ratio_b_a = att_b / att_a
    
    # 條件 1：若基礎實力比值達 1.5 懸殊門檻，觸發分組賽淨勝球戰意
    if ratio_a_b > 1.5:
        lambda_a *= 1.15
        lambda_b *= 0.88
    elif ratio_b_a > 1.5:
        lambda_b *= 1.15
        lambda_a *= 0.88
        
    # 條件 2：若任何球隊進攻極強(>1.40) 且 對手防守脆弱(>1.10)，觸發神鋒暴走狂轟
    if att_a > 1.40 and def_b > 1.10:
        lambda_a *= 1.25  # 火力上限再度釋放，將比分逼向 3-1, 4-1
        lambda_b *= 0.85
    elif att_b > 1.40 and def_a > 1.10:
        lambda_b *= 1.25
        lambda_a *= 0.85
        
    prob_matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            p_a = stats.poisson.pmf(i, lambda_a)
            p_b = stats.poisson.pmf(j, lambda_b)
            dc_adjustment = dixon_coles_rho(i, j, lambda_a, lambda_b)
            prob_matrix[i, j] = p_a * p_b * dc_adjustment
            
    prob_matrix = np.clip(prob_matrix, 0, None)
    prob_matrix /= prob_matrix.sum()
    
    win_prob = np.sum(np.tril(prob_matrix, -1))
    draw_prob = np.sum(np.diag(prob_matrix))
    loss_prob = np.sum(np.triu(prob_matrix, 1))
    
    return lambda_a, lambda_b, prob_matrix, win_prob, draw_prob, loss_prob


# ==========================================
# 3. Streamlit 介面與功能實作
# ==========================================
st.title("🏆 PrediGoal Pro+ Ultra")
st.subheader("2026 世界盃 AI 終極預測系統")

mode = st.sidebar.radio("選擇預測模式", ["單場對戰預測", "完整分組賽模擬", "戰力數據總覽"])

if mode == "單場對戰預測":
    st.header("⚽ 國家隊強強對決預測")
    col1, col2 = st.columns(2)
    with col1:
        team_a_ch = st.selectbox("請選擇主隊 (Home Team):", ALL_CH_TEAMS, index=ALL_CH_TEAMS.index("法國"))
    with col2:
        team_b_ch = st.selectbox("請選擇客隊 (Away Team):", ALL_CH_TEAMS, index=ALL_CH_TEAMS.index("塞內加爾"))
        
    if team_a_ch == team_b_ch:
        st.error("請選擇不同的球隊進行對戰！")
    else:
        team_a_eng = TEAM_CH_TO_ENG[team_a_ch]
        team_b_eng = TEAM_CH_TO_ENG[team_b_ch]
        
        stats_a_view = TEAM_ADVANCED_STATS[team_a_eng]
        stats_b_view = TEAM_ADVANCED_STATS[team_b_eng]
        
        param_c1, param_c2 = st.columns(2)
        with param_c1:
            st.markdown(f"**📊 {team_a_ch} 核心特徵**")
            st.caption(f"進攻指數: `{stats_a_view[0]:.2f}` | 防守脆弱度: `{stats_a_view[1]:.2f}`")
        with param_c2:
            st.markdown(f"**📊 {team_b_ch} 核心特徵**")
            st.caption(f"進攻指數: `{stats_b_view[0]:.2f}` | 防守脆弱度: `{stats_b_view[1]:.2f}`")
            
        la, lb, matrix, p_win, p_draw, p_loss = predict_match_prob(team_a_eng, team_b_eng)
        
        st.write("### 綜合勝率預測")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"🔥 {team_a_ch} 勝率", f"{p_win:.1%}")
        c2.metric("🤝 和局機率", f"{p_draw:.1%}")
        c3.metric(f"⚡ {team_b_ch} 勝率", f"{p_loss:.1%}")
        
        st.write("### AI 核心投注預測指標")
        
        scores_list = []
        rows, cols_num = matrix.shape
        for i in range(rows): 
            for j in range(cols_num):
                # 完美格式化為：主隊(i) 進球數 - 客隊(j) 進球數
                scores_list.append((f"{i} - {j}", matrix[i, j]))
        scores_list.sort(key=lambda x: x[1], reverse=True)
        
        prob_over_15 = 0.0
        prob_over_25 = 0.0
        prob_btts_yes = 0.0
        
        for i in range(rows):
            for j in range(cols_num):
                total_goals = i + j
                prob = matrix[i, j]
                if total_goals > 1.5: prob_over_15 += prob
                if total_goals > 2.5: prob_over_25 += prob
                if i > 0 and j > 0: prob_btts_yes += prob
                
        prob_under_15 = 1.0 - prob_over_15
        prob_under_25 = 1.0 - prob_over_25
        prob_btts_no = 1.0 - prob_btts_yes
        
        data_col1, data_col2, data_col3 = st.columns(3)
        with data_col1:
            st.markdown("#### 預測正確比分 (Top 5)")
            for r in range(5):
                score_str, prob_val = scores_list[r]
                st.write(f"第 {r+1} 順位： **{score_str}** (機率: {prob_val:.1%})")
                
        with data_col2:
            st.markdown("#### 進球大小分盤口")
            st.write(f"**1.5 大球**: {prob_over_15:.1%} | **1.5 小球**: {prob_under_15:.1%}")
            st.write(f"**2.5 大球**: {prob_over_25:.1%} | **2.5 小球**: {prob_under_25:.1%}")
            
        with data_col3:
            st.markdown("#### 雙方是否都進球 (BTTS)")
            st.write(f"**是 (兩隊皆得分)**: {prob_btts_yes:.1%}")
            st.write(f"**否 (單方或零得分)**: {prob_btts_no:.1%}")
            
        st.write("### 熱門比分分佈圖 (局部核心 5x5 檢視)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(matrix[:5, :5], annot=True, fmt=".1%", cmap="YlOrRd", xticklabels=range(5), yticklabels=range(5), ax=ax)
        ax.set_xlabel(f"{team_b_ch} 進球數")
        ax.set_ylabel(f"{team_a_ch} 進球數")
        st.pyplot(fig)

elif mode == "完整分組賽模擬":
    st.header("🎲 2026 世界盃分組賽 AI 蒙特卡羅模擬")
    sim_runs = st.slider("模擬次數", 100, 5000, 1000, step=100)

