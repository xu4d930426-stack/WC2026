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

# ==========================================
# 2. Dixon-Coles 補正演算法 與 新賽制擺大巴完全體模型
# ==========================================
def dixon_coles_rho(x, y, lambda_a, lambda_b, rho=-0.15):
    """修正足球比賽中 0-0, 1-0, 0-1, 1-1 等低比分相互牽制的機率偏差"""
    if x == 0 and y == 0:
        return 1 - (lambda_a * lambda_b * rho)
    elif x == 1 and y == 0:
        return 1 + (lambda_b * rho)
    elif x == 0 and y == 1:
        return 1 + (lambda_a * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0

def predict_match_prob(team_a_eng, team_b_eng, max_goals=10):
    """
    【2026世界盃完全體後台】
    精確提取純浮點數變數，結合美墨加主場限制、神鋒暴走與新賽制擺大巴機制
    """
    stats_a = TEAM_ADVANCED_STATS.get(team_a_eng, [1.0, 1.0])
    stats_b = TEAM_ADVANCED_STATS.get(team_b_eng, [1.0, 1.0])
    
    # =======================================================
    # 🎯 核心修復：精確提取 [0]進攻特徵 與 [1]防守脆弱度
    # =======================================================
    att_a = float(stats_a[0])  # 主隊進攻
    def_a = float(stats_a[1])  # 主隊防守
    att_b = float(stats_b[0])  # 客隊進攻
    def_b = float(stats_b[1])  # 客隊防守

    
    # 2. 東道主限制：僅加拿大、美國、墨西哥被選為主隊時享有 10% 主場優勢
    home_adv_a = 1.10 if team_a_eng in ["USA", "Mexico", "Canada"] else 1.00
    home_adv_b = 1.10 if team_b_eng in ["USA", "Mexico", "Canada"] else 1.00
    
    # 3. 使用純浮點數變數計算基礎期望值 (主攻 * 客防)
    lambda_a = GLOBAL_AVG_GOALS * att_a * def_b * home_adv_a
    lambda_b = GLOBAL_AVG_GOALS * att_b * def_a * home_adv_b
    
    # =======================================================
    # 🎯 戰意、神鋒與新賽制大巴機制（互斥階梯重構版）
    # =======================================================
    ratio_a_b = att_a / att_b
    ratio_b_a = att_b / att_a
    
    # 【核心修復】：改用 if-elif 互斥結構，一場比賽只會觸發一種極端戰術，絕不重複相乘！
    if ratio_a_b > 1.6:
        # 情況 A：實力極端懸殊，弱隊全面退守擺大巴，強隊火力受到戰術嚴重壓縮
        lambda_a *= 0.85  # 西班牙、葡萄牙場強制在此處攔截，火力急凍！
        lambda_b *= 0.50
    elif ratio_b_a > 1.6:
        lambda_b *= 0.85
        lambda_a *= 0.50
        
    elif ratio_a_b > 1.5:
        # 情況 B：標準強弱懸殊，觸發常態分組賽淨勝球爭分戰意
        lambda_a *= 1.15
        lambda_b *= 0.88
    elif ratio_b_a > 1.5:
        lambda_b *= 1.15
        lambda_a *= 0.88
        
    elif att_a > 1.40 and def_b > 1.10:
        # 情況 C：實力有差距且遇到防線漏勺，強隊神鋒狂轟暴走
        lambda_a *= 1.25
        lambda_b *= 0.85
    elif att_b > 1.40 and def_a > 1.10:
        lambda_b *= 1.25
        lambda_a *= 0.85

        
    # 5. 構建機率矩陣與中立國 Dixon-Coles 智慧切換
    is_neutral = team_a_eng not in ["USA", "Mexico", "Canada"] and team_b_eng not in ["USA", "Mexico", "Canada"]
    current_rho = 0.0 if is_neutral else -0.15
        
    prob_matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            p_a = stats.poisson.pmf(i, lambda_a)
            p_b = stats.poisson.pmf(j, lambda_b)
            dc_adjustment = dixon_coles_rho(i, j, lambda_a, lambda_b, rho=current_rho)
            prob_matrix[i, j] = p_a * p_b * dc_adjustment
            
    prob_matrix = np.clip(prob_matrix, 0, None)
    prob_matrix /= prob_matrix.sum()
    
    win_prob = np.sum(np.tril(prob_matrix, -1))
    draw_prob = np.sum(np.diag(prob_matrix))
    loss_prob = np.sum(np.triu(prob_matrix, 1))
    
    return lambda_a, lambda_b, prob_matrix, win_prob, draw_prob, loss_prob

def simulate_match_score(lambda_a, lambda_b):
    return np.random.poisson(lambda_a), np.random.poisson(lambda_b)

mode = st.sidebar.radio("選擇預測模式", ["單場對戰預測", "完整分組賽模擬", "戰力數據總覽"])

# 模式 1：單場對戰預測
if mode == "單場對戰預測":
    
    st.header("⚽ 國家隊強強對決預測")
    
    # 🎯 投注優化：調整為客隊在左、主隊在右的直觀順序
    col1, col2 = st.columns(2)
    with col1:
        team_b_ch = st.selectbox("請選擇客隊 (Away Team):", ALL_CH_TEAMS, index=ALL_CH_TEAMS.index("維德角"))
    with col2:
        team_a_ch = st.selectbox("請選擇主隊 (Home Team):", ALL_CH_TEAMS, index=ALL_CH_TEAMS.index("西班牙"))
        
    if team_a_ch == team_b_ch:
        st.error("請選擇不同的球隊進行對戰！")
    else:
        team_a_eng = TEAM_CH_TO_ENG[team_a_ch]
        team_b_eng = TEAM_CH_TO_ENG[team_b_ch]
        
        stats_a_view = TEAM_ADVANCED_STATS[team_a_eng]
        stats_b_view = TEAM_ADVANCED_STATS[team_b_eng]
        
        # 🎯 戰力特徵卡片同步對調
        param_c1, param_c2 = st.columns(2)
        with param_c1:
            st.markdown(f"**📊 {team_b_ch} (客隊) 核心特徵**")
            st.caption(f"進攻指數: `{stats_b_view[0]:.2f}` | 防守脆弱度: `{stats_b_view[1]:.2f}`")
        with param_c2:
            st.markdown(f"**📊 {team_a_ch} (主隊) 核心特徵**")
            st.caption(f"進攻指數: `{stats_a_view[0]:.2f}` | 防守脆弱度: `{stats_a_view[1]:.2f}`")
            
        la, lb, matrix, p_win, p_draw, p_loss = predict_match_prob(team_a_eng, team_b_eng)
        
        st.write("### 綜合勝率預測")
        c1, c2, c3 = st.columns(3)
        # 🎯 勝率看板同步調整為：客勝 | 和局 | 主勝
        c1.metric(f"⚡ {team_b_ch} (客勝) 機率", f"{p_loss:.1%}")
        c2.metric("🤝 和局機率", f"{p_draw:.1%}")
        c3.metric(f"🔥 {team_a_ch} (主勝) 機率", f"{p_win:.1%}")
        
        st.write("### AI 核心投注預測指標")
        
        scores_list = []
        rows, cols_num = matrix.shape
        for i in range(rows): 
            for j in range(cols_num):
                # 🎯 投注優化：精確調整比分格式為「客隊進球數 - 主隊進球數」，完美對齊手抄筆記
                scores_list.append((f"{j} - {i}", matrix[i, j]))
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
        # 🎯 熱圖的 X 軸與 Y 軸也對調，讓 X 軸在左邊代表客隊進球，Y 軸在右邊代表主隊進球
        sns.heatmap(matrix[:5, :5], annot=True, fmt=".1%", cmap="YlOrRd", xticklabels=range(5), yticklabels=range(5), ax=ax)
        ax.set_xlabel(f"{team_b_ch} (客隊) 進球數")
        ax.set_ylabel(f"{team_a_ch} (主隊) 進球數")
        st.pyplot(fig)


# 模式 2：完整分組賽模擬
elif mode == "完整分組賽模擬":
    st.header("🎲 2026 世界盃分組賽 AI 蒙特卡羅模擬")
    sim_runs = st.slider("選擇蒙特卡羅模擬次數", 100, 2000, 1000, step=100, help="次數越多越精準")
    
    if st.button("🚀 開始執行全量盃賽模擬"):
        # 初始化統計字典，記錄每隊在數千次模擬中晉級十六強的總次數
        advance_counts = {t: 0 for teams in WORLD_CUP_48_TEAMS.values() for t in teams.values()}
        
        with st.spinner("AI 正在使用最新神鋒戰意與擺大巴模型模擬數萬場對戰中..."):
            for sim in range(sim_runs):
                # 每次模擬獨立初始化積分榜
                sim_standings = {g: {t: {"points": 0, "gf": 0, "ga": 0} for t in teams.values()} for g, teams in WORLD_CUP_48_TEAMS.items()}
                
                for grp, teams in WORLD_CUP_48_TEAMS.items():
                    team_list = list(teams.values())
                    # 每個小組進行 6 場循環賽
                    for i in range(len(team_list)):
                        for j in range(i + 1, len(team_list)):
                            # 🎯 投注對調優化：完美同步單場預測的變數結構
                            tb, ta = team_list[i], team_list[j]
                            ta_eng, tb_eng = TEAM_CH_TO_ENG[ta], TEAM_CH_TO_ENG[tb]
                            
                            stats_a = TEAM_ADVANCED_STATS.get(ta_eng, [1.0, 1.0])
                            stats_b = TEAM_ADVANCED_STATS.get(tb_eng, [1.0, 1.0])
                            
                            # 1. 嚴格提取純浮點數，阻絕 Array 交叉污染 Bug
                            att_a, def_a = float(stats_a[0]), float(stats_a[1])
                            att_b, def_b = float(stats_b[0]), float(stats_b[1])
                            
                            # 2. 東道主限制主場優勢 (僅美墨加享有 1.10)
                            home_adv_a = 1.10 if ta_eng in ["USA", "Mexico", "Canada"] else 1.00
                            home_adv_b = 1.10 if tb_eng in ["USA", "Mexico", "Canada"] else 1.00
                            
                            # 3. 基礎期望值交叉相乘 (主攻 * 客防)
                            lambda_a = GLOBAL_AVG_GOALS * att_a * def_b * home_adv_a
                            lambda_b = GLOBAL_AVG_GOALS * att_b * def_a * home_adv_b
                            
                            # =======================================================
                            # 🎯 戰意、神鋒與新賽制大巴機制（完整分組賽模擬同步互斥版）
                            # =======================================================
                            ratio_a_b = att_a / att_b
                            ratio_b_a = att_b / att_a
                            
                            # 嚴格與單場預測同步，一場虛擬對戰只觸發一種戰術加成
                            if ratio_a_b > 1.6:
                                lambda_a *= 0.85  
                                lambda_b *= 0.50  
                            elif ratio_b_a > 1.6:
                                lambda_b *= 0.85
                                lambda_a *= 0.50
                                
                            elif ratio_a_b > 1.5:
                                lambda_a *= 1.15
                                lambda_b *= 0.88
                            elif ratio_b_a > 1.5:
                                lambda_b *= 1.15
                                lambda_a *= 0.88
                                
                            elif att_a > 1.40 and def_b > 1.10:
                                lambda_a *= 1.25
                                lambda_b *= 0.85
                            elif att_b > 1.40 and def_a > 1.10:
                                lambda_b *= 1.25
                                lambda_a *= 0.85

                            
                            # 6. 使用修正後的 λ 通過卜瓦松隨機抽樣得出該場比分
                            score_a = np.random.poisson(lambda_a)
                            score_b = np.random.poisson(lambda_b)
                            
                            # 7. 積分榜數據累加
                            sim_standings[grp][ta]["gf"] += score_a
                            sim_standings[grp][ta]["ga"] += score_b
                            sim_standings[grp][tb]["gf"] += score_b
                            sim_standings[grp][tb]["ga"] += score_a
                            
                            if score_a > score_b:
                                sim_standings[grp][ta]["points"] += 3
                            elif score_b > score_a:
                                sim_standings[grp][tb]["points"] += 3
                            else:
                                sim_standings[grp][ta]["points"] += 1
                                sim_standings[grp][tb]["points"] += 1
                
                # 判定 12 個小組的前兩名晉級十六強
                # 判定 12 個小組的前兩名晉級十六強
                for grp, teams_data in sim_standings.items():
                    sorted_teams = sorted(
                        teams_data.items(),
                        key=lambda x: (x[1]["points"], x[1]["gf"] - x[1]["ga"], x[1]["gf"]),
                        reverse=True
                    )
                    top1_team = sorted_teams[0][0]
                    top2_team = sorted_teams[1][0]
                    advance_counts[top1_team] += 1
                    advance_counts[top2_team] += 1
        
        st.success(f"🎉 已成功更新同步！並順利執行完畢 {sim_runs} 次分組賽完整演練！")
        
        # 畫面優雅渲染
        st.write("### 📊 AI 預測各小組十六強晉級機率榜 (新賽制擺大巴同步版)")
        display_cols = st.columns(3)
        for idx, (grp, teams) in enumerate(WORLD_CUP_48_TEAMS.items()):
            col_target = display_cols[idx % 3]
            with col_target:
                st.markdown(f"#### 🏆 {grp}")
                grp_res = []
                for t in teams.values():
                    prob = advance_counts[t] / sim_runs
                    grp_res.append({"球隊": t, "十六強晉級率": f"{prob:.1%}", "_raw": prob})
                
                df_grp = pd.DataFrame(grp_res).sort_values(by="_raw", ascending=False).drop(columns=["_raw"])
                st.dataframe(df_grp, use_container_width=True, hide_index=True)

# 模式 3：戰力數據總覽（與左側完全切齊，不留空格！）
elif mode == "戰力數據總覽":
    st.header("📊 2026 世界盃參賽球隊 AI 戰力指數總覽")

    st.write("底層雙維度特徵矩陣：進攻指數越高代表進攻火力越猛；防守脆弱度越低代表防線越穩固。")
    
    # 重新整理 DataFrame 排行榜
    leaderboard = []
    for grp, teams in WORLD_CUP_48_TEAMS.items():
        for eng, ch in teams.items():
            stats_vals = TEAM_ADVANCED_STATS.get(eng, [1.0, 1.0])
            leaderboard.append({
                "所屬小組": grp,
                "球隊中文名": ch,
                "球隊英文名": eng,
                "🚀 AI 進攻特徵指數": round(float(stats_vals[0]), 2),
                "🛡️ AI 防守脆弱度": round(float(stats_vals[1]), 2),
                "綜合戰力分級": "👑 超級豪門" if stats_vals[0] > 2.0 else ("⚔️ 一線強權" if stats_vals[0] > 1.5 else "🏐 中游韌性")
            })
            
    df_all = pd.DataFrame(leaderboard).sort_values(by="🚀 AI 進攻特徵指數", ascending=False)
    
    # 介面搜尋與篩選器
    search_q = st.text_input("🔍 輸入球隊名稱快速檢索戰力：", "")
    if search_q:
        df_all = df_all[df_all["球隊中文名"].str.contains(search_q) | df_all["球隊英文名"].str.contains(search_q, case=False)]
        
    st.dataframe(df_all, use_container_width=True, hide_index=True, height=500)
    
    # 戰力特徵分佈圖可視化
    st.write("### 📈 48 強戰力雙維度分佈象限")
    fig_all, ax_all = plt.subplots(figsize=(10, 5))
    # 這裡過濾主要豪門與有特色球隊進行打點展示，避免 48 顆點字體重疊
    show_labels = ["Argentina", "France", "Brazil", "Spain", "Norway", "Iraq", "Algeria", "Jordan", "Mexico", "USA"]
    
    x_vals = [t["🚀 AI 進攻特徵指數"] for t in leaderboard]
    y_vals = [t["🛡️ AI 防守脆弱度"] for t in leaderboard]
    names = [t["球隊中文名"] for t in leaderboard]
    eng_names = [t["球隊英文名"] for t in leaderboard]
    
    ax_all.scatter(x_vals, y_vals, color="crimson", alpha=0.6, edgecolors="white", s=80)
    for i, txt in enumerate(names):
        if eng_names[i] in show_labels:
            ax_all.annotate(txt, (x_vals[i]+0.02, y_vals[i]), fontsize=9, fontproperties="Microsoft JhengHei")
            
    ax_all.set_xlabel("🚀 進攻特徵指數 (越高越強)")
    ax_all.set_ylabel("🛡️ 防守脆弱度 (越低越穩)")
    ax_all.invert_yaxis()  # 反轉Y軸，讓防守好的(數值小)排在上方
    ax_all.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig_all)



