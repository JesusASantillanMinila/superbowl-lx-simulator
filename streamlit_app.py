import streamlit as st
import numpy as np
import pandas as pd
import nflreadpy as nfl 

# --- CONFIG & DATA LOAD ---
st.set_page_config(page_title="Super Bowl LX Simulator", layout="wide")

@st.cache_data
def load_nfl_metadata():
    teams = nfl.load_teams().to_pandas()
    teams = teams[teams['team_conf'].isin(['AFC', 'NFC'])]
    
    stats = nfl.load_team_stats([2024]).to_pandas()
    
    team_performance = stats.groupby('team').agg({
        'passing_epa': 'sum',
        'rushing_epa': 'sum',
    }).reset_index()

    team_performance['total_epa_calc'] = team_performance['passing_epa'] + team_performance['rushing_epa']
    
    df = pd.merge(
        teams[['team_abbr', 'team_conf', 'team_name']], 
        team_performance, 
        left_on='team_abbr', 
        right_on='team'
    )
    
    epa_min = df['total_epa_calc'].min()
    epa_max = df['total_epa_calc'].max()
    df['momentum'] = ((df['total_epa_calc'] - epa_min) / (epa_max - epa_min)) * 0.6 + 0.7
    
    return df

data = load_nfl_metadata()

st.title(" 🏈 Super Bowl LX: Advanced Simulator")

# --- 1) TOP EXPANDER ---
with st.expander("🛠️ Simulation Settings & Team Selection", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    
    afc_teams_df = data[data['team_conf'] == 'AFC'].sort_values('team_name')
    nfc_teams_df = data[data['team_conf'] == 'NFC'].sort_values('team_name')

    # FIX 1: Improved logic to find indices for Patriots and Seahawks
    afc_list = afc_teams_df['team_name'].tolist()
    nfc_list = nfc_teams_df['team_name'].tolist()
    
    try:
        pats_idx = afc_list.index('New England Patriots')
    except ValueError:
        pats_idx = 0
        
    try:
        sea_idx = nfc_list.index('Seattle Seahawks')
    except ValueError:
        sea_idx = 0

    with col_a:
        st.markdown("**The Matchup**")
        # Applying the default indices
        afc_choice = st.selectbox("Select AFC Champion", afc_list, index=pats_idx)
        nfc_choice = st.selectbox("Select NFC Champion", nfc_list, index=sea_idx)
        
        st.divider()
        st.markdown("**Strategy**")
        strat_map = {"Defensive": 0.90, "Balanced": 1.0, "Offensive": 1.10}
        afc_strat = st.select_slider(f"{afc_choice} Strategy", options=["Defensive", "Balanced", "Offensive"], value="Balanced")
        nfc_strat = st.select_slider(f"{nfc_choice} Strategy", options=["Defensive", "Balanced", "Offensive"], value="Balanced")

    with col_b:
        st.markdown("**Game State**")
        score_afc = st.number_input(f"{afc_choice} Current Score", 0, 100, 0)
        score_nfc = st.number_input(f"{nfc_choice} Current Score", 0, 100, 0)
        time_left = st.slider("Minutes Remaining", 1, 60, 60)
    
    with col_c:
        st.markdown("**Injury Report**")
        inj_map = {"None": 0.0, "Role": 0.03, "Starter": 0.07, "Star": 0.15, "Elite": 0.30}
        afc_inj_lvl = st.select_slider(f"{afc_choice} Injuries", options=["None", "Role", "Starter", "Star", "Elite"])
        nfc_inj_lvl = st.select_slider(f"{nfc_choice} Injuries", options=["None", "Role", "Starter", "Star", "Elite"])

        st.divider()
        st.markdown("**Environment**")
        weather = st.selectbox("Weather Conditions", ["Clear/Dome", "Rain/Wind", "Snow"])
        weather_map = {"Clear/Dome": 1.0, "Rain/Wind": 0.85, "Snow": 0.75}
        sim_count = st.select_slider("Simulations to Run", options=[1000, 5000, 10000, 25000, 50000], value=10000)

afc_mod = afc_teams_df[afc_teams_df['team_name'] == afc_choice]['momentum'].values[0]
nfc_mod = nfc_teams_df[nfc_teams_df['team_name'] == nfc_choice]['momentum'].values[0]

# --- SIMULATION ENGINE ---
def run_simulation(iterations):
    base_ppm = 0.45 * weather_map[weather]
    afc_final_rate = base_ppm * afc_mod * strat_map[afc_strat] * (1 - inj_map[afc_inj_lvl])
    nfc_final_rate = base_ppm * nfc_mod * strat_map[nfc_strat] * (1 - inj_map[nfc_inj_lvl])
    
    afc_sim = score_afc + np.random.poisson(afc_final_rate * time_left, iterations)
    nfc_sim = score_nfc + np.random.poisson(nfc_final_rate * time_left, iterations)
    return afc_sim, nfc_sim

# --- UI RESULTS ---
if st.button(f"🚀 Run Super Bowl Simulation", use_container_width=True):
    afc_res, nfc_res = run_simulation(sim_count)
    
    # 1. Identify raw wins and ties
    afc_raw_wins = (afc_res > nfc_res)
    nfc_raw_wins = (nfc_res > afc_res)
    ties = (afc_res == nfc_res)
    
    # 2. Calculate a "Strength Weight" based on their season performance (EPA Momentum)
    # This ensures the better team has a higher chance to win the "Overtime" tie-breaker
    total_momentum = afc_mod + nfc_mod
    afc_weight = afc_mod / total_momentum
    
    # 3. Resolve ties using the weighted probability
    # np.random.random generates a float between 0 and 1
    tie_breaker = np.random.random(sim_count) < afc_weight
    
    # 4. Final Win Totals (Raw wins + Ties won via tie-breaker)
    afc_final_wins = afc_raw_wins | (ties & tie_breaker)
    nfc_final_wins = nfc_raw_wins | (ties & ~tie_breaker)
    
    afc_win_pct = afc_final_wins.mean() * 100
    nfc_win_pct = nfc_final_wins.mean() * 100
    
    with col1:
        st.subheader("Win Probability")
        
        l_col, c_col, r_col = st.columns([2, 1, 2])
        l_col.markdown(f"<p style='text-align: right;'><b>{afc_choice}</b> ({afc_win_pct:.1f}%)</p>", unsafe_allow_html=True)
        c_col.markdown("<p style='text-align: center;'>vs</p>", unsafe_allow_html=True)
        r_col.markdown(f"<p style='text-align: left;'><b>{nfc_choice}</b> ({nfc_win_pct:.1f}%)</p>", unsafe_allow_html=True)
        
        # FIX 2: Custom Red and Blue Progress Bar using CSS
        st.markdown(f"""
            <div style="width: 100%; background-color: #003366; border-radius: 5px; height: 25px;">
                <div style="width: {afc_win_pct}%; background-color: #C60C30; height: 25px; border-radius: 5px 0 0 5px; text-align: center; color: white; line-height: 25px;">
                </div>
            </div>
            <p style='font-size: 10px; color: gray; margin-top: 5px;'>Red: {afc_choice} | Blue: {nfc_choice}</p>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Projected Final Score")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric(afc_choice, f"{afc_res.mean():.1f}")
        m_col2.metric(nfc_choice, f"{nfc_res.mean():.1f}")

    st.divider()
    st.subheader("Point Spread Distribution")
    st.info(f"Positive values favor {afc_choice}, negative values favor {nfc_choice}.")
    
    spreads = afc_res - nfc_res
    spread_data = pd.Series(spreads).value_counts().sort_index().reset_index()
    spread_data.columns = ['Point Spread', 'Frequency']
    st.area_chart(spread_data.set_index('Point Spread'))
