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

st.title("🏈 Super Bowl LX: Advanced Simulator")

# --- 1) TOP EXPANDER (FORMER SIDEBAR) ---
with st.expander("🛠️ Simulation Settings & Team Selection", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    
    afc_teams_df = data[data['team_conf'] == 'AFC'].sort_values('team_name')
    nfc_teams_df = data[data['team_conf'] == 'NFC'].sort_values('team_name')

    # Logic to find the index of Patriots and Seahawks for defaults
    try:
        pats_idx = afc_teams_df.get_loc(afc_teams_df[afc_teams_df['team_name'] == 'New England Patriots'].index[0])
        sea_idx = nfc_teams_df.get_loc(nfc_teams_df[nfc_teams_df['team_name'] == 'Seattle Seahawks'].index[0])
    except:
        pats_idx, sea_idx = 0, 0 # Fallback if names don't match exactly

    with col_a:
        st.markdown("**The Matchup**")
        # 1) Make Patriots and Seahawks default
        afc_choice = st.selectbox("Select AFC Champion", afc_teams_df['team_name'], index=int(pats_idx))
        nfc_choice = st.selectbox("Select NFC Champion", nfc_teams_df['team_name'], index=int(sea_idx))
        
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
        # 3) Simulation control modifier
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
    
    col1, col2 = st.columns(2)
    afc_win_pct = (afc_res > nfc_res).mean() * 100
    nfc_win_pct = (nfc_res > afc_res).mean() * 100
    
    with col1:
        st.subheader("Win Probability")
        
        # 2) Aligned Labels: Team 1 (Right), vs (Center), Team 2 (Left)
        l_col, c_col, r_col = st.columns([2, 1, 2])
        l_col.markdown(f"<p style='text-align: right;'><b>{afc_choice}</b> ({afc_win_pct:.1f}%)</p>", unsafe_allow_html=True)
        c_col.markdown("<p style='text-align: center;'>vs</p>", unsafe_allow_html=True)
        r_col.markdown(f"<p style='text-align: left;'><b>{nfc_choice}</b> ({nfc_win_pct:.1f}%)</p>", unsafe_allow_html=True)
        
        st.progress(afc_win_pct / 100)

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
