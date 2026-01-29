import streamlit as st
import numpy as np
import pandas as pd
import nflreadpy as nfl 
import altair as alt

# --- CONFIG & DATA LOAD ---
st.set_page_config(page_title="Super Bowl LX Simulator", layout="wide")

@st.cache_data
def load_nfl_metadata():
    # Load basic team info
    teams = nfl.load_teams().to_pandas()
    teams = teams[teams['team_conf'].isin(['AFC', 'NFC'])]
    
    # Load 2025 Stats
    stats = nfl.load_team_stats([2025]).to_pandas()
    
    # Aggregate both Offensive AND Defensive EPA
    team_performance = stats.groupby('team').agg({
        'passing_epa': 'mean',     
        'rushing_epa': 'mean',
        'def_passing_epa': 'mean', 
        'def_rushing_epa': 'mean'
    }).reset_index()

    # Calculate Strengths
    # Offensive strength: How much you score
    team_performance['off_epa'] = team_performance['passing_epa'] + team_performance['rushing_epa']
    # Defensive strength: How much you stop scoring (Negative is better)
    team_performance['def_epa'] = team_performance['def_passing_epa'] + team_performance['def_rushing_epa']
    
    # Net EPA for "Momentum" (Offense - Defense)
    # A team with +0.10 Offense and -0.05 Defense (Elite) = +0.15 Net
    team_performance['net_epa'] = team_performance['off_epa'] - team_performance['def_epa']
    
    df = pd.merge(
        teams[['team_abbr', 'team_conf', 'team_name', 'team_logo_wikipedia']], 
        team_performance, 
        left_on='team_abbr', 
        right_on='team'
    )
    
    # Normalize Momentum based on Net EPA
    epa_min, epa_max = df['net_epa'].min(), df['net_epa'].max()
    df['momentum'] = ((df['net_epa'] - epa_min) / (epa_max - epa_min)) * 0.6 + 0.7
    
    return df

data = load_nfl_metadata()

# --- UI --- 
st.title(" 🏈 Super Bowl LX: Advanced Simulator")

with st.expander("🛠️ Simulation Settings & Team Selection", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    
    afc_teams_df = data[data['team_conf'] == 'AFC'].sort_values('team_name')
    nfc_teams_df = data[data['team_conf'] == 'NFC'].sort_values('team_name')

    afc_list = afc_teams_df['team_name'].tolist()
    nfc_list = nfc_teams_df['team_name'].tolist()
    
    pats_idx = afc_list.index('New England Patriots') if 'New England Patriots' in afc_list else 0
    sea_idx = nfc_list.index('Seattle Seahawks') if 'Seattle Seahawks' in nfc_list else 0
    
    with col_a:
        st.markdown("**The Matchup**")
        afc_choice = st.selectbox("Select AFC Champion", afc_list, index=pats_idx)
        nfc_choice = st.selectbox("Select NFC Champion", nfc_list, index=sea_idx)
        
        afc_data = afc_teams_df[afc_teams_df['team_name'] == afc_choice].iloc[0]
        nfc_data = nfc_teams_df[nfc_teams_df['team_name'] == nfc_choice].iloc[0]

        afc_logo_url = afc_data['team_logo_wikipedia']
        nfc_logo_url = nfc_data['team_logo_wikipedia']
        
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
        weather_map = {"Clear/Dome": 1.0, "Rain/Wind": 0.85, "Snow": 0.75}
        weather = st.selectbox("Weather Conditions", list(weather_map.keys()))
        sim_count = st.select_slider("Simulations to Run", options=[1000, 5000, 10000], value=10000)

# --- NEW SIMULATION ENGINE ---
def run_simulation(iterations):
    # Base PPM (Points Per Minute)
    base_ppm = 0.45 * weather_map[weather]
    
    # LOGIC: A's scoring = (A's Offense Efficiency - B's Defense Efficiency)
    afc_eff = (afc_data['off_epa'] - nfc_data['def_epa']) * 5 + 1.0
    nfc_eff = (nfc_data['off_epa'] - afc_data['def_epa']) * 5 + 1.0
    
    afc_final_rate = base_ppm * afc_eff * strat_map[afc_strat] * (1 - inj_map[afc_inj_lvl])
    nfc_final_rate = base_ppm * nfc_eff * strat_map[nfc_strat] * (1 - inj_map[nfc_inj_lvl])
    
    afc_sim = score_afc + np.random.poisson(afc_final_rate * time_left, iterations)
    nfc_sim = score_nfc + np.random.poisson(nfc_final_rate * time_left, iterations)
    return afc_sim, nfc_sim

# --- UI RESULTS ---
if st.button(f"🚀 Run Super Bowl Simulation", use_container_width=True):
    afc_res, nfc_res = run_simulation(sim_count)
    
    afc_win_pct = (afc_res > nfc_res).mean() * 100
    nfc_win_pct = (nfc_res > afc_res).mean() * 100
    
    st.markdown("### 🏟️ Matchup Forecast")
    res_col1, res_col_vs, res_col2 = st.columns([2, 1, 2])
    
    with res_col1:
        st.markdown(f'<div style="text-align: right;"><img src="{nfc_logo_url}" width="100"><h2 style="margin:0;">{nfc_choice}</h2><p style="font-size:24px;"><b>Score: {nfc_res.mean():.1f}</b></p><p>Win: {nfc_win_pct:.1f}%</p></div>', unsafe_allow_html=True)
    with res_col_vs:
        st.markdown("<h1 style='text-align: center; margin-top: 60px; color: #888;'>VS</h1>", unsafe_allow_html=True)
    with res_col2:
        st.markdown(f'<div style="text-align: left;"><img src="{afc_logo_url}" width="100"><h2 style="margin:0;">{afc_choice}</h2><p style="font-size:24px;"><b>Score: {afc_res.mean():.1f}</b></p><p>Win: {afc_win_pct:.1f}%</p></div>', unsafe_allow_html=True)

    # Simple Progress Bar
    st.markdown(f"""<div style="width: 100%; background-color: #eee; border-radius: 10px; height: 15px; display: flex; overflow: hidden; margin-top: 25px;">
            <div style="width: {nfc_win_pct}%; background-color: #002244;"></div>
            <div style="width: {afc_win_pct}%; background-color: #C60C30;"></div>
        </div>""", unsafe_allow_html=True)
