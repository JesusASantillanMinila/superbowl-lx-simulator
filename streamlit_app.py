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

# --- SIDEBAR: TEAM SELECTION ---
st.sidebar.header("🏆 The Matchup")

afc_teams_df = data[data['team_conf'] == 'AFC']
nfc_teams_df = data[data['team_conf'] == 'NFC']

afc_choice = st.sidebar.selectbox("Select AFC Champion", afc_teams_df['team_name'])
nfc_choice = st.sidebar.selectbox("Select NFC Champion", nfc_teams_df['team_name'])

afc_mod = afc_teams_df[afc_teams_df['team_name'] == afc_choice]['momentum'].values[0]
nfc_mod = nfc_teams_df[nfc_teams_df['team_name'] == nfc_choice]['momentum'].values[0]

# --- SIDEBAR: GAME CONTEXT ---
st.sidebar.divider()
st.sidebar.header("Game Environment")

# Weather Modifiers
weather = st.sidebar.selectbox("Weather Conditions", ["Clear/Dome", "Rain/Wind", "Snow"])
weather_map = {"Clear/Dome": 1.0, "Rain/Wind": 0.85, "Snow": 0.75}

# Coaching Strategy Modifiers
strat_map = {"Defensive": 0.90, "Balanced": 1.0, "Offensive": 1.10}
afc_strat = st.sidebar.select_slider(f"{afc_choice} Strategy", options=["Defensive", "Balanced", "Offensive"], value="Balanced")
nfc_strat = st.sidebar.select_slider(f"{nfc_choice} Strategy", options=["Defensive", "Balanced", "Offensive"], value="Balanced")

# --- SIDEBAR: TEAM-BASED INJURIES ---
st.sidebar.divider()
st.sidebar.header("Injury Impact")
inj_map = {"None": 0.0, "Role": 0.03, "Starter": 0.07, "Star": 0.15, "Elite": 0.30}

afc_inj_lvl = st.sidebar.select_slider(f"{afc_choice} Injuries", options=["None", "Role", "Starter", "Star", "Elite"])
nfc_inj_lvl = st.sidebar.select_slider(f"{nfc_choice} Injuries", options=["None", "Role", "Starter", "Star", "Elite"])

# Current Game State
st.sidebar.divider()
score_afc = st.sidebar.number_input(f"{afc_choice} Current Score", 0, 100, 0)
score_nfc = st.sidebar.number_input(f"{nfc_choice} Current Score", 0, 100, 0)
time_left = st.sidebar.slider("Minutes Remaining", 1, 60, 60)

# --- SIMULATION ENGINE ---
def run_simulation():
    iterations = 10000
    # Base points per minute (League Average)
    base_ppm = 0.45 * weather_map[weather]
    
    # AFC Rate: Base * Team Strength * Coach Strategy * (1 - Team's Injuries)
    afc_final_rate = base_ppm * afc_mod * strat_map[afc_strat] * (1 - inj_map[afc_inj_lvl])
    
    # NFC Rate: Base * Team Strength * Coach Strategy * (1 - Team's Injuries)
    nfc_final_rate = base_ppm * nfc_mod * strat_map[nfc_strat] * (1 - inj_map[nfc_inj_lvl])
    
    afc_sim = score_afc + np.random.poisson(afc_final_rate * time_left, iterations)
    nfc_sim = score_nfc + np.random.poisson(nfc_final_rate * time_left, iterations)
    
    return afc_sim, nfc_sim

# --- UI RESULTS ---
if st.button(f"🚀 Run Super Bowl Simulation"):
    afc_res, nfc_res = run_simulation()
    
    col1, col2 = st.columns(2)
    afc_win_pct = (afc_res > nfc_res).mean() * 100
    
    with col1:
        st.subheader("Win Probability")
        st.write(f"**{afc_choice}:** {afc_win_pct:.1f}%")
        st.write(f"**{nfc_choice}:** {(100 - afc_win_pct):.1f}%")
        st.progress(afc_win_pct / 100)

    with col2:
        st.subheader("Projected Final Score")
        st.metric(afc_choice, f"{afc_res.mean():.1f}")
        st.metric(nfc_choice, f"{nfc_res.mean():.1f}")

    st.divider()
    st.subheader("Point Spread Distribution")
    spreads = afc_res - nfc_res
    st.bar_chart(pd.Series(spreads).value_counts().sort_index())
