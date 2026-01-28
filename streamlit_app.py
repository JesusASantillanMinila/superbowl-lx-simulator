import streamlit as st
import numpy as np
import pandas as pd
import nflreadpy as nfl

# --- CONFIG & DATA LOAD ---
st.set_page_config(page_title="Super Bowl LX Simulator", layout="wide")

@st.cache_data
def load_nfl_metadata():
    # 1. Fetch team metadata (equivalent to import_team_desc)
    # nflreadpy returns pandas DataFrames by default
    teams = nfl.load_teams()
    teams = teams[teams['team_conf'].isin(['AFC', 'NFC'])]
    
    # 2. Fetch historical team stats for 2024 as a proxy for 2025/2026 performance
    # load_team_stats provides comprehensive seasonal data
    stats = nfl.load_team_stats(seasons=[2024])
    
    # Calculate a strength metric (Point Differential per game)
    # We'll use this to calculate our 'momentum' factor
    stats['pt_diff'] = (stats['score'] - stats['opp_score'])
    
    # Aggregate stats by team
    team_strength = stats.groupby('team').agg({
        'pt_diff': 'mean'
    }).reset_index()

    # 3. Merge metadata with historical performance
    df = pd.merge(
        teams[['team_abbr', 'team_conf', 'team_name']], 
        team_strength, 
        left_on='team_abbr', 
        right_on='team'
    )
    
    # Calculate Momentum Factor (Normalized: Average team = 1.0)
    # We shift the point diff so the worst team isn't 0, then normalize
    min_diff = df['pt_diff'].min()
    df['momentum'] = (df['pt_diff'] - min_diff + 5) / (df['pt_diff'].mean() - min_diff + 5)
    
    return df

data = load_nfl_metadata()

st.title("🏈 Super Bowl LX: Conference-Locked Simulator")

# --- SIDEBAR: TEAM SELECTION ---
st.sidebar.header("🏆 The Matchup")

afc_teams_df = data[data['team_conf'] == 'AFC']
nfc_teams_df = data[data['team_conf'] == 'NFC']

afc_choice = st.sidebar.selectbox("Select AFC Champion", afc_teams_df['team_name'])
nfc_choice = st.sidebar.selectbox("Select NFC Champion", nfc_teams_df['team_name'])

afc_mod = afc_teams_df[afc_teams_df['team_name'] == afc_choice]['momentum'].values[0]
nfc_mod = nfc_teams_df[nfc_teams_df['team_name'] == nfc_choice]['momentum'].values[0]

# --- ADDITIONAL INPUTS ---
st.sidebar.divider()
st.sidebar.header("Game Context")
score_afc = st.sidebar.number_input(f"{afc_choice} Score", 0, 100, 0)
score_nfc = st.sidebar.number_input(f"{nfc_choice} Score", 0, 100, 0)
time_left = st.sidebar.slider("Total Game Time Remaining (Mins)", 1, 60, 60)

st.sidebar.header("Injury Impact")
off_inj = st.sidebar.select_slider("Offense Injury Level", options=["None", "Role", "Starter", "Star", "Elite"])
def_inj = st.sidebar.select_slider("Defense Injury Level", options=["None", "Role", "Starter", "Star", "Elite"])

inj_map = {"None": 0.0, "Role": 0.03, "Starter": 0.07, "Star": 0.15, "Elite": 0.30}

# --- SIMULATION ENGINE ---
def run_simulation():
    iterations = 10000
    base_ppm = 0.45 
    
    # Multipliers apply to the scoring rates
    afc_final_rate = base_ppm * afc_mod * (1 - inj_map[off_inj] + inj_map[def_inj])
    nfc_final_rate = base_ppm * nfc_mod * (1 - inj_map[off_inj] + inj_map[def_inj])
    
    # Monte Carlo simulation using Poisson distribution
    afc_sim = score_afc + np.random.poisson(max(0, afc_final_rate * time_left), iterations)
    nfc_sim = score_nfc + np.random.poisson(max(0, nfc_final_rate * time_left), iterations)
    
    return afc_sim, nfc_sim

# --- UI RESULTS ---
if st.button(f"Simulate {afc_choice} vs {nfc_choice}"):
    afc_res, nfc_res = run_simulation()
    
    col1, col2 = st.columns(2)
    afc_win_pct = (afc_res > nfc_res).mean() * 100
    
    with col1:
        st.subheader("Win Probability")
        st.write(f"**{afc_choice}:** {afc_win_pct:.1f}%")
        st.write(f"**{nfc_choice}:** {(100 - afc_win_pct):.1f}%")
        st.progress(afc_win_pct / 100)

    with col2:
        st.subheader("Average Final Score")
        st.metric(afc_choice, int(afc_res.mean()))
        st.metric(nfc_choice, int(nfc_res.mean()))

    st.divider()
    st.subheader("Point Spread Distribution")
    spreads = afc_res - nfc_res
    st.bar_chart(pd.Series(spreads).value_counts().sort_index())
