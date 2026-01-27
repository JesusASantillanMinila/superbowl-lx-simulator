import streamlit as st
import numpy as np
import pandas as pd
import nfl_data_py as nfl

# --- CONFIG & DATA LOAD ---
st.set_page_config(page_title="Super Bowl LX Simulator", layout="wide")

@st.cache_data
def load_nfl_metadata():
    # 1. Fetch team descriptions
    teams = nfl.import_team_desc()
    teams = teams[teams['team_conf'].isin(['AFC', 'NFC'])]
    
    # 2. Fetch win totals
    # Note: nfl_data_py win totals often use 'team_abbr' and 'win_total'
    offseason = nfl.import_win_totals([2025])
    
    # Troubleshooting/Normalization:
    # We need to ensure the columns match our merge logic
    if 'team_abbr' in offseason.columns and 'team' not in offseason.columns:
        offseason = offseason.rename(columns={'team_abbr': 'team'})
    
    if 'win_total' in offseason.columns and 'line' not in offseason.columns:
        offseason = offseason.rename(columns={'win_total': 'line'})

    # 3. Merge
    df = pd.merge(
        teams[['team_abbr', 'team_conf', 'team_name']], 
        offseason[['team', 'line']], 
        left_on='team_abbr', 
        right_on='team'
    )
    
    df['momentum'] = df['line'] / 8.5
    return df

data = load_nfl_metadata()

st.title("🏈 Super Bowl LX: Conference-Locked Simulator")

# --- SIDEBAR: TEAM SELECTION ---
st.sidebar.header("🏆 The Matchup")

# Filter dataframes by conference
afc_teams_df = data[data['team_conf'] == 'AFC']
nfc_teams_df = data[data['team_conf'] == 'NFC']

# Create selectboxes using the filtered lists
afc_choice = st.sidebar.selectbox("Select AFC Champion", afc_teams_df['team_name'])
nfc_choice = st.sidebar.selectbox("Select NFC Champion", nfc_teams_df['team_name'])

# Extract the momentum modifiers for the selected teams
afc_mod = afc_teams_df[afc_teams_df['team_name'] == afc_choice]['momentum'].values[0]
nfc_mod = nfc_teams_df[nfc_teams_df['team_name'] == nfc_choice]['momentum'].values[0]

# --- ADDITIONAL INPUTS ---
st.sidebar.divider()
st.sidebar.header("Game Context")
score_afc = st.sidebar.number_input(f"{afc_choice} Score", 0, 100, 0)
score_nfc = st.sidebar.number_input(f"{nfc_choice} Score", 0, 100, 0)
time_left = st.sidebar.slider("Total Game Time Remaining (Mins)", 1, 60, 60)

# Injury logic from previous version
st.sidebar.header("Injury Impact")
off_inj = st.sidebar.select_slider("Offense Injury Level", options=["None", "Role", "Starter", "Star", "Elite"])
def_inj = st.sidebar.select_slider("Defense Injury Level", options=["None", "Role", "Starter", "Star", "Elite"])

# Mapping for the simulation math
inj_map = {"None": 0.0, "Role": 0.03, "Starter": 0.07, "Star": 0.15, "Elite": 0.30}

# --- SIMULATION ENGINE ---
def run_simulation():
    # Base scoring logic
    iterations = 10000
    base_ppm = 0.45 # Points Per Minute average
    
    # Multipliers
    afc_final_rate = base_ppm * afc_mod * (1 - inj_map[off_inj] + inj_map[def_inj])
    nfc_final_rate = base_ppm * nfc_mod * (1 - inj_map[off_inj] + inj_map[def_inj])
    
    # Run Monte Carlo
    afc_sim = score_afc + np.random.poisson(afc_final_rate * time_left, iterations)
    nfc_sim = score_nfc + np.random.poisson(nfc_final_rate * time_left, iterations)
    
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

    # Detailed Histogram
    st.divider()
    st.subheader("Point Spread Distribution")
    spreads = afc_res - nfc_res
    st.bar_chart(pd.Series(spreads).value_counts().sort_index())
