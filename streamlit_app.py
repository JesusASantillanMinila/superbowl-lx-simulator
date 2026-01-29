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
    
    # Load 2025 Team Stats
    stats = nfl.load_team_stats([2025]).to_pandas()
    
    # 1. Calculate OFFENSIVE EPA (by team)
    offense = stats.groupby('team').agg({
        'passing_epa': 'mean',
        'rushing_epa': 'mean'
    }).rename(columns={'passing_epa': 'off_pass_epa', 'rushing_epa': 'off_rush_epa'})

    # 2. Calculate DEFENSIVE EPA (by opponent)
    # What others did AGAINST this team = this team's defense
    defense = stats.groupby('opponent').agg({
        'passing_epa': 'mean',
        'rushing_epa': 'mean'
    }).rename(columns={'passing_epa': 'def_pass_epa', 'rushing_epa': 'def_rush_epa'})
    
    # Combine stats
    perf = pd.concat([offense, defense], axis=1).reset_index().rename(columns={'index': 'team_abbr'})

    # Net EPA Calculation (Offense - Defense Allowed)
    perf['off_total'] = perf['off_pass_epa'] + perf['off_rush_epa']
    perf['def_total'] = perf['def_pass_epa'] + perf['def_rush_epa']
    perf['net_epa'] = perf['off_total'] - perf['def_total']
    
    df = pd.merge(
        teams[['team_abbr', 'team_conf', 'team_name', 'team_logo_wikipedia']], 
        perf, 
        on='team_abbr'
    )
    
    # Normalize Momentum (0.7 to 1.3 range)
    e_min, e_max = df['net_epa'].min(), df['net_epa'].max()
    df['momentum'] = ((df['net_epa'] - e_min) / (e_max - e_min)) * 0.6 + 0.7
    
    return df

data = load_nfl_metadata()

st.title(" 🏈 Super Bowl LX: Advanced Simulator")

# --- UI CONTROLS ---
with st.expander("🛠️ Simulation Settings", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    
    afc_df = data[data['team_conf'] == 'AFC'].sort_values('team_name')
    nfc_df = data[data['team_conf'] == 'NFC'].sort_values('team_name')

    pats_idx = afc_df['team_name'].tolist().index('New England Patriots') if 'New England Patriots' in afc_df['team_name'].values else 0
    sea_idx = nfc_df['team_name'].tolist().index('Seattle Seahawks') if 'Seattle Seahawks' in nfc_df['team_name'].values else 0
    
    with col_a:
        afc_choice = st.selectbox("AFC Team", afc_df['team_name'], index=pats_idx)
        nfc_choice = st.selectbox("NFC Team", nfc_df['team_name'], index=sea_idx)
        
        afc_row = afc_df[afc_df['team_name'] == afc_choice].iloc[0]
        nfc_row = nfc_df[nfc_df['team_name'] == nfc_choice].iloc[0]
        
        st.divider()
        strat_map = {"Defensive": 0.85, "Balanced": 1.0, "Offensive": 1.15}
        afc_strat = st.select_slider(f"{afc_choice} Strategy", options=list(strat_map.keys()), value="Balanced")
        nfc_strat = st.select_slider(f"{nfc_choice} Strategy", options=list(strat_map.keys()), value="Balanced")

    with col_b:
        score_afc = st.number_input(f"{afc_choice} Score", 0, 100, 0)
        score_nfc = st.number_input(f"{nfc_choice} Score", 0, 100, 0)
        time_left = st.slider("Minutes Left", 1, 60, 60)
    
    with col_c:
        inj_map = {"None": 0.0, "Starter": 0.07, "Star": 0.15, "Elite": 0.25}
        afc_inj = st.select_slider(f"{afc_choice} Injury Hit", options=list(inj_map.keys()))
        nfc_inj = st.select_slider(f"{nfc_choice} Injury Hit", options=list(inj_map.keys()))
        weather_map = {"Clear": 1.0, "Rain": 0.85, "Snow": 0.75}
        weather = st.selectbox("Weather", list(weather_map.keys()))
        sim_count = st.select_slider("Sims", options=[1000, 5000, 10000], value=10000)

# --- SIMULATION ENGINE ---
def run_simulation(n):
    base_ppm = 0.45 * weather_map[weather]
    
    # Scoring Rate = (My Offense - Your Defense) scaled to PPM
    # We multiply by 5 to amplify the EPA differences into realistic point spreads
    afc_rate = base_ppm * (1 + (afc_row['off_total'] - nfc_row['def_total']) * 5) * strat_map[afc_strat] * (1 - inj_map[afc_inj])
    nfc_rate = base_ppm * (1 + (nfc_row['off_total'] - afc_row['def_total']) * 5) * strat_map[nfc_strat] * (1 - inj_map[nfc_inj])
    
    # Poisson prevents negative scores and models football scoring frequency
    afc_sim = score_afc + np.random.poisson(max(0.1, afc_rate) * time_left, n)
    nfc_sim = score_nfc + np.random.poisson(max(0.1, nfc_rate) * time_left, n)
    return afc_sim, nfc_sim

if st.button("🚀 Run Super Bowl Simulation", use_container_width=True):
    afc_res, nfc_res = run_simulation(sim_count)
    afc_win_pct = (afc_res > nfc_res).mean() * 100
    nfc_win_pct = (nfc_res > afc_res).mean() * 100

    # Display Results
    st.markdown("### 🏟️ Matchup Forecast")
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        st.markdown(f"<div style='text-align:right;'><img src='{nfc_row['team_logo_wikipedia']}' width='100'><h2>{nfc_choice}</h2><p><b>Proj Score: {nfc_res.mean():.1f}</b></p><p>Win: {nfc_win_pct:.1f}%</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<h1 style='text-align:center; margin-top:60px;'>VS</h1>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div style='text-align:left;'><img src='{afc_row['team_logo_wikipedia']}' width='100'><h2>{afc_choice}</h2><p><b>Proj Score: {afc_res.mean():.1f}</b></p><p>Win: {afc_win_pct:.1f}%</p></div>", unsafe_allow_html=True)

    # Spread Chart
    spreads = afc_res - nfc_res
    chart = alt.Chart(pd.DataFrame({'Spread': spreads})).mark_bar().encode(
        x=alt.X('Spread:Q', bin=alt.Bin(maxbins=40)),
        y='count()',
        color=alt.condition(alt.datum.Spread > 0, alt.value("#002244"), alt.value("#69BE28"))
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)
