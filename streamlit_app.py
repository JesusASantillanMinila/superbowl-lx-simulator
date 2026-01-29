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
    
    # Load 2025 season stats
    # We need to calculate both Offensive EPA and Defensive EPA
    stats = nfl.load_team_stats([2025]).to_pandas()
    
    # Offensive Stats (EPA gained)
    offense = stats.groupby('team').agg({
        'passing_epa': 'sum',
        'rushing_epa': 'sum',
    }).rename(columns={'passing_epa': 'off_pass_epa', 'rushing_epa': 'off_rush_epa'})

    # Defensive Stats (EPA allowed - this is usually found by looking at opponents)
    # Note: In nflreadpy team_stats, 'opp_passing_epa' is what the defense allowed
    defense = stats.groupby('team').agg({
        'opp_passing_epa': 'sum',
        'opp_rushing_epa': 'sum',
    }).rename(columns={'opp_passing_epa': 'def_pass_epa', 'opp_rushing_epa': 'def_rush_epa'})

    # Combine
    df_stats = pd.concat([offense, defense], axis=1).reset_index()
    
    # Net EPA is the true measure of team quality
    # We subtract defensive EPA because 'allowing' EPA is bad
    df_stats['net_epa'] = (df_stats['off_pass_epa'] + df_stats['off_rush_epa']) - \
                          (df_stats['def_pass_epa'] + df_stats['def_rush_epa'])

    df = pd.merge(
        teams[['team_abbr', 'team_conf', 'team_name', 'team_logo_wikipedia']], 
        df_stats, 
        left_on='team_abbr', 
        right_on='team'
    )
    
    # Normalize Net EPA to a "Power Rating" (0.7 to 1.3 range)
    epa_min = df['net_epa'].min()
    epa_max = df['net_epa'].max()
    df['power_rating'] = ((df['net_epa'] - epa_min) / (epa_max - epa_min)) * 0.6 + 0.7
    
    return df

data = load_nfl_metadata()

st.title(" 🏈 Super Bowl LX: Advanced Simulator")

# --- 1) SETTINGS ---
with st.expander("🛠️ Simulation Settings & Team Selection", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    
    afc_teams_df = data[data['team_conf'] == 'AFC'].sort_values('team_name')
    nfc_teams_df = data[data['team_conf'] == 'NFC'].sort_values('team_name')

    afc_list = afc_teams_df['team_name'].tolist()
    nfc_list = nfc_teams_df['team_name'].tolist()
    
    try: pats_idx = afc_list.index('New England Patriots')
    except ValueError: pats_idx = 0
    try: sea_idx = nfc_list.index('Seattle Seahawks')
    except ValueError: sea_idx = 0
    
    with col_a:
        afc_choice = st.selectbox("Select AFC Champion", afc_list, index=pats_idx)
        nfc_choice = st.selectbox("Select NFC Champion", nfc_list, index=sea_idx)
        afc_logo_url = afc_teams_df[afc_teams_df['team_name'] == afc_choice]['team_logo_wikipedia'].values[0]
        nfc_logo_url = nfc_teams_df[nfc_teams_df['team_name'] == nfc_choice]['team_logo_wikipedia'].values[0]
        
        st.divider()
        strat_map = {"Defensive": 0.85, "Balanced": 1.0, "Offensive": 1.15}
        afc_strat = st.select_slider(f"{afc_choice} Strategy", options=["Defensive", "Balanced", "Offensive"], value="Balanced")
        nfc_strat = st.select_slider(f"{nfc_choice} Strategy", options=["Defensive", "Balanced", "Offensive"], value="Balanced")

    with col_b:
        score_afc = st.number_input(f"{afc_choice} Score", 0, 100, 0)
        score_nfc = st.number_input(f"{nfc_choice} Score", 0, 100, 0)
        time_left = st.slider("Minutes Remaining", 1, 60, 60)
    
    with col_c:
        inj_map = {"None": 0.0, "Role": 0.03, "Starter": 0.07, "Star": 0.15, "Elite": 0.30}
        afc_inj_lvl = st.select_slider(f"{afc_choice} Injuries", options=["None", "Role", "Starter", "Star", "Elite"])
        nfc_inj_lvl = st.select_slider(f"{nfc_choice} Injuries", options=["None", "Role", "Starter", "Star", "Elite"])
        weather_map = {"Clear/Dome": 1.0, "Rain/Wind": 0.88, "Snow": 0.80}
        weather = st.selectbox("Weather", ["Clear/Dome", "Rain/Wind", "Snow"])
        sim_count = st.select_slider("Simulations", options=[1000, 10000, 50000], value=10000)

afc_power = afc_teams_df[afc_teams_df['team_name'] == afc_choice]['power_rating'].values[0]
nfc_power = nfc_teams_df[nfc_teams_df['team_name'] == nfc_choice]['power_rating'].values[0]

# --- SIMULATION ENGINE ---
def run_simulation(iterations):
    # Base points per minute (~24 pts per game / 60)
    base_ppm = 0.40 * weather_map[weather]
    
    # Rate = Base * (My Offense/Power) * (Opponent's Defense/Inverse)
    # This ensures Seattle's elite defense actually suppresses the Patriots' score
    afc_final_rate = base_ppm * (afc_power / nfc_power) * strat_map[afc_strat] * (1 - inj_map[afc_inj_lvl])
    nfc_final_rate = base_ppm * (nfc_power / afc_power) * strat_map[nfc_strat] * (1 - inj_map[nfc_inj_lvl])
    
    afc_sim = score_afc + np.random.poisson(afc_final_rate * time_left, iterations)
    nfc_sim = score_nfc + np.random.poisson(nfc_final_rate * time_left, iterations)
    return afc_sim, nfc_sim

# --- UI RESULTS ---
if st.button(f"🚀 Run Super Bowl LX Simulation", use_container_width=True):
    afc_res, nfc_res = run_simulation(sim_count)
    
    afc_wins = (afc_res > nfc_res).mean()
    nfc_wins = (nfc_res > afc_res).mean()
    
    st.markdown("### 🏟️ Matchup Forecast")
    c1, c_vs, c2 = st.columns([2, 1, 2])
    
    with c1:
        st.markdown(f"<div style='text-align: right;'><img src='{nfc_logo_url}' width='100'><h2>{nfc_choice}</h2><p style='font-size: 24px;'><b>Proj: {nfc_res.mean():.1f}</b></p><p>Win Prob: {nfc_wins*100:.1;f}%</p></div>", unsafe_allow_html=True)
    with c_vs:
        st.markdown("<h1 style='text-align: center; margin-top: 60px; color: #888;'>VS</h1>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div style='text-align: left;'><img src='{afc_logo_url}' width='100'><h2>{afc_choice}</h2><p style='font-size: 24px;'><b>Proj: {afc_res.mean():.1f}</b></p><p>Win Prob: {afc_wins*100:.1f}%</p></div>", unsafe_allow_html=True)

    # Visualization
    spreads = afc_res - nfc_res
    chart_df = pd.DataFrame({'Spread': spreads})
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X('Spread:Q', bin=alt.Bin(maxbins=40), title='Point Spread (Negative = Seahawks Favor)'),
        y='count()',
        color=alt.condition(alt.datum.Spread > 0, alt.value('#003366'), alt.value('#C60C30'))
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)
