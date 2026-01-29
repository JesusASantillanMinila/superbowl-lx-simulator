import streamlit as st
import numpy as np
import pandas as pd
import nflreadpy as nfl 
import altair as alt

# --- CONFIG & DATA LOAD ---
st.set_page_config(page_title="Super Bowl LX Simulator", layout="wide")

@st.cache_data
def load_nfl_metadata():
    teams = nfl.load_teams().to_pandas()
    teams = teams[teams['team_conf'].isin(['AFC', 'NFC'])]
    
    stats = nfl.load_team_stats([2025]).to_pandas()
    
    # 1. Calculate Offensive EPA
    off_epa = stats.groupby('team').agg({
        'passing_epa': 'sum',
        'rushing_epa': 'sum',
    }).reset_index()
    off_epa['off_epa_total'] = off_epa['passing_epa'] + off_epa['rushing_epa']

    # 2. Calculate Defensive EPA (Points Allowed Logic)
    def_epa = stats.groupby('opponent_team').agg({
        'passing_epa': 'sum',
        'rushing_epa': 'sum',
    }).reset_index()
    def_epa['def_epa_total'] = (def_epa['passing_epa'] + def_epa['rushing_epa']) * (-1)
    def_epa = def_epa.rename(columns={'opponent_team': 'team'})
    
    # Merge and calculate total performance
    team_performance = pd.merge(off_epa[['team', 'off_epa_total']], def_epa[['team', 'def_epa_total']], on='team')
    team_performance['total_epa_calc'] = team_performance['off_epa_total'] + team_performance['def_epa_total']
    
    df = pd.merge(
        teams[['team_abbr', 'team_conf', 'team_name', 'team_logo_wikipedia']], 
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
        afc_choice = st.selectbox("Select AFC Champion", afc_list, index=pats_idx)
        nfc_choice = st.selectbox("Select NFC Champion", nfc_list, index=sea_idx)

        afc_logo_url = afc_teams_df[afc_teams_df['team_name'] == afc_choice]['team_logo_wikipedia'].values[0]
        nfc_logo_url = nfc_teams_df[nfc_teams_df['team_name'] == nfc_choice]['team_logo_wikipedia'].values[0]
        
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
    
    afc_raw_wins = (afc_res > nfc_res)
    nfc_raw_wins = (nfc_res > afc_res)
    ties = (afc_res == nfc_res)
    
    total_momentum = afc_mod + nfc_mod
    afc_weight = afc_mod / total_momentum
    
    tie_breaker = np.random.random(sim_count) < afc_weight
    afc_final_wins = afc_raw_wins | (ties & tie_breaker)
    nfc_final_wins = nfc_raw_wins | (ties & ~tie_breaker)
    
    afc_win_pct = afc_final_wins.mean() * 100
    nfc_win_pct = nfc_final_wins.mean() * 100

    # --- CONSOLIDATED SECTION ---
    st.markdown("### 🏟️ Matchup Forecast")
    
    res_col1, res_col_vs, res_col2 = st.columns([2, 1, 2])
    
    with res_col1:
        st.markdown(f"""
            <div style="text-align: right;">
                <img src="{nfc_logo_url}" width="100">
                <h2 style="margin: 0;">{nfc_choice}</h2>
                <p style="font-size: 24px; margin: 0;"><b>Projected Score: {nfc_res.mean():.1f}</b></p>
                <p style="font-size: 18px; color: #555;">Win Probability: {nfc_win_pct:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
        
    with res_col_vs:
        st.markdown("<h1 style='text-align: center; margin-top: 60px; color: #888;'>VS</h1>", unsafe_allow_html=True)
        
    with res_col2:
        st.markdown(f"""
            <div style="text-align: left;">
                <img src="{afc_logo_url}" width="100">
                <h2 style="margin: 0;">{afc_choice}</h2>
                <p style="font-size: 24px; margin: 0;"><b>Projected Score: {afc_res.mean():.1f}</b></p>
                <p style="font-size: 18px; color: #555;">Win Probability: {afc_win_pct:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="width: 100%; background-color: #eee; border-radius: 10px; height: 15px; display: flex; overflow: hidden; border: 1px solid #ddd; margin-top: 25px; margin-bottom: 10px;">
            <div style="width: {nfc_win_pct}%; background-color: #C60C30;"></div>
            <div style="width: {afc_win_pct}%; background-color: #003366;"></div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Point Spread Distribution")
    st.info(f"Negative values favor {nfc_choice}, positive values favor {afc_choice}.")
    
    spreads = afc_res - nfc_res
    spread_data = pd.DataFrame({'Point Spread': spreads})
    
    chart = alt.Chart(spread_data).mark_bar().encode(
        x=alt.X('Point Spread:Q', bin=alt.Bin(maxbins=50), title='Point Spread'),
        y=alt.Y('count()', title='Frequency'),
        color=alt.condition(
            alt.datum['Point Spread'] > 0,
            alt.value('#003366'),  # AFC Color
            alt.value('#C60C30')   # NFC Color
        ),
        tooltip=['count()', 'Point Spread']
    ).properties(
        height=400
    ).configure_view(
        strokeOpacity=0 
    )

    st.altair_chart(chart, use_container_width=True)
