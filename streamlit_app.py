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
    
    # Load 2025 Stats
    stats = nfl.load_team_stats([2025]).to_pandas()
    
    # 1. Offensive EPA (Points Scored Logic)
    off_epa = stats.groupby('team').agg({
        'passing_epa': 'sum',
        'rushing_epa': 'sum',
    }).reset_index()
    off_epa['off_epa_total'] = off_epa['passing_epa'] + off_epa['rushing_epa']

    # 2. Defensive EPA (Points Allowed Logic - Inverse for Strength)
    def_epa = stats.groupby('opponent_team').agg({
        'passing_epa': 'sum',
        'rushing_epa': 'sum',
    }).reset_index()
    # Note: High defensive EPA allowed is bad, so we multiply by -1 
    # so that a "higher" number represents a "better" defense.
    def_epa['def_epa_total'] = (def_epa['passing_epa'] + def_epa['rushing_epa']) * (-1)
    def_epa = def_epa.rename(columns={'opponent_team': 'team'})
    
    # Merge Stats
    team_performance = pd.merge(off_epa[['team', 'off_epa_total']], 
                                def_epa[['team', 'def_epa_total']], on='team')

    df = pd.merge(
        teams[['team_abbr', 'team_conf', 'team_name', 'team_logo_wikipedia']], 
        team_performance, 
        left_on='team_abbr', 
        right_on='team'
    )
    
    # Normalize Momentum (Combining Offense and Defense)
    df['total_epa_calc'] = df['off_epa_total'] + df['def_epa_total']
    epa_min = df['total_epa_calc'].min()
    epa_max = df['total_epa_calc'].max()
    df['momentum'] = ((df['total_epa_calc'] - epa_min) / (epa_max - epa_min)) * 0.6 + 0.7
    
    return df

data = load_nfl_metadata()

st.title("🏈 Super Bowl LX: Advanced Simulator")
st.caption("Now featuring Defensive EPA and Opponent-Adjusted Scoring Logic")

# --- 1) SETTINGS & SELECTION ---
with st.expander("🛠️ Simulation Settings & Team Selection", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    
    afc_teams_df = data[data['team_conf'] == 'AFC'].sort_values('team_name')
    nfc_teams_df = data[data['team_conf'] == 'NFC'].sort_values('team_name')

    afc_list = afc_teams_df['team_name'].tolist()
    nfc_list = nfc_teams_df['team_name'].tolist()
    
    # Default to Super Bowl LX favorites
    pats_idx = afc_list.index('New England Patriots') if 'New England Patriots' in afc_list else 0
    sea_idx = nfc_list.index('Seattle Seahawks') if 'Seattle Seahawks' in nfc_list else 0
    
    with col_a:
        st.markdown("**The Matchup**")
        afc_choice = st.selectbox("Select AFC Champion", afc_list, index=pats_idx)
        nfc_choice = st.selectbox("Select NFC Champion", nfc_list, index=sea_idx)

        # --- EXTRACT TEAM DATA ---
        afc_data = afc_teams_df[afc_teams_df['team_name'] == afc_choice].iloc[0]
        nfc_data = nfc_teams_df[nfc_teams_df['team_name'] == nfc_choice].iloc[0]
        
        st.divider()
        st.markdown("**Strategy**")
        strat_map = {"Defensive": 0.85, "Balanced": 1.0, "Offensive": 1.15}
        afc_strat = st.select_slider(f"{afc_choice} Focus", options=["Defensive", "Balanced", "Offensive"], value="Balanced")
        nfc_strat = st.select_slider(f"{nfc_choice} Focus", options=["Defensive", "Balanced", "Offensive"], value="Balanced")

    with col_b:
        st.markdown("**Game State**")
        score_afc = st.number_input(f"{afc_choice} Score", 0, 100, 0)
        score_nfc = st.number_input(f"{nfc_choice} Score", 0, 100, 0)
        time_left = st.slider("Minutes Remaining", 1, 60, 60)
    
    with col_c:
        st.markdown("**Modifiers**")
        inj_map = {"None": 0.0, "Star": 0.10, "Elite": 0.25}
        afc_inj = st.select_slider(f"{afc_choice} Injuries", options=["None", "Star", "Elite"])
        nfc_inj = st.select_slider(f"{nfc_choice} Injuries", options=["None", "Star", "Elite"])

        st.divider()
        weather = st.selectbox("Weather", ["Dome", "Wind/Rain", "Snow"])
        weather_map = {"Dome": 1.0, "Wind/Rain": 0.85, "Snow": 0.70}
        sim_count = st.select_slider("Iterations", options=[1000, 10000, 50000], value=10000)

# --- SIMULATION ENGINE ---
def run_simulation(iterations):
    # base_ppm is the league average points per minute (~0.45)
    base_ppm = 0.45 * weather_map[weather]
    
    # LOGIC: Team A's scoring rate depends on (Team A Offense + Team B Defense)
    # We use the momentum/EPA as a multiplier against the league base.
    
    # AFC Scoring Rate = Base * AFC Offense * NFC Defense Influence
    # Since we normalized 'momentum' already, we can use the individual components
    # or the weighted average. Here we use the specific matchups:
    
    def get_matchup_mod(off_team, def_team):
        # Combining Offense of one and Defense of another
        # We normalize these relative to the mean to get a multiplier
        off_mod = 1 + (off_team['off_epa_total'] / data['off_epa_total'].abs().max()) * 0.2
        def_mod = 1 + (def_team['def_epa_total'] / data['def_epa_total'].abs().max()) * 0.2
        return (off_mod + def_mod) / 2

    afc_matchup_mod = get_matchup_mod(afc_data, nfc_data)
    nfc_matchup_mod = get_matchup_mod(nfc_data, afc_data)

    afc_final_rate = base_ppm * afc_matchup_mod * strat_map[afc_strat] * (1 - inj_map[afc_inj_lvl])
    nfc_final_rate = base_ppm * nfc_matchup_mod * strat_map[nfc_strat] * (1 - inj_map[nfc_inj_lvl])
    
    afc_sim = score_afc + np.random.poisson(afc_final_rate * time_left, iterations)
    nfc_sim = score_nfc + np.random.poisson(nfc_final_rate * time_left, iterations)
    return afc_sim, nfc_sim
    
# --- RESULTS ---
if st.button(f"🚀 Simulate Super Bowl LX", use_container_width=True):
    afc_res, nfc_res = run_simulation(sim_count)
    
    # Calculate Wins
    afc_wins = (afc_res > nfc_res).mean()
    nfc_wins = (nfc_res > afc_res).mean()
    
    st.markdown("### 🏟️ Simulation Results")
    res_col1, res_col_vs, res_col2 = st.columns([2, 1, 2])
    
    with res_col1:
        st.markdown(f"""<div style="text-align: right;">
            <img src="{nfc_data['team_logo_wikipedia']}" width="100">
            <h2>{nfc_choice}</h2>
            <p style="font-size: 24px;"><b>Proj. Score: {nfc_res.mean():.1f}</b></p>
            <p>Win Prob: {nfc_wins*100:.1f}%</p>
        </div>""", unsafe_allow_html=True)
        
    with res_col_vs:
        st.markdown("<h1 style='text-align: center; margin-top: 50px;'>VS</h1>", unsafe_allow_html=True)
        
    with res_col2:
        st.markdown(f"""<div style="text-align: left;">
            <img src="{afc_data['team_logo_wikipedia']}" width="100">
            <h2>{afc_choice}</h2>
            <p style="font-size: 24px;"><b>Proj. Score: {afc_res.mean():.1f}</b></p>
            <p>Win Prob: {afc_wins*100:.1f}%</p>
        </div>""", unsafe_allow_html=True)

    # Win Probability Bar
    st.markdown(f"""
        <div style="width: 100%; background-color: #eee; height: 15px; display: flex; border-radius: 10px; overflow: hidden; margin: 20px 0;">
            <div style="width: {nfc_wins*100}%; background-color: #002244;"></div>
            <div style="width: {afc_wins*100}%; background-color: #C60C30;"></div>
        </div>
    """, unsafe_allow_html=True)

    # Distribution Chart
    st.divider()
    spreads = afc_res - nfc_res
    spread_df = pd.DataFrame({'Spread': spreads})
    
    chart = alt.Chart(spread_df).mark_bar().encode(
        x=alt.X('Spread:Q', bin=alt.Bin(maxbins=40), title='Point Spread (Negative = Seahawks Win)'),
        y='count()',
        color=alt.condition(alt.datum.Spread > 0, alt.value('#C60C30'), alt.value('#002244'))
    ).properties(height=300)
    
    st.altair_chart(chart, use_container_width=True)
