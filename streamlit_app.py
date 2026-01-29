import streamlit as st
import numpy as np
import pandas as pd
import nflreadpy as nfl 
import altair as alt

# --- CONFIG & DATA LOAD ---
st.set_page_config(page_title="Super Bowl LX Simulator", layout="wide")

@st.cache_data
def load_nfl_metadata():
    # 1. Load Teams
    teams = nfl.load_teams().to_pandas()
    teams = teams[teams['team_conf'].isin(['AFC', 'NFC'])]
    
    # 2. Load Team Stats (Game Level)
    # nflreadpy load_team_stats returns game-by-team rows
    stats = nfl.load_team_stats([2025]).to_pandas()
    
    # To get defensive EPA, we look at the EPA of the team's opponents
    # We'll create a summary of EPA gained (Offense) and EPA allowed (Defense)
    
    # Offensive EPA: Sum of EPA gained by the team
    offense = stats.groupby('team').agg({
        'passing_epa': 'sum',
        'rushing_epa': 'sum'
    }).reset_index()
    offense.columns = ['team', 'off_pass_epa', 'off_rush_epa']

    # Defensive EPA: This is the EPA gained by the OPPONENTS in those games
    # We find this by looking at the 'opponent' and their offensive EPA
    defense = stats.groupby('opponent').agg({
        'passing_epa': 'sum',
        'rushing_epa': 'sum'
    }).reset_index()
    defense.columns = ['team', 'def_pass_epa', 'def_rush_epa']

    # Merge Offense and Defense
    team_perf = pd.merge(offense, defense, on='team')
    
    # Net EPA = (Offense) - (Defense)
    # A high Net EPA means you score a lot AND stop the other team
    team_perf['net_epa'] = (team_perf['off_pass_epa'] + team_perf['off_rush_epa']) - \
                           (team_perf['def_pass_epa'] + team_perf['def_rush_epa'])
    
    df = pd.merge(
        teams[['team_abbr', 'team_conf', 'team_name', 'team_logo_wikipedia']], 
        team_perf, 
        left_on='team_abbr', 
        right_on='team'
    )
    
    # Create a Power Rating (normalized 0.7 to 1.3)
    # This acts as a multiplier for scoring probability
    epa_min, epa_max = df['net_epa'].min(), df['net_epa'].max()
    df['power_rating'] = ((df['net_epa'] - epa_min) / (epa_max - epa_min)) * 0.6 + 0.7
    
    return df

data = load_nfl_metadata()

st.title(" 🏈 Super Bowl LX: Advanced Simulator")

# --- UI SETTINGS ---
with st.expander("🛠️ Simulation Settings & Team Selection", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    
    afc_teams = data[data['team_conf'] == 'AFC'].sort_values('team_name')
    nfc_teams = data[data['team_conf'] == 'NFC'].sort_values('team_name')
    
    with col_a:
        afc_choice = st.selectbox("AFC Team", afc_teams['team_name'].tolist(), index=afc_teams['team_name'].tolist().index('New England Patriots') if 'New England Patriots' in afc_teams['team_name'].tolist() else 0)
        nfc_choice = st.selectbox("NFC Team", nfc_teams['team_name'].tolist(), index=nfc_teams['team_name'].tolist().index('Seattle Seahawks') if 'Seattle Seahawks' in nfc_teams['team_name'].tolist() else 0)
        
        afc_logo = afc_teams[afc_teams['team_name'] == afc_choice]['team_logo_wikipedia'].values[0]
        nfc_logo = nfc_teams[nfc_teams['team_name'] == nfc_choice]['team_logo_wikipedia'].values[0]

        st.divider()
        strat_map = {"Defensive": 0.85, "Balanced": 1.0, "Offensive": 1.15}
        afc_strat = st.select_slider(f"{afc_choice} Style", options=["Defensive", "Balanced", "Offensive"], value="Balanced")
        nfc_strat = st.select_slider(f"{nfc_choice} Style", options=["Defensive", "Balanced", "Offensive"], value="Balanced")

    with col_b:
        score_afc = st.number_input(f"{afc_choice} Score", 0, 100, 0)
        score_nfc = st.number_input(f"{nfc_choice} Score", 0, 100, 0)
        time_left = st.slider("Minutes Remaining", 1, 60, 60)
    
    with col_c:
        inj_map = {"None": 0.0, "Role": 0.03, "Starter": 0.07, "Star": 0.15, "Elite": 0.30}
        afc_inj = st.select_slider(f"{afc_choice} Injuries", options=["None", "Role", "Starter", "Star", "Elite"])
        nfc_inj = st.select_slider(f"{nfc_choice} Injuries", options=["None", "Role", "Starter", "Star", "Elite"])
        weather_map = {"Clear": 1.0, "Wind/Rain": 0.88, "Snow": 0.82}
        weather = st.selectbox("Weather", ["Clear", "Wind/Rain", "Snow"])
        sim_count = st.select_slider("Iterations", options=[1000, 10000, 25000], value=10000)

# --- SIMULATION ---
afc_p = afc_teams[afc_teams['team_name'] == afc_choice]['power_rating'].values[0]
nfc_p = nfc_teams[nfc_teams['team_name'] == nfc_choice]['power_rating'].values[0]

def run_sim():
    base_ppm = 0.42 * weather_map[weather]
    # Scored points is influenced by your power vs their power
    afc_rate = base_ppm * (afc_p / nfc_p) * strat_map[afc_strat] * (1 - inj_map[afc_inj])
    nfc_rate = base_ppm * (nfc_p / afc_p) * strat_map[nfc_strat] * (1 - inj_map[nfc_inj])
    
    afc_scores = score_afc + np.random.poisson(afc_rate * time_left, sim_count)
    nfc_scores = score_nfc + np.random.poisson(nfc_rate * time_left, sim_count)
    return afc_scores, nfc_scores

if st.button("🚀 Execute Simulation"):
    afc_res, nfc_res = run_sim()
    afc_win_p = (afc_res > nfc_res).mean() * 100
    nfc_win_p = (nfc_res > afc_res).mean() * 100

    st.markdown("### 🏟️ Final Projection")
    res_c1, res_vs, res_c2 = st.columns([2, 1, 2])
    
    with res_c1:
        st.markdown(f"<div style='text-align: right;'><img src='{nfc_logo}' width='100'><h2>{nfc_choice}</h2><h3>Win Prob: {nfc_win_p:.1f}%</h3><p>Avg Score: {nfc_res.mean():.1f}</p></div>", unsafe_allow_html=True)
    with res_vs:
        st.markdown("<h1 style='text-align: center; margin-top: 40px;'>VS</h1>", unsafe_allow_html=True)
    with res_c2:
        st.markdown(f"<div style='text-align: left;'><img src='{afc_logo}' width='100'><h2>{afc_choice}</h2><h3>Win Prob: {afc_win_p:.1f}%</h3><p>Avg Score: {afc_res.mean():.1f}</p></div>", unsafe_allow_html=True)

    # Chart
    spreads = afc_res - nfc_res
    chart_data = pd.DataFrame({'Spread': spreads})
    hist = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Spread:Q', bin=alt.Bin(maxbins=40), title='Point Spread (Negative = Seahawks Favor)'),
        y='count()',
        color=alt.condition(alt.datum.Spread > 0, alt.value('#003366'), alt.value('#C60C30'))
    ).properties(height=350)
    st.altair_chart(hist, use_container_width=True)
