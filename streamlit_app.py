import streamlit as st
import numpy as np
import pandas as pd
import nflreadpy as nfl 
import altair as alt

# --- CONFIG & DATA LOAD ---
st.set_page_config(page_title="Super Bowl LX Simulator", layout="wide")

@st.cache_data
def load_nfl_data():
    # 1. Load Team Info (Logos/Names)
    teams = nfl.load_teams().to_pandas()
    
    # 2. Load 2025 Schedule/Results to calculate true Power Ratings
    # This contains home_team, away_team, home_score, away_score
    sched = nfl.load_schedules([2025]).to_pandas()
    sched = sched[sched['game_type'] == 'REG'] # Stick to regular season for base stats
    
    # Calculate Points Scored and Points Allowed for every team
    home_stats = sched.groupby('home_team').agg({'home_score': 'sum', 'away_score': 'sum', 'home_rest': 'count'}).rename(columns={'home_score': 'pts_scored', 'away_score': 'pts_allowed', 'home_rest': 'games'})
    away_stats = sched.groupby('away_team').agg({'away_score': 'sum', 'home_score': 'sum', 'away_rest': 'count'}).rename(columns={'away_score': 'pts_scored', 'home_score': 'pts_allowed', 'away_rest': 'games'})
    
    # Combine Home and Away stats
    full_stats = home_stats.add(away_stats, fill_value=0)
    full_stats['avg_diff'] = (full_stats['pts_scored'] - full_stats['pts_allowed']) / full_stats['games']
    
    # Merge with team metadata
    df = pd.merge(teams[['team_abbr', 'team_conf', 'team_name', 'team_logo_wikipedia']], 
                  full_stats.reset_index(), left_on='team_abbr', right_on='index')
    
    # Power Rating based on Point Differential (Normalized 0.8 to 1.2)
    # This captures the "Seahawks Defense" effect because their pts_allowed is low
    diff_min, diff_max = df['avg_diff'].min(), df['avg_diff'].max()
    df['power_rating'] = ((df['avg_diff'] - diff_min) / (diff_max - diff_min)) * 0.4 + 0.8
    
    return df

data = load_nfl_data()

st.title(" 🏈 Super Bowl LX: Final Forecast")

# --- UI SETTINGS ---
with st.expander("🛠️ Simulation Settings", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    
    afc_teams = data[data['team_conf'] == 'AFC'].sort_values('team_name')
    nfc_teams = data[data['team_conf'] == 'NFC'].sort_values('team_name')
    
    with col_a:
        afc_choice = st.selectbox("AFC Champion", afc_teams['team_name'].tolist(), index=afc_teams['team_name'].tolist().index('New England Patriots') if 'New England Patriots' in afc_teams['team_name'].tolist() else 0)
        nfc_choice = st.selectbox("NFC Champion", nfc_teams['team_name'].tolist(), index=nfc_teams['team_name'].tolist().index('Seattle Seahawks') if 'Seattle Seahawks' in nfc_teams['team_name'].tolist() else 0)
        
        afc_logo = afc_teams[afc_teams['team_name'] == afc_choice]['team_logo_wikipedia'].values[0]
        nfc_logo = nfc_teams[nfc_teams['team_name'] == nfc_choice]['team_logo_wikipedia'].values[0]

    with col_b:
        st.write("**Game State**")
        time_left = st.slider("Minutes Remaining", 1, 60, 60)
        sim_count = st.select_slider("Iterations", options=[1000, 10000, 25000], value=10000)
    
    with col_c:
        weather_map = {"Dome/Clear": 1.0, "Rain/Wind": 0.85, "Snow": 0.75}
        weather = st.selectbox("Weather Conditions", list(weather_map.keys()))
        boost = st.slider("Upsert Momentum Boost (%)", -20, 20, 0) / 100

# --- SIM ENGINE ---
afc_p = afc_teams[afc_teams['team_name'] == afc_choice]['power_rating'].values[0]
nfc_p = nfc_teams[nfc_teams['team_name'] == nfc_choice]['power_rating'].values[0]

def run_sim():
    # Base scoring rate (NFL average is ~0.4 pts per minute)
    base_ppm = 0.40 * weather_map[weather]
    
    # Calculate relative scoring rates
    # If nfc_p (Seahawks) is higher than afc_p (Patriots), afc_rate drops
    afc_rate = base_ppm * (afc_p / nfc_p) * (1 + boost)
    nfc_rate = base_ppm * (nfc_p / afc_p)
    
    afc_scores = np.random.poisson(afc_rate * time_left, sim_count)
    nfc_scores = np.random.poisson(nfc_rate * time_left, sim_count)
    return afc_scores, nfc_scores

if st.button("🚀 Run Monte Carlo Simulation", use_container_width=True):
    afc_res, nfc_res = run_sim()
    afc_win_p = (afc_res > nfc_res).mean() * 100
    nfc_win_p = (nfc_res > afc_res).mean() * 100

    # Display Results
    st.markdown("---")
    res_c1, res_vs, res_c2 = st.columns([2, 1, 2])
    
    with res_c1:
        st.markdown(f"<div style='text-align: center;'><img src='{nfc_logo}' width='120'><h2>{nfc_choice}</h2><h1 style='color: #C60C30;'>{nfc_win_p:.1f}%</h1><p>Avg Points: {nfc_res.mean():.1f}</p></div>", unsafe_allow_html=True)
    with res_vs:
        st.markdown("<h1 style='text-align: center; margin-top: 80px;'>VS</h1>", unsafe_allow_html=True)
    with res_c2:
        st.markdown(f"<div style='text-align: center;'><img src='{afc_logo}' width='120'><h2>{afc_choice}</h2><h1 style='color: #003366;'>{afc_win_p:.1f}%</h1><p>Avg Points: {afc_res.mean():.1f}</p></div>", unsafe_allow_html=True)

    # Spread Chart
    spreads = afc_res - nfc_res
    chart_df = pd.DataFrame({'Spread': spreads})
    hist = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X('Spread:Q', bin=alt.Bin(maxbins=30), title='Point Spread (Negative = Seahawks Leader)'),
        y=alt.Y('count()', title='Frequency'),
        color=alt.condition(alt.datum.Spread > 0, alt.value('#003366'), alt.value('#C60C30'))
    ).properties(height=300)
    st.altair_chart(hist, use_container_width=True)
