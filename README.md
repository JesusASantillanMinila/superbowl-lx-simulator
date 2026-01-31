# Super Bowl LX Monte Carlo Simulator

NFL simulation engine utilizing Poisson distribution modeling to forecast game outcomes. 

### 🛠 Technical Architecture
* **Data Pipeline:** Integrated via `nflreadpy` to ingest seasonal performance metrics.
* **Performance Metric:** Calculates a custom `momentum` coefficient derived from **Offensive EPA** (Passing + Rushing) and **Defensive EPA** (weighted at $1.3\times$ to prioritize defensive suppression).
* **Simulation Engine:** * Generates $N$ iterations (up to 50,000) using `numpy.random.poisson`.
    * **Rate Calculation:** $\text{PPM}_{\text{adj}} = (\text{Base} \times \text{Momentum} \times \text{Strategy}) \times (1 - \text{Opponent\_Momentum} \times \text{Def\_Weight})$.
    * Adjusts for **environmental scalars** (weather) and **personnel availability** (injury severity scalars).
* **Visualization:** Interactive point-spread frequency distribution rendered via **Altair**.

### Dependencies
`streamlit`, `numpy`, `pandas`, `nflreadpy`, `altair`
