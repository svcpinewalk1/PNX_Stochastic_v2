import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import beta
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests
import io
from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid
from matplotlib.ticker import FixedLocator
import plotly.express as px
import plotly.graph_objects as go

#cd Python Files && cd SM Model && cd Live
#streamlit run Model_10_25_Cloud.py

# Must be first Streamlit call
st.set_page_config(
    page_title="Pernix Stochastic Model",
    layout="wide"  # expands app to full width
)

st.title("Pernix Stochastic Model")

# ----------------------
# Data Preparation Class
# ----------------------
class Data:
    def __init__(self, transition_table, portfolio_data, exposure_data):
        self.transition_table = transition_table
        self.portfolio_data = portfolio_data
        self.exposure_data = exposure_data

# ----------------------
# Simulation Class
# ----------------------
class Simulation:
    def __init__(self, data):
        self.data = data

    def apply_stress(self, high_stress_factor: float):
        """Scale downgrade probabilities by a stress factor and renormalize rows.
        Upgrades are zeroed; stay prob is not directly divided (prevents extreme inflation).
        Returns (stressed_matrix, standard_matrix).
        """
        standard_matrix = self.data.transition_table.copy()
        stressed_matrix = standard_matrix.copy()

        # Derive medium/low for information & potential future use
        medium_stress_factor = 1 + (high_stress_factor - 1) / 2
        low_stress_factor = 1 + (medium_stress_factor - 1) / 2

        rating_stress_factors = {
            "AAA": low_stress_factor, "AA+": low_stress_factor, "AA": low_stress_factor, "AA-": low_stress_factor,
            "A+": medium_stress_factor, "A": medium_stress_factor, "A-": medium_stress_factor,
            "BBB+": medium_stress_factor, "BBB": medium_stress_factor, "BBB-": medium_stress_factor,
            "BB+": high_stress_factor, "BB": high_stress_factor, "BB-": high_stress_factor,
            "B+": high_stress_factor, "B": high_stress_factor, "B-": high_stress_factor,
            "CCC+": high_stress_factor, "CCC": high_stress_factor, "CCC-": high_stress_factor,
            "CC": high_stress_factor, "C": high_stress_factor, "D": 1.0
        }

        cols = standard_matrix.columns
        for rating in standard_matrix.index:
            base_row = standard_matrix.loc[rating].copy()

            # Keep D absorbing
            if rating == 'D':
                stressed_matrix.loc[rating] = base_row
                continue

            # Zero upgrades
            upgrade_cols = cols[:cols.get_loc(rating)]
            base_row[upgrade_cols] = 0.0

            # Scale downgrades by factor for this rating
            downgrade_cols = cols[cols.get_loc(rating) + 1:]
            factor = float(rating_stress_factors.get(rating, 1.0))
            if len(downgrade_cols) > 0:
                base_row[downgrade_cols] = base_row[downgrade_cols] * factor

            # Renormalize the entire row (stay + scaled downgrades)
            s = base_row.sum()
            if s > 0:
                base_row = base_row / s

            stressed_matrix.loc[rating] = base_row

        return stressed_matrix, standard_matrix

    def calculate_yearlypayout(self, new_ratings_df: pd.DataFrame, sim_id=None) -> pd.DataFrame:
        """Compute payouts for first default per riskref; attach sim_id when provided."""
        out_cols = [
            'riskref','Year','Mean_Yearly_Payout','Max_yearly_payouts','Min_yearly_payouts',
            'Initial_Rating','LGD','Binder','Binder_Year','Coverage','Industry','Risk_Code','sim_id'
        ]
        if new_ratings_df.empty:
            return pd.DataFrame(columns=out_cols)

        defaults = new_ratings_df[new_ratings_df['New Rating'] == 'D']
        if defaults.empty:
            return pd.DataFrame(columns=out_cols)

        min_defaults = defaults.groupby('Risk Reference')['Year'].min()
        exposure_dict = self.data.exposure_data.set_index(['RISKREF', 'tenor'])['Exposure_at_date'].to_dict()
        portfolio_dict = self.data.portfolio_data.set_index('riskref').to_dict('index')

        rows = []
        for riskref, year in min_defaults.items():
            exposure = float(exposure_dict.get((riskref, int(year)), 0.0))
            pinfo = portfolio_dict.get(riskref)
            if pinfo is None:
                continue
            mean_lgd = float(pinfo['LGD'])
            max_lgd = float(pinfo['Max LGD'])
            min_lgd = float(pinfo['Min LGD'])
            rows.append({
                'riskref': riskref,
                'Year': int(year),
                'Mean_Yearly_Payout': mean_lgd * exposure,
                'Max_yearly_payouts': max_lgd * exposure,
                'Min_yearly_payouts': min_lgd * exposure,
                'Initial_Rating': pinfo['Initial Rating'],
                'LGD': mean_lgd,
                'Binder': pinfo['Binder'],
                'Binder_Year': pinfo['Binder Year'],
                'Coverage': pinfo['Coverage'],
                'Industry': pinfo['Industry'],
                'Risk_Code': pinfo['Risk Code'],
                'sim_id': sim_id
            })
        return pd.DataFrame(rows, columns=out_cols)

# ----------------------
# Load Data (Streamlit Cache)
# ----------------------

# Cloud Data

#Username of your GitHub account

username = 'svcpinewalk1'

# Personal Access Token (PAO) from your GitHub account

token = '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'

# Creates a re-usable session object with your creds in-built

github_session = requests.Session()

github_session.auth = (username, token)

# Downloading the xlsx file from your GitHub

exposure_url = "https://raw.githubusercontent.com/svcpinewalk1/PNX_Stochastic_v2/main/Exposure%20Table.xlsx"
#  "https://raw.githubusercontent.com/o7shirePW/PNX_Stochastic/main/Exposure%20Table.xlsx"

portfolio_url = "https://raw.githubusercontent.com/svcpinewalk1/PNX_Stochastic_v2/main/Pernix%20Data.xlsx"
# "https://raw.githubusercontent.com/o7shirePW/PNX_Stochastic/main/Pernix%20Data.xlsx"

Transition_url = "https://raw.githubusercontent.com/svcpinewalk1/PNX_Stochastic_v2/main/Transition_Table.xlsx"
#"https://raw.githubusercontent.com/o7shirePW/PNX_Stochastic/main/Transition_Table.xlsx"


Transition_download = github_session.get(Transition_url).content
portfolio_download = github_session.get(portfolio_url).content
exposure_download = github_session.get(exposure_url).content

@st.cache_data
def load_data():
    transition_table = pd.read_excel(io.BytesIO(Transition_download), engine='openpyxl', index_col=0)
    portfolio_data = pd.read_excel(io.BytesIO(portfolio_download), engine='openpyxl')
    exposure_data = pd.read_excel(io.BytesIO(exposure_download), engine='openpyxl')
    return Data(transition_table, portfolio_data, exposure_data)

# ----------------------
# Helper functions for worker
# ----------------------

def _payouts_from_new_ratings(new_ratings_df, exposure_dict, portfolio_dict, sim_id):
    out_cols = [
        'riskref','Year','Mean_Yearly_Payout','Max_yearly_payouts','Min_yearly_payouts',
        'Initial_Rating','LGD','Binder','Binder_Year','Coverage','Industry','Risk_Code','sim_id'
    ]
    if new_ratings_df.empty:
        return pd.DataFrame(columns=out_cols)
    defaults = new_ratings_df[new_ratings_df['New Rating'] == 'D']
    if defaults.empty:
        return pd.DataFrame(columns=out_cols)
    min_defaults = defaults.groupby('Risk Reference')['Year'].min()
    rows = []
    for riskref, year in min_defaults.items():
        exposure = float(exposure_dict.get((riskref, int(year)), 0.0))
        pinfo = portfolio_dict.get(riskref)
        if pinfo is None:
            continue
        mean_lgd = float(pinfo['LGD'])
        max_lgd = float(pinfo['Max LGD'])
        min_lgd = float(pinfo['Min LGD'])
        rows.append({
            'riskref': riskref,
            'Year': int(year),
            'Mean_Yearly_Payout': mean_lgd * exposure,
            'Max_yearly_payouts': max_lgd * exposure,
            'Min_yearly_payouts': min_lgd * exposure,
            'Initial_Rating': pinfo['Initial Rating'],
            'LGD': mean_lgd,
            'Binder': pinfo['Binder'],
            'Binder_Year': pinfo['Binder Year'],
            'Coverage': pinfo['Coverage'],
            'Industry': pinfo['Industry'],
            'Risk_Code': pinfo['Risk Code'],
            'sim_id': sim_id
        })
    return pd.DataFrame(rows, columns=out_cols)


def _single_run_worker(matrix_df: pd.DataFrame, portfolio_df: pd.DataFrame,
                       exposure_dict: dict, portfolio_dict: dict, sim_id: int) -> pd.DataFrame:
    """One simulation: advance ratings for each policy across its tenor, then compute payouts."""
    results = []
    ratings_cols = matrix_df.columns
    for _, row in portfolio_df.iterrows():
        rating = row['Initial Rating']
        riskref = row['riskref']
        tenor = int(row['Tenor'])
        for yr in range(tenor):
            probs = matrix_df.loc[rating].fillna(0.0).values
            new_rating = np.random.choice(ratings_cols, p=probs)
            results.append((yr + 1, new_rating, riskref))
            rating = new_rating
    nr = pd.DataFrame(results, columns=['Year', 'New Rating', 'Risk Reference']) if results else pd.DataFrame(columns=['Year', 'New Rating', 'Risk Reference'])
    return _payouts_from_new_ratings(nr, exposure_dict, portfolio_dict, sim_id)

# ----------------------
# Main App Execution
# ----------------------

data = load_data()
simulation = Simulation(data)

# Scenario selection
Transition_Stress = st.selectbox("Pick a Stress Scenario", ['Standard', 'Global Financial Crisis', 'Covid Pandemic', 'Early 90s Recession'])
stress_factors = {
    'Standard': 1.0,
    'Covid Pandemic': 4.30,
    'Global Financial Crisis': 3.79,
    'Early 90s Recession': 5.78
}
stressed_matrix, standard_matrix = simulation.apply_stress(stress_factors[Transition_Stress])

# Number of simulations
num_simulations = st.selectbox("Number of Simulations", [250, 500, 1000, 5000, 10000], index=0)

# Filters
binder_year = st.multiselect("Binder Year", ['ALL', 2022, 2023, 2024, 2025], default='ALL')
if 'ALL' in binder_year: binder_year = [2022, 2023, 2024, 2025]

industry = st.multiselect("Industry", ['ALL'] + list(data.portfolio_data['Industry'].unique()), default='ALL')
if 'ALL' in industry: industry = data.portfolio_data['Industry'].unique()

risk_code = st.multiselect("Risk Code", ['ALL', 'CF', 'CR'], default='ALL')
if 'ALL' in risk_code: risk_code = ['CF', 'CR']

coverage = st.multiselect("Coverage", ['ALL'] + list(data.portfolio_data['Coverage'].unique()), default='ALL')
if 'ALL' in coverage: coverage = data.portfolio_data['Coverage'].unique()

binder = st.multiselect("Binder", ['ALL'] + list(data.portfolio_data['Binder'].unique()), default='ALL')
if 'ALL' in binder: binder = data.portfolio_data['Binder'].unique()

filtered_portfolio = data.portfolio_data[(data.portfolio_data['Binder Year'].isin(binder_year)) &
                                         (data.portfolio_data['Industry'].isin(industry)) &
                                         (data.portfolio_data['Risk Code'].isin(risk_code)) &
                                         (data.portfolio_data['Coverage'].isin(coverage)) &
                                         (data.portfolio_data['Binder'].isin(binder))]

# Choose matrix according to scenario (use stressed when not Standard)
matrix_to_use = standard_matrix if Transition_Stress == 'Standard' else stressed_matrix

with st.expander("Show Transition Matrix and Stress Factors", expanded=False):
    st.write("### Transition Matrix in Use")
    st.dataframe(matrix_to_use)

    rating_stress_factors = {
        "AAA": "Low",  "AA+": "Low",  "AA": "Low",  "AA-": "Low",
        "A+": "Medium", "A": "Medium", "A-": "Medium",
        "BBB+": "Medium", "BBB": "Medium", "BBB-": "Medium",
        "BB+": "High", "BB": "High", "BB-": "High",
        "B+": "High", "B": "High", "B-": "High",
        "CCC+": "High", "CCC": "High", "CCC-": "High",
        "CC": "High", "C": "High", "D": "1.0"
    }

    st.write("### Rating Stress Factors")
    st.dataframe(pd.DataFrame.from_dict(rating_stress_factors, orient="index", columns=["Stress Level"]))

    # Calculate medium and low stress factors
    medium_stress_factor = 1 + (stress_factors[Transition_Stress] - 1) / 2
    low_stress_factor = 1 + (medium_stress_factor - 1) / 2

    factor_table = pd.DataFrame([(stress_factors[Transition_Stress], medium_stress_factor, low_stress_factor)],
    columns=['High Stress Factor', 'Medium Stress Factor', 'Low Stress Factor'])

    st.dataframe(factor_table)

if st.button("Run Simulation"):
    start = time.time()

    # Precompute dicts for fast access in workers
    exposure_dict = data.exposure_data.set_index(['RISKREF','tenor'])['Exposure_at_date'].to_dict()
    portfolio_dict = data.portfolio_data.set_index('riskref').to_dict('index')

    # Progress UI
    progress_bar = st.progress(0)
    status = st.empty()
    status.text("Starting Simulations....")
    runner = st.empty()

    total = int(num_simulations)
    completed = 0
    bar_width = 120

    def draw_bowling_strike(progress):
        bowler = "🏌️‍♂️"
        ball_frames = ["◐", "◓", "◑", "◒"]
        frame = ball_frames[completed % len(ball_frames)]
        pins = "<span style='display:inline-block; transform:scaleX(-1)'>🎳</span>"
        strike = "💥"
        pos = min(bar_width, int(progress * bar_width))
        lane = ["·"] * bar_width
        if progress < 1.0:
            lane[pos] = frame
            end = pins
        else:
            lane[-1] = " "
            end = strike
        lane_str = "|" + "".join(lane) + "|"
        runner.markdown(
            f"<div style='font-size:20px; font-family:monospace'>{bowler} {lane_str}{end}</div>",
            unsafe_allow_html=True
        )

    chunks = []
    max_workers = min(32, cpu_count() or 1, total)
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [
            exe.submit(_single_run_worker, matrix_to_use, filtered_portfolio, exposure_dict, portfolio_dict, i)
            for i in range(total)
        ]
        for f in as_completed(futures):
            df = f.result()
            chunks.append(df)
            completed += 1
            frac = completed/total
            progress_bar.progress(frac)
            status.text(f"Running simulation {completed} of {total}...")
            draw_bowling_strike(frac)

    sim_data = (pd.concat([c for c in chunks if c is not None and not c.empty], ignore_index=True)
                if chunks else pd.DataFrame(columns=['riskref','Year','Mean_Yearly_Payout','Max_yearly_payouts','Min_yearly_payouts',
                                                     'Initial_Rating','LGD','Binder','Binder_Year','Coverage','Industry','Risk_Code','sim_id']))

    # Finalize progress
    progress_bar.progress(1.0)
    status.text("Done!")
    draw_bowling_strike(1.0)

    # Runtime formatting
    run_time = time.time() - start
    if run_time < 60:
        st.success(f"Simulation completed in {round(run_time, 2)} seconds.")
    else:
        minutes, seconds = divmod(run_time, 60)
        st.success(f"Simulation completed in {int(minutes)} min {round(seconds, 1)} sec.")

    # Results
    st.write("## Defaulted Risks")
    if sim_data.empty:
        st.info("No defaults observed across the simulations with current filters.")
    else:
        # =============================
        # Defaulted Risks Display
        # =============================
        # Drop duplicates
        unique_risks = sim_data.drop_duplicates(subset='riskref')

        # Rename columns for better readability
        unique_risks = unique_risks.rename(columns={
            "riskref": "Risk Ref",
            "Year": "Year of Default",
            "Mean_Yearly_Payout": "Mean Payout",
            "Max_yearly_payouts": "Max Payout",
            "Min_yearly_payouts": "Min Payout",
            "Initial_Rating": "Initial Rating",
            "LGD": "LGD",
            "Binder_Year": "Binder Year",
            "Coverage": "Coverage",
            "Industry": "Industry",
            "Risk_Code": "Risk Code",
            "sim_id": "Sim Number"
        })

        # Display nicely formatted dataframe
        styled_df = unique_risks.style.format({
            "Mean Payout": "{:,.0f}",
            "Max Payout": "{:,.0f}",
            "Min Payout": "{:,.0f}"
        })

        # Display styled DataFrame
        st.dataframe(styled_df, width="stretch")

        # =============================
        # Risk Level Distribution
        # =============================
        risk_vals_mean = sim_data['Mean_Yearly_Payout'].astype(float) / 1e6
        risk_vals_max = sim_data['Max_yearly_payouts'].astype(float) / 1e6
        risk_vals_min = sim_data['Min_yearly_payouts'].astype(float) / 1e6

        # KDEs
        risk_vals_mean = np.asarray(risk_vals_mean)
        risk_vals_mean = risk_vals_mean[np.isfinite(risk_vals_mean)]

        risk_vals_max = np.asarray(risk_vals_max)
        risk_vals_max = risk_vals_max[np.isfinite(risk_vals_max)]

        risk_vals_min = np.asarray(risk_vals_min)
        risk_vals_min = risk_vals_min[np.isfinite(risk_vals_min)]


        x_grid = np.linspace(0, max(risk_vals_max.max(), risk_vals_mean.max()), 500)
        if len(risk_vals_mean) > 1:
            pdf_mean = gaussian_kde(risk_vals_mean)(x_grid)
        pdf_max = gaussian_kde(risk_vals_max)(x_grid)
        pdf_min = gaussian_kde(risk_vals_min)(x_grid)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x_grid, y=pdf_mean, name="Mean Payout",
                                line=dict(color="#323755", width=2), fill="tozeroy"))
        fig.add_trace(go.Scatter(x=x_grid, y=pdf_min, name="Min Payout",
                                line=dict(color="#8A92C4", width=2), fill="tozeroy"))
        fig.add_trace(go.Scatter(x=x_grid, y=pdf_max, name="Max Payout",
                                line=dict(color="#DEE2FF", width=2), fill="tozeroy"))

        # Percentile markers on mean payouts
        for percentile in [90, 95, 99]:
            p_val = np.percentile(risk_vals_mean, percentile)
            fig.add_vline(x=p_val, line=dict(color="red", dash="dot"))
            fig.add_annotation(x=p_val, y=max(pdf_mean)*0.9,
                            text=f"{percentile}%", showarrow=False,
                            xanchor="right", font=dict(color="red", size=10))

        fig.update_layout(
            title="**Risk Level** Modelled Loss Distribution (Millions)",
            xaxis_title="Payout (Millions)",
            yaxis_title="Density",
            template="plotly_white",
            legend=dict(title="Toggle Curves", orientation="h", y=-0.2),
            xaxis=dict(showgrid=True, gridcolor="lightgrey"),
            yaxis=dict(showgrid=True, gridcolor="lightgrey")
        )

        st.plotly_chart(fig, width="stretch", height=600)

        # =============================
        # Portfolio Level Distribution
        # =============================

        totals = sim_data.groupby('sim_id')['Mean_Yearly_Payout'].sum()
        totals = totals.reindex(np.arange(total), fill_value=0.0) / 1e6

        totals_min = sim_data.groupby('sim_id')['Min_yearly_payouts'].sum()
        totals_min = totals_min.reindex(np.arange(total), fill_value=0.0) / 1e6

        totals_max = sim_data.groupby('sim_id')['Max_yearly_payouts'].sum()
        totals_max = totals_max.reindex(np.arange(total), fill_value=0.0) / 1e6

        # KDEs
        x_grid = np.linspace(0, max(totals_max.max(), totals.max()), 500)
        pdf_mean = gaussian_kde(totals)(x_grid)
        pdf_max = gaussian_kde(totals_max)(x_grid)
        pdf_min = gaussian_kde(totals_min)(x_grid)

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(x=x_grid, y=pdf_mean, name="Mean Payout",
                                line=dict(color="#323755", width=2), fill="tozeroy"))
        fig2.add_trace(go.Scatter(x=x_grid, y=pdf_min, name="Min Payout",
                                line=dict(color="#8A92C4", width=2), fill="tozeroy"))
        fig2.add_trace(go.Scatter(x=x_grid, y=pdf_max, name="Max Payout",
                                line=dict(color="#DEE2FF", width=2), fill="tozeroy"))

        # Percentile markers on mean distribution
        for percentile in [90, 95, 99]:
            p_val = np.percentile(totals, percentile)
            fig2.add_vline(x=p_val, line=dict(color="red", dash="dot"))
            fig2.add_annotation(x=p_val, y=max(pdf_mean)*0.9,
                                text=f"{percentile}%", showarrow=False,
                                xanchor="right", font=dict(color="red", size=10))

        fig2.update_layout(
            title="**Portfolio Level** Modelled Loss Distribution (Millions)",
            xaxis_title="Total Simulation Payout (Millions)",
            yaxis_title="Density",
            template="plotly_white",
            legend=dict(title="Toggle Curves", orientation="h", y=-0.2),
            xaxis=dict(showgrid=True, gridcolor="lightgrey"),
            yaxis=dict(showgrid=True, gridcolor="lightgrey")
        )

        st.plotly_chart(fig2, width="stretch", height=600)

        # =============================
        # Portfolio-Level Stats (KDE only)
        # =============================
        st.write("## Portfolio-Level Modelled Stats")

        totals = totals * 1e6

        if len(totals) > 1:
            kde = gaussian_kde(totals)
            x_grid = np.linspace(totals.min(), totals.max(), 2000)
            pdf_vals = kde(x_grid)

            dx = x_grid[1] - x_grid[0]
            cdf_vals = np.cumsum(pdf_vals) * dx
            cdf_vals /= cdf_vals[-1]

            def kde_quantile(q):
                return np.interp(q, cdf_vals, x_grid)

            mean_kde = trapezoid(x_grid * pdf_vals, x_grid)
            std_kde = np.sqrt(trapezoid((x_grid - mean_kde) ** 2 * pdf_vals, x_grid))

            kde_stats = {
                'Default Count': len(unique_risks),  # unique risks that defaulted
                'mean (portfolio level)': mean_kde,
                'mean (risk level)': (sim_data['Mean_Yearly_Payout'].mean() if not sim_data.empty else 0.0),
                'std': std_kde,
                'min': totals.min(),
                '25%': kde_quantile(0.25),
                '50%': kde_quantile(0.50),
                '75%': kde_quantile(0.75),
                '1 in 10': kde_quantile(0.90),
                '1 in 20': kde_quantile(0.95),
                '1 in 100': kde_quantile(0.99),
                'max': totals.max(),
            }

            stats_df = pd.DataFrame.from_dict(kde_stats, orient='index', columns=['KDE Stats'])

        else:
            stats_df = pd.DataFrame({
                'KDE Stats': {
                    'Default Count': len(unique_risks),
                    'mean (portfolio level)': float(totals.mean()),
                    'mean (risk level)': float(sim_data['Mean_Yearly_Payout'].mean()) if not sim_data.empty else 0.0,
                    'std': 0.0,
                    'min': float(totals.min()) if len(totals) else 0.0,
                    '50%': float(totals.median()) if len(totals) else 0.0,
                    'max': float(totals.max()) if len(totals) else 0.0,
                }
            })

        # Format nicely
        stats_df_fmt = stats_df.map(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)

        # Add default rate rows
        total_periods = int(filtered_portfolio['Tenor'].sum() * total) if not filtered_portfolio.empty else 0
        num_defaults = len(sim_data)
        default_rate = num_defaults / total_periods if total_periods > 0 else 0.0

        sp_ref_map = {
            'Early 90s Recession': 0.0277,
            'Covid Pandemic': 0.0234,
            'Global Financial Crisis': 0.0156,
            'Standard': 0.0054,
        }
        sp_ref = sp_ref_map.get(Transition_Stress, 0.0)

        default_label = 'Stressed Default Rate' if Transition_Stress != 'Standard' else 'Default Rate'
        stats_df_fmt.loc['S&P Reference Default Rate'] = [f"{sp_ref:.2%}"]
        stats_df_fmt.loc[default_label] = [f"{default_rate:.4%}"]

        # Styling
        def _style_rows(row):
            if row.name == "S&P Reference Default Rate":
                return ['background-color: #C6F6D5; color: #22543D; font-weight: 700']
            if row.name in ("Default Rate", "Stressed Default Rate"):
                return ['background-color: #FFBF00; font-weight: 700']
            return ['']

        styled = stats_df_fmt.style.apply(_style_rows, axis=1)
        st.dataframe(styled)

        



st.sidebar.title("Information Sidebar ℹ️")

st.sidebar.markdown('''
                                    
This tool allows you to explore the distribution of losses interactively. 
It uses a transition table to determine if a risk will default during the policy's life cycle.
If a default occurs, the year of default is recorded, and the exposure at that year is used as the expected loss.  

Exposure at a given year is calculated as 
1/2(Exposure at start of year + Exposure at end of year).

#### Features:

1. **Exposure Model**:
    - This model assumes a normally distributed exposure.
    - It details the most probable exposure during a market crash.
    - It shows the total exposure for rare events (1 in 10 or 1 in 100).
    - **Portfolio Level** Measures and Loss Distribution breaks down the entire Simulation payout which in turn models on the expected loss across your entire Bound Portfolio.
    - **Risk Level** Measures and Loss Distribution reports at a single default payout; risk by risk showing the distribution of individual payouts across your portfolio and number of simulations ran.
2. **LGD Distribution**:
    - This feature uses a beta distribution for the Loss Given Default (**LGD**), grouped by the initial rating of all defaulted entries.
    - It generates random values between [0, 1], centered around the mean LGD value for each rating.
3. **Stress Scenarios**:
    - This feature uses downgrade and upgrade ratios (**Provided by S&P**) to simulate the effects of historical financial crises on the portfolio.
    - For example, selecting the **"Global Financial Crisis"** option enables the model to replicate conditions similar to those of  **2008–2009** effectively simulating the impact as if your insurance portfolio had been active during that period of severe market downturn.
    - By analyzing how the portfolio would have responded to major financial stress events, users can gain insights into potential vulnerabilities and risk exposures under extreme economic conditions.        
    - <p style = 'color: #cc5500;'>The model applies high stress for the first three years, medium stress for the next three years, low stress for the next three years and then standard rates until the end of the policy coverage.</p>
    - <p style = 'color: #cc5500;'>Low, Medium and High stress are multiples of the downgrade and upgrade ratio provided by S&P</p>

#### Customization:
- You can adjust the beta distribution to see how changes affect the model. 
- You can adjust the Portfolio Expected Loss Distribution to filter specific risks by Binder Year, Industry, Risk Code, or Coverage Type. 
- This allows for a more focused simulation of the desired risk distribution.                   


#### Parameters:
- **mean**: Desired mean of the distribution (0 <= mean <= 1).
- **concentration**: Concentration parameter (higher values lead to lower variance).
- **size**: Number of random values to generate (default is the simulation size).

                    
#### Save run as PDF:
    - Click the three-dot icon in the top-right corner.
    - Select Print.
    - For Destination, choose Save as PDF.
    - Click More Settings and adjust the following:
            -    Paper size: A1      
            -    Scale: 100%
            -    Margins: None                                                        

For further assistance, contact: 
**systemsandreports@pinewalkcapital.com**

''',unsafe_allow_html=True)




