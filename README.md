# Workforce Demand Forecasting

**SARIMA + Prophet + Weighted Ensemble | Rolling Validation | 5.1% MAPE**

A production-oriented workforce headcount forecasting system built to support hiring demand projections and recruiter capacity planning. HR and TA leadership need reliable 1-2 quarter projections to allocate recruiter bandwidth and budget proactively — this project delivers that with a validated, interpretable ensemble model.

---

## Results

| Model | MAPE | MAE | Notes |
|---|---|---|---|
| SARIMA(1,1,1)(1,1,1,12) | ~6.8% | — | Strong on autocorrelation |
| Prophet | ~6.2% | — | Strong on changepoints |
| **Ensemble (weighted)** | **~5.1%** | **best** | Optimal blending via scipy |

Ensemble outperforms both individual models. Optimal SARIMA weight found via `scipy.optimize.minimize_scalar` on held-out validation MAPE.

---

## Structure

```
workforce-demand-forecasting/
├── notebooks/
│   └── workforce_forecasting.ipynb   # Full analysis notebook
├── src/
│   ├── data_generator.py             # Synthetic workforce data with realistic patterns
│   ├── sarima_model.py               # SARIMA fitting + rolling walk-forward evaluation
│   ├── prophet_model.py              # Prophet fitting + cross-validation
│   └── ensemble.py                   # Weighted blending + model comparison
├── outputs/                          # Generated charts and forecast CSV
├── requirements.txt
└── README.md
```

---

## Methodology

### Data
Synthetic monthly headcount across 5 departments (Engineering, Sales, Operations, Analytics, Finance) from Jan 2019 to Jun 2025. Features:
- Exponential growth trend (dept-specific rates)
- Annual and quarterly seasonality
- COVID shock in 2020 Q2-Q3
- Realistic Gaussian noise

### SARIMA
- Differencing order `d=1` confirmed via ADF test
- Seasonal period `m=12` for monthly data
- Rolling walk-forward validation: re-fit on expanding window, forecast 1 step ahead

### Prophet
- Yearly + quarterly seasonality components
- `changepoint_prior_scale=0.05` to capture structural breaks without overfitting
- `interval_width=0.90` for planning-grade confidence intervals

### Ensemble
- Weighted average: `ensemble = w * SARIMA + (1-w) * Prophet`
- Weight `w` optimized on validation MAPE using bounded scalar minimization
- Typically converges to ~55-65% SARIMA weight

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook notebooks/workforce_forecasting.ipynb
```

Or run the data generator standalone:

```bash
cd src
python data_generator.py
```

---

## Business Context

This forecasting model was designed to answer: *"How many engineers/analysts will we need to hire next quarter?"*

Outputs feed into:
- **Recruiter capacity planning** — how many open reqs per recruiter
- **Budget forecasting** — cost per hire projections
- **Diversity pipeline planning** — early identification of imbalance risk

The ensemble's 5.1% MAPE on a 12-month test window translates to an average forecast error of roughly ±50 headcount on a 1,000-person organization — a planning-grade result.

---

*Part of [Sunidhi Sharma's People Analytics portfolio](https://sunidhi-analytics.github.io/Portfolio-Sunidhi)*
