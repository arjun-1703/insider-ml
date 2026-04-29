# 📈 Insider Trading Signal Detector

Detects potential insider trading in NSE-listed stocks by estimating what a stock's
**fair price should be** using PE, PEG, EPS, fundamentals + market data, comparing it
to the **actual price**, and scoring each day with a **0–100% insider trading probability**.

---

## 🎯 How It Works

```
STEP 1 — TRAIN
10 large-cap NSE companies → Features (PE, PEG, EPS, returns, beta…) → Fair-Value Model

STEP 2 — ANALYZE any company
Type ticker in web app → Download live data → Predict fair price
→ Compare actual vs predicted → Insider Score (0–100%) → Charts + CSV
```

The score is built from 4 components:
| Component | Weight | What it measures |
|---|---|---|
| Price inflation magnitude | 40 pts | How many σ above fair value is the price |
| Volume spike | 30 pts | How unusual is the trading volume |
| Inflation direction | 20 pts | Is price pumped UP (classic insider pattern) |
| Suddenness | 10 pts | Was it a sudden spike or gradual drift |

---

## 📁 Project Structure

```
insider-ml/
├── app.py                   ← Streamlit web interface (run this)
├── main.py                  ← Command line pipeline (train here)
├── training_tickers.txt     ← Companies used to train the model
├── requirements.txt
├── src/
│   ├── data_loader.py       ← Downloads live data from yfinance
│   ├── features.py          ← Builds all features (PE, PEG, beta etc.)
│   ├── model.py             ← Trains and loads the fair-value model
│   ├── detector.py          ← Inflation + insider score logic
│   └── plotter.py           ← Analysis charts
└── outputs/
    ├── models/              ← Trained model saved here
    ├── plots/               ← Charts saved here (command line mode)
    └── *.csv                ← Per-ticker signal data
```

---

## 🚀 Setup (Conda)

```bash
# Create and activate environment
conda create -n insider-ml python=3.11
conda activate insider-ml

# Navigate to project
cd path\to\insider-ml

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Step 1 — Train the model (do this ONCE)
```bash
python main.py --train
```
- Downloads data for 10 large-cap NSE stocks
- Trains a Gradient Boosting fair-value model
- Saves model to `outputs/models/fair_value_model.pkl`
- Takes approximately 3–5 minutes

### Step 2 — Launch the web app
```bash
streamlit run app.py
```
- Opens automatically at `http://localhost:8501`
- Type any NSE ticker (e.g. `ADANIENT.NS`, `NUCLEUS.NS`)
- Set date range and click Analyze
- Results appear instantly in the browser

### Step 3 — (Optional) Command line analysis
```bash
python main.py --analyze NUCLEUS.NS
python main.py --analyze ADANIENT.NS --start 2021-01-01 --end 2024-12-31
```

---

## 🌐 Web App Features

| Feature | Description |
|---|---|
| Ticker input | Type any NSE ticker with .NS suffix |
| Date range | Customize the analysis period |
| Score metrics | Max score, peak date, inflation %, avg score |
| Verdict banner | Colour-coded summary (LOW / MEDIUM / HIGH / CRITICAL) |
| Alert breakdown | Count of days at each alert level |
| Fundamentals panel | PE, PEG, EPS, earnings growth |
| 4 chart tabs | Actual vs Fair Value, Inflation %, Insider Score, Volume |
| Suspicious days table | Top 20 most anomalous days colour-coded by alert level |
| CSV download | Export full signal data with one click |

---

## 📊 Features Used by the Model

| Category | Features |
|---|---|
| **Price** | returns, log returns, MA(5/10/20/50), volatility(10/20), price/MA20 ratio |
| **Volume** | rolling avg volume (10), volume z-score |
| **Fundamental** | PE ratio, forward PE, PEG ratio, EPS, forward EPS, earnings growth, revenue growth, price-to-book, ROE, debt-to-equity |
| **Risk** | index return (NIFTY/SENSEX), rolling 30-day beta |
| **Derived** | volatility compression ratio, momentum (5/10 day) |

---

## 🤖 ML Model

**Gradient Boosting Regressor** (scikit-learn)
- 400 estimators, learning rate 0.05, max depth 4
- Trained on combined data from 10 NSE companies
- Learns what "normal" price behaviour looks like
- Deviations from its predictions = potential anomalies

---

## 🚨 Score Interpretation

| Score | Alert Level | Meaning |
|---|---|---|
| 0–25% | LOW | Normal trading, no concern |
| 25–50% | MEDIUM | Some unusual patterns |
| 50–75% | HIGH | Significant price-volume anomalies |
| 75–100% | CRITICAL | Strong inflation + volume spike — investigate |

---

## 🔁 Every Time You Return

```bash
conda activate insider-ml
cd path\to\insider-ml
streamlit run app.py
```
*(No need to retrain unless you want to add new training tickers)*

---

## ➕ Adding New Training Tickers

Open `training_tickers.txt` and add any NSE ticker on a new line:
```
BAJFINANCE.NS
TATAMOTORS.NS
ADANIENT.NS
```
Then retrain:
```bash
python main.py --train
```

---

## ⚠️ Disclaimer
This tool flags *potential abnormal activity* only.
It does **not** make legal claims of insider trading.
All outputs are for research and educational purposes only.
