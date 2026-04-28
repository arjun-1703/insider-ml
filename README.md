# 📈 Insider Trading Signal Detector

Detects potential insider trading by estimating what a stock's **fair price should be**
(using PE, PEG, EPS, fundamentals + market data), comparing it to the **actual price**,
and scoring each day with a **0–100% insider trading probability**.

---

## 🎯 How It Works

```
Step 1 — TRAIN
Multiple companies → Features (PE, PEG, EPS, returns, beta…) → Fair-Value Model

Step 2 — ANALYZE any company
Download live data → Predict fair price → Compare actual vs predicted
→ Insider Score (0–100%) → Chart + CSV
```

The score is based on:
- How much the actual price exceeds the predicted fair value
- Whether volume spiked at the same time
- Whether the inflation was sudden (not a gradual drift)
- Whether the price was inflated upward (classic pre-announcement pump)

---

## 🚀 Setup (Conda)

```bash
conda create -n insider-ml python=3.11
conda activate insider-ml
cd path\to\insider-ml
pip install -r requirements.txt
```

---

## ▶️ Usage

### Step 1 — Train the model (do this once)
```bash
python main.py --train
```
Trains on 10 large-cap NSE stocks. Takes ~5 minutes (downloads data from yfinance).

### Step 2 — Analyze any company
```bash
python main.py --analyze NUCLEUS.NS
python main.py --analyze ZOMATO.NS
python main.py --analyze ADANIENT.NS
```

You can analyze any NSE stock — it doesn't need to be in the training set.

### Custom date range
```bash
python main.py --analyze NUCLEUS.NS --start 2021-01-01 --end 2024-12-31
```

---

## 📊 Output

### Console report:
```
============================================================
  INSIDER TRADING ANALYSIS: NUCLEUS.NS
============================================================
  VERDICT      : SUSPICIOUS — significant price inflation detected
  MAX SCORE    : 82.4% insider probability
  AVG SCORE    : 31.2% (last 1 year)

  PEAK SUSPICIOUS DATE : 2023-08-14
  Actual Price         : ₹892.50
  Predicted Fair Price : ₹741.20
  Price Inflation      : ₹151.30 (20.4%)
  Volume Z-Score       : 3.8σ
  ...
```

### Chart (outputs/plots/TICKER_analysis.png):
- Panel 1: Actual price vs predicted fair value (red = inflated above fair)
- Panel 2: Inflation % over time
- Panel 3: Insider score 0–100% coloured by risk level
- Panel 4: Volume anomalies (red bars = unusual spikes)

### CSV (outputs/TICKER_signals.csv):
Every trading day with actual price, predicted price, inflation %, insider score, alert level.

---

## 📁 Project Structure

```
insider-ml/
├── main.py                  ← entry point
├── training_tickers.txt     ← companies used to train the model
├── requirements.txt
├── src/
│   ├── data_loader.py       ← yfinance downloader
│   ├── features.py          ← feature engineering (PE, PEG, beta, etc.)
│   ├── model.py             ← trains/loads/predicts fair value
│   ├── detector.py          ← inflation + insider score logic
│   └── plotter.py           ← 4-panel analysis chart
└── outputs/
    ├── models/              ← trained model saved here
    ├── plots/               ← charts saved here
    └── *.csv                ← per-ticker signal data
```

---

## ⚠️ Disclaimer
This tool flags *potential abnormal activity* only.
It does **not** make legal claims of insider trading.
All outputs are for research and educational purposes.
