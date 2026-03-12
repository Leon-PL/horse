"""
Trial Run — Real Data Predictions
==================================
Scrapes today's UK racecards from Sporting Life and predicts each race
using the trained ensemble model.
"""
import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from src.data_scraper import get_scraped_racecards
from src.data_processor import process_data
from src.feature_engineer import engineer_features
from src.model import EnsemblePredictor

# ── 1. Load trained model ────────────────────────────────────────────
print("=" * 70)
print("  🏇  HORSE RACING PREDICTION — TRIAL RUN (Real Data)")
print("=" * 70)

predictor = EnsemblePredictor()
predictor.load()
print("✅ Ensemble model loaded\n")

# ── 2. Scrape today's racecards ─────────────────────────────────────
print("🌐 Scraping today's UK & Ireland racecards from Sporting Life...\n")
racecards = get_scraped_racecards(uk_only=True)

if racecards.empty:
    print("❌ No racecards found for today. Racing may not be scheduled.")
    sys.exit(0)

print(f"📋 Found {len(racecards)} runners across {racecards['race_id'].nunique()} races")
print(f"   Venues: {', '.join(racecards['track'].unique())}")
print()

# ── 3. Process & engineer features ──────────────────────────────────
# Load training data to build historical features
train_path = os.path.join("data", "raw", "race_results.csv")
if os.path.exists(train_path):
    historical = pd.read_csv(train_path)
    combined = pd.concat([historical, racecards], ignore_index=True)
    print(f"📊 Combined {len(historical)} historical + {len(racecards)} racecard records")
else:
    combined = racecards
    print("⚠️  No historical data found — features will be limited")

processed = process_data(combined, save=False)
featured = engineer_features(processed, save=False)

# Filter to just today's entries (finish_position == 0)
today_featured = featured[featured["finish_position"] == 0].copy()

if today_featured.empty:
    print("❌ No processable racecard entries found.")
    sys.exit(0)

print(f"✅ Engineered {len(today_featured)} runners with {featured.shape[1]} features\n")

# ── 4. Predict each race ────────────────────────────────────────────
race_ids = today_featured["race_id"].unique()
print(f"🔮 Predicting {len(race_ids)} races...\n")

for i, rid in enumerate(race_ids):
    race_df = today_featured[today_featured["race_id"] == rid].copy()

    # Get metadata
    track = race_df["track"].iloc[0] if "track" in race_df.columns else "?"
    race_name = race_df["race_name"].iloc[0] if "race_name" in race_df.columns else "?"
    off_time = race_df["off_time"].iloc[0] if "off_time" in race_df.columns else ""
    going = race_df["going"].iloc[0] if "going" in race_df.columns else ""
    dist = race_df["distance_furlongs"].iloc[0] if "distance_furlongs" in race_df.columns else ""

    print("─" * 70)
    print(f"  {off_time}  {track} — {race_name}")
    if going or dist:
        print(f"  Going: {going}  |  Distance: {dist}f  |  Runners: {len(race_df)}")
    print("─" * 70)

    try:
        predictions = predictor.predict_race(race_df)

        # Sort by predicted probability
        if "win_probability" in predictions.columns:
            predictions = predictions.sort_values("win_probability", ascending=False)
        elif "predicted_prob" in predictions.columns:
            predictions = predictions.sort_values("predicted_prob", ascending=False)

        prob_col = "win_probability" if "win_probability" in predictions.columns else "predicted_prob"

        print(f"  {'Rank':<5} {'Horse':<25} {'Win%':>7} {'Odds':>7} {'Value':>8}")
        print(f"  {'─'*5} {'─'*25} {'─'*7} {'─'*7} {'─'*8}")

        for rank, (_, row) in enumerate(predictions.iterrows(), 1):
            name = row.get("horse_name", "?")
            prob = row.get(prob_col, 0) * 100
            odds = row.get("odds", 0)
            implied = 1 / odds if odds > 0 else 0
            value = row.get(prob_col, 0) - implied
            star = " ⭐" if value > 0.05 else ""

            print(f"  {rank:<5} {str(name):<25} {prob:>6.1f}% {odds:>7.1f} {value:>+7.3f}{star}")

    except Exception as e:
        print(f"  ⚠️  Prediction failed: {e}")

    print()

print("=" * 70)
print("  ✅ Trial run complete!")
print("=" * 70)
