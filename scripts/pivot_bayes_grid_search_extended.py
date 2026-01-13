"""
pivot_bayes_grid_search_extended.py

Wersja "Snajper":
- Skupiona na obszarach zidentyfikowanych jako zyskowne (High Bayes, Low Buffer)
- Pełny Bayes (Beta-Binomial) + Open Exec
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import loadera
try:
    from src.data_loader import load_bars
except ImportError:
    print("Błąd: Brak pliku data_loader.py.")
    exit()

# ==========================================
# KONFIGURACJA (EXTENDED)
# ==========================================

SYMBOL = "BTCUSDT"
START_DATE = "2024-01-01" # Skupiamy się na 2024
END_DATE = "2024-12-31"

SPREAD = 2.0
B_PCT = 0.0035            # Wypośrodkowane "Sweet Spot" (między 0.1% a 0.2%)
BAYES_MIN_EVENTS = 50 
BAYES_WINDOW = 200

# Parametry Bayesa (Prior)
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0

# Stały próg Short (zakładamy, że 0.52 jest OK)
FIXED_SHORT_THRESHOLD = 0.52

# === ROZSZERZONA SIATKA (Fine Tuning) ===

# 1. SL: Skupiamy się na zakresie 1% - 3%, ale gęściej
SL_RANGE = [
    0.010, 0.012, 0.014, 0.016, 0.018, 
    0.020, 0.022, 0.024, 0.026, 0.028, 0.030, 0.032, 0.034, 0.036, 0.038, 0.040
]

# 2. Bayes Long: Badamy wysokie progi (0.54 - 0.60), bo tam było zielono
BAYES_LONG_RANGE = [
    0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60
]

# ==========================================
# SILNIK
# ==========================================

def run_simulation_fast(df: pd.DataFrame, sl_pct: float, bayes_long_threshold: float):
    
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    n = len(df)
    current_equity = 0.0
    
    history_s1 = deque(maxlen=BAYES_WINDOW)
    history_r1 = deque(maxlen=BAYES_WINDOW)
    
    bayes_denom_const = PRIOR_ALPHA + PRIOR_BETA
    
    for i in range(1, n):
        h_prev, l_prev, c_prev = highs[i-1], lows[i-1], closes[i-1]
        o_curr = opens[i]
        h_curr, l_curr, c_curr = highs[i], lows[i], closes[i]
        
        PP = (h_prev + l_prev + c_prev) / 3.0
        R1 = 2 * PP - l_prev
        S1 = 2 * PP - h_prev
        
        current_b = c_prev * B_PCT
        current_sl = c_prev * sl_pct
        
        # Bayes Prob
        n_r1 = len(history_r1)
        k_r1 = sum(history_r1)
        if n_r1 >= BAYES_MIN_EVENTS:
            p_r1 = (PRIOR_ALPHA + k_r1) / (bayes_denom_const + n_r1)
        else:
            p_r1 = 0.5
            
        n_s1 = len(history_s1)
        k_s1 = sum(history_s1)
        if n_s1 >= BAYES_MIN_EVENTS:
            p_s1 = (PRIOR_ALPHA + k_s1) / (bayes_denom_const + n_s1)
        else:
            p_s1 = 0.5

        # Sygnały
        event_short = c_prev > (R1 - current_b)
        event_long = c_prev < (S1 + current_b)
        
        if event_short and event_long:
            event_short, event_long = False, False
            
        trade_dir = None
        if event_short and p_r1 > FIXED_SHORT_THRESHOLD:
            trade_dir = 'SHORT'
        elif event_long and p_s1 > bayes_long_threshold:
            trade_dir = 'LONG'
            
        # Exec (Open)
        raw_pnl_short = o_curr - c_curr
        is_sl_short = h_curr > (o_curr + current_sl)
        real_pnl_short = (-current_sl - SPREAD) if is_sl_short else (raw_pnl_short - SPREAD)
        
        raw_pnl_long = c_curr - o_curr
        is_sl_long = l_curr < (o_curr - current_sl)
        real_pnl_long = (-current_sl - SPREAD) if is_sl_long else (raw_pnl_long - SPREAD)
        
        if event_short: history_r1.append(1 if real_pnl_short > 0 else 0)
        if event_long: history_s1.append(1 if real_pnl_long > 0 else 0)
        
        if trade_dir == 'SHORT': current_equity += real_pnl_short
        elif trade_dir == 'LONG': current_equity += real_pnl_long
        
    return current_equity

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print(f"Pobieranie danych {SYMBOL}...")
    try:
        sy, sm = int(START_DATE[:4]), int(START_DATE[5:7])
        ey, em = int(END_DATE[:4]), int(END_DATE[5:7])
        
        df = load_bars(SYMBOL, "1h", sy, sm, ey, em)
        df = df.loc[START_DATE:END_DATE]
        
        if df.empty:
            print("Brak danych!")
            exit()
            
        total_combinations = len(SL_RANGE) * len(BAYES_LONG_RANGE)
        print(f"Start Extended Grid Search ({total_combinations} komb.)...")
        print(f"Buffer={B_PCT*100:.2f}%, ShortThreshold={FIXED_SHORT_THRESHOLD}")
        
        results = np.zeros((len(BAYES_LONG_RANGE), len(SL_RANGE)))
        
        for i, b_thresh in enumerate(BAYES_LONG_RANGE):
            for j, sl in enumerate(SL_RANGE):
                profit = run_simulation_fast(df, sl, b_thresh)
                results[i, j] = profit
                
            # Progress bar
            print(f"Postęp: {(i+1)/len(BAYES_LONG_RANGE):.0%}", end='\r')
            
        print("\nGenerowanie wykresu...")
        
        plt.figure(figsize=(14, 10))
        
        xticklabels = [f"{x*100:.1f}%" for x in SL_RANGE]
        yticklabels = [f"{y:.2f}" for y in BAYES_LONG_RANGE]
        
        sns.heatmap(results[::-1], annot=True, fmt=".0f", cmap="RdYlGn",
                   xticklabels=xticklabels, yticklabels=yticklabels[::-1])
        
        plt.title(f"EXTENDED Profit Heatmap (Fine Tuning) - {SYMBOL} 2024\nBuffer={B_PCT*100:.2f}%, ShortThreshold={FIXED_SHORT_THRESHOLD}")
        plt.xlabel("Stop Loss (%)")
        plt.ylabel("Bayes Threshold Long")
        
        plt.tight_layout()
        plt.savefig("heatmap_extended.png")
        plt.show()
        
        print("Gotowe! Zapisano wykres jako: heatmap_extended.png")
        
    except Exception as e:
        print(f"Błąd: {e}")
