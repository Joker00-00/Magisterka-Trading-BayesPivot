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
# KONFIGURACJA BADANIA (ROZSZERZONA)
# ==========================================
SYMBOL = "BTCUSDT"
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"

# Stałe parametry (z Twojego najlepszego wyniku)
SPREAD = 2.0
B_PCT = 0.0020          # Bufor 0.2%
BAYES_MIN_EVENTS = 200  # Warm-up (kluczowe!)
BAYES_WINDOW = 200      # Okno pamięci

# Shorty ustawiamy sztywno na 0.51 (tak jak w Twoim rekordowym wyniku)
# Będziemy badać wpływ progu dla Longów oraz Stop Lossa.
FIXED_SHORT_THRESHOLD = 0.51 

# === ROZSZERZONA SIATKA PARAMETRÓW ===

# 1. Stop Loss (% ceny) - gęstsza siatka od 0.5% do 3.0%
# (Co 0.25%, żeby dokładnie trafić w "sweet spot")
SL_RANGE = [
    0.0050, 0.0075, 0.0100, 0.0125, 0.0150, 
    0.0175, 0.0200, 0.0225, 0.0250, 0.0275, 0.0300
]

# 2. Próg Bayesa Long - szeroki zakres od 0.48 do 0.58
BAYES_LONG_RANGE = [
    0.48, 0.49, 0.50, 0.51, 0.52, 
    0.53, 0.54, 0.55, 0.56, 0.57, 0.58
]

# ==========================================
# SILNIK SYMULACJI (Szybki)
# ==========================================
def run_simulation_fast(df: pd.DataFrame, sl_pct: float, bayes_long_threshold: float):
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)
    
    current_equity = 0.0
    history_s1 = deque(maxlen=BAYES_WINDOW)
    history_r1 = deque(maxlen=BAYES_WINDOW)
    
    # Pre-calc stałej
    min_events = BAYES_MIN_EVENTS
    
    for i in range(1, n):
        h_prev, l_prev, c_prev = highs[i-1], lows[i-1], closes[i-1]
        h_curr, l_curr, c_curr = highs[i], lows[i], closes[i]
        
        PP = (h_prev + l_prev + c_prev) / 3.0
        R1 = 2 * PP - l_prev
        S1 = 2 * PP - h_prev
        
        current_b = c_prev * B_PCT
        current_sl = c_prev * sl_pct
        
        # Bayes Prob
        p_r1 = np.mean(history_r1) if len(history_r1) >= min_events else 0.5
        p_s1 = np.mean(history_s1) if len(history_s1) >= min_events else 0.5
        
        # Sygnały
        event_short = c_prev > (R1 - current_b)
        event_long = c_prev < (S1 + current_b)
        
        if event_short and event_long:
            event_short, event_long = False, False

        # Decyzja
        trade_dir = None
        
        # Short (Stały próg 0.51)
        if event_short and p_r1 > FIXED_SHORT_THRESHOLD:
            trade_dir = 'SHORT'
        # Long (Badany z siatki)
        elif event_long and p_s1 > bayes_long_threshold:
            trade_dir = 'LONG'

        # Egzekucja
        # Short
        raw_pnl_short = c_prev - c_curr
        is_sl_short = (h_curr - c_prev) > current_sl
        real_pnl_short = (-current_sl - SPREAD) if is_sl_short else (raw_pnl_short - SPREAD)
        
        # Long
        raw_pnl_long = c_curr - c_prev
        is_sl_long = (c_prev - l_curr) > current_sl
        real_pnl_long = (-current_sl - SPREAD) if is_sl_long else (raw_pnl_long - SPREAD)
        
        # Uczenie Bayesa (zawsze!)
        if event_short: history_r1.append(1 if real_pnl_short > 0 else 0)
        if event_long: history_s1.append(1 if real_pnl_long > 0 else 0)
            
        # Wynik
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
        print(f"Dane gotowe: {len(df)} świec.")
        print(f"Rozpoczynam Extended Grid Search ({total_combinations} kombinacji)...")
        print(f"Parametry stałe: Warmup={BAYES_MIN_EVENTS}, ShortThreshold={FIXED_SHORT_THRESHOLD}")
        
        results = np.zeros((len(BAYES_LONG_RANGE), len(SL_RANGE)))
        
        # Pętla po parametrach
        for i, b_thresh in enumerate(BAYES_LONG_RANGE):
            for j, sl in enumerate(SL_RANGE):
                profit = run_simulation_fast(df, sl, b_thresh)
                results[i, j] = profit
                
                # Progress bar w konsoli
                idx = i * len(SL_RANGE) + j + 1
                if idx % 5 == 0:
                    print(f"Postęp: {idx}/{total_combinations} ({idx/total_combinations:.1%})", end='\r')

        print("\nGenerowanie wykresu...")
        
        # Rysowanie Heatmapy
        plt.figure(figsize=(14, 10))
        
        xticklabels = [f"{x*100:.2f}%" for x in SL_RANGE]
        yticklabels = [f"{y:.2f}" for y in BAYES_LONG_RANGE]
        
        # Odwracamy oś Y (rosnąco w górę)
        sns.heatmap(results[::-1], annot=True, fmt=".0f", cmap="RdYlGn", 
                    xticklabels=xticklabels, yticklabels=yticklabels[::-1])
        
        plt.title(f"EXTENDED Profit Heatmap (USD) - {SYMBOL} 2024\nWarmup=200, ShortThreshold={FIXED_SHORT_THRESHOLD}")
        plt.xlabel("Stop Loss (% ceny)")
        plt.ylabel("Bayes Threshold Long")
        
        plt.tight_layout()
        plt.savefig("heatmap_extended.png")
        plt.show()
        
        print("Gotowe! Zapisano wykres jako: heatmap_extended.png")
        
    except Exception as e:
        print(f"Błąd: {e}")
