from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from datetime import datetime

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
# Import loadera (musi być w tym samym folderze)
try:
    from src.data_loader import load_bars
except ImportError:
    print("Błąd: Brak pliku data_loader.py.")
    exit()

# ==========================================
# KONFIGURACJA BADANIA
# ==========================================
SYMBOL = "BTCUSDT"
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"

# Stałe parametry
SPREAD = 2.0
B_PCT = 0.0020            # Bufor 0.2%
BAYES_THRESHOLD_SHORT = 0.60 # Shorty wycięte
BAYES_WINDOW = 200

# SIATKA PARAMETRÓW DO PRZEBADANIA
# 1. Stop Loss (% ceny)
SL_RANGE = [0.005, 0.010, 0.015, 0.020, 0.025] 
# (0.5%, 1.0%, 1.5%, 2.0%, 2.5%)

# 2. Próg Bayesa Long
BAYES_LONG_RANGE = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55]

# ==========================================
# SILNIK STRATEGII (wersja 'silent' - szybka)
# ==========================================
def run_simulation_fast(df: pd.DataFrame, sl_pct: float, bayes_long_threshold: float):
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)
    
    current_equity = 0.0
    
    history_s1 = deque(maxlen=BAYES_WINDOW)
    history_r1 = deque(maxlen=BAYES_WINDOW)
    
    bayes_min_events = 10
    
    for i in range(1, n):
        h_prev, l_prev, c_prev = highs[i-1], lows[i-1], closes[i-1]
        
        # --- DANE BIEŻĄCE ---
        h_curr, l_curr, c_curr = highs[i], lows[i], closes[i]
        
        # Pivoty
        PP = (h_prev + l_prev + c_prev) / 3.0
        R1 = 2 * PP - l_prev
        S1 = 2 * PP - h_prev
        
        # Parametry dynamiczne
        current_b = c_prev * B_PCT
        current_sl = c_prev * sl_pct
        
        # Bayes Prob
        p_r1 = np.mean(history_r1) if len(history_r1) >= bayes_min_events else 0.5
        p_s1 = np.mean(history_s1) if len(history_s1) >= bayes_min_events else 0.5
        
        # Sygnały
        event_short = c_prev > (R1 - current_b)
        event_long = c_prev < (S1 + current_b)
        
        if event_short and event_long:
            event_short, event_long = False, False

        # Decyzja
        trade_dir = None
        
        if event_short and p_r1 > BAYES_THRESHOLD_SHORT:
            trade_dir = 'SHORT'
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
        
        # Uczenie Bayesa
        if event_short: history_r1.append(1 if real_pnl_short > 0 else 0)
        if event_long: history_s1.append(1 if real_pnl_long > 0 else 0)
            
        # Wynik portfela
        if trade_dir == 'SHORT': current_equity += real_pnl_short
        elif trade_dir == 'LONG': current_equity += real_pnl_long
            
    return current_equity

# ==========================================
# MAIN - GRID SEARCH
# ==========================================
if __name__ == "__main__":
    # 1. Pobranie danych
    print(f"Pobieranie danych {SYMBOL}...")
    try:
        sy, sm = int(START_DATE[:4]), int(START_DATE[5:7])
        ey, em = int(END_DATE[:4]), int(END_DATE[5:7])
        
        df = load_bars(SYMBOL, "1h", sy, sm, ey, em)
        df = df.loc[START_DATE:END_DATE]
        
        if df.empty:
            print("Brak danych!")
            exit()
            
        print(f"Dane gotowe: {len(df)} świec. Rozpoczynam Grid Search...")
        print(f"Liczba kombinacji: {len(SL_RANGE) * len(BAYES_LONG_RANGE)}")
        
        # Macierz wyników
        results = np.zeros((len(BAYES_LONG_RANGE), len(SL_RANGE)))
        
        # 2. Pętla Grid Search
        for i, b_thresh in enumerate(BAYES_LONG_RANGE):
            for j, sl in enumerate(SL_RANGE):
                profit = run_simulation_fast(df, sl, b_thresh)
                results[i, j] = profit
                print(f"Testing: Bayes={b_thresh}, SL={sl*100:.1f}% -> Profit: {profit:.2f}")

        # 3. Rysowanie Heatmapy
        plt.figure(figsize=(10, 8))
        
        xticklabels = [f"{x*100:.1f}%" for x in SL_RANGE]
        yticklabels = [f"{y:.2f}" for y in BAYES_LONG_RANGE]
        
        # Odwracamy oś Y, żeby wyższe wartości Bayesa były na górze
        sns.heatmap(results[::-1], annot=True, fmt=".0f", cmap="RdYlGn", 
                    xticklabels=xticklabels, yticklabels=yticklabels[::-1])
        
        plt.title(f"Profit Heatmap (USD) - {SYMBOL} {START_DATE[:4]}\nShort Threshold={BAYES_THRESHOLD_SHORT}, Buffer={B_PCT*100:.1f}%")
        plt.xlabel("Stop Loss (%)")
        plt.ylabel("Bayes Threshold Long")
        
        plt.tight_layout()
        plt.savefig("heatmap_profit.png")
        plt.show()
        
        print("\nZakończono. Wykres zapisany jako heatmap_profit.png")
        
    except Exception as e:
        print(f"Błąd: {e}")
