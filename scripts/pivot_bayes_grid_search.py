"""
pivot_bayes_grid_search.py

Zaktualizowana wersja Grid Search:
- Implementacja pełnego Bayesa (Beta-Binomial)
- Egzekucja transakcji na Open (zgodnie ze strategią)
"""

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

# Import loadera (musi być w tym samym folderze lub w ścieżce)
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

# Stałe parametry (zgodne ze strategią)
SPREAD = 2.0
B_PCT = 0.0020            # Bufor 0.2%
BAYES_THRESHOLD_SHORT = 0.52 # Zgodnie ze strategią (było 0.60, ale strategia ma 0.52)
BAYES_WINDOW = 200
BAYES_MIN_EVENTS = 50     # Zgodnie ze strategią

# Parametry Bayesa (Prior - Rozkład A Priori)
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0

# SIATKA PARAMETRÓW DO PRZEBADANIA
# 1. Stop Loss (% ceny)
SL_RANGE = [0.005, 0.010, 0.015, 0.020, 0.025]
# (0.5%, 1.0%, 1.5%, 2.0%, 2.5%)

# 2. Próg Bayesa Long
BAYES_LONG_RANGE = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55]

# ==========================================
# SILNIK STRATEGII (Zaktualizowany)
# ==========================================

def run_simulation_fast(df: pd.DataFrame, sl_pct: float, bayes_long_threshold: float):
    # Pobieramy tablice numpy dla szybkości
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    n = len(df)
    current_equity = 0.0
    
    history_s1 = deque(maxlen=BAYES_WINDOW)
    history_r1 = deque(maxlen=BAYES_WINDOW)
    
    # Pre-calc stałej mianownika Bayesa
    bayes_denom_const = PRIOR_ALPHA + PRIOR_BETA
    
    for i in range(1, n):
        # DANE POPRZEDNIE (do sygnału)
        h_prev, l_prev, c_prev = highs[i-1], lows[i-1], closes[i-1]
        
        # DANE BIEŻĄCE (do egzekucji)
        o_curr = opens[i]
        h_curr, l_curr, c_curr = highs[i], lows[i], closes[i]
        
        # Pivoty
        PP = (h_prev + l_prev + c_prev) / 3.0
        R1 = 2 * PP - l_prev
        S1 = 2 * PP - h_prev
        
        # Parametry dynamiczne
        current_b = c_prev * B_PCT
        current_sl = c_prev * sl_pct
        
        # --- BAYES PROBABILITY (BETA-BINOMIAL) ---
        # Short (R1)
        n_r1 = len(history_r1)
        k_r1 = sum(history_r1)
        if n_r1 >= BAYES_MIN_EVENTS:
            p_r1 = (PRIOR_ALPHA + k_r1) / (bayes_denom_const + n_r1)
        else:
            p_r1 = 0.5
            
        # Long (S1)
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
            
        # Decyzja
        trade_dir = None
        
        if event_short and p_r1 > BAYES_THRESHOLD_SHORT:
            trade_dir = 'SHORT'
        elif event_long and p_s1 > bayes_long_threshold:
            trade_dir = 'LONG'
            
        # --- EGZEKUCJA (OPEN-BASED) ---
        
        # Short Logic
        # Wchodzimy na Open, SL liczymy od Open
        raw_pnl_short = o_curr - c_curr
        # Sprawdzamy czy High przebił SL (Entry + SL_dist)
        is_sl_short = h_curr > (o_curr + current_sl)
        real_pnl_short = (-current_sl - SPREAD) if is_sl_short else (raw_pnl_short - SPREAD)
        
        # Long Logic
        # Wchodzimy na Open, SL liczymy od Open
        raw_pnl_long = c_curr - o_curr
        # Sprawdzamy czy Low przebił SL (Entry - SL_dist)
        is_sl_long = l_curr < (o_curr - current_sl)
        real_pnl_long = (-current_sl - SPREAD) if is_sl_long else (raw_pnl_long - SPREAD)
        
        # Uczenie Bayesa
        # Dodajemy wynik do historii, jeśli wystąpił sygnał techniczny (event),
        # niezależnie od tego czy weszliśmy w pozycję (filtrowanie Bayesa).
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