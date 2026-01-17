from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from datetime import datetime
import os, sys

# Ścieżka importu (zabezpieczenie)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.data_loader import load_bars
except ImportError:
    # Fallback dla lokalnego uruchamiania bez struktury src
    try:
        from data_loader import load_bars
    except ImportError:
        print("Błąd: Brak pliku data_loader.py.")
        exit()

# ==========================================
# KONFIGURACJA BADANIA
# ==========================================

SYMBOL = "BTCUSDT"
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

# Stałe parametry
SPREAD = 2.0
B_PCT = 0.0020             # Bufor 0.2%
BAYES_WINDOW = 200

# WAŻNE: Ustawiamy Short na 0.0, żeby test był czysty (Baseline = brak filtru)
BAYES_THRESHOLD_SHORT = 0.0 # Shorty wycięte

# SIATKA PARAMETRÓW DO PRZEBADANIA
# 1. Stop Loss (% ceny)
SL_RANGE = [0.005, 0.010, 0.015, 0.020, 0.025] 
# (0.5%, 1.0%, 1.5%, 2.0%, 2.5%)

# 2. Próg Bayesa Long
# 0.0 = RAW (Baseline), reszta to filtr Bayesowski
BAYES_LONG_RANGE = [0.0, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55]

# ==========================================
# SILNIK STRATEGII (wersja 'fast' z Drawdownem)
# ==========================================

def run_simulation_fast(df: pd.DataFrame, sl_pct: float, bayes_long_threshold: float):
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)
    
    current_equity = 0.0
    equity_curve = [0.0]  # Lista do śledzenia krzywej kapitału
    
    history_s1 = deque(maxlen=BAYES_WINDOW)
    history_r1 = deque(maxlen=BAYES_WINDOW)
    bayes_min_events = 10
    total_trades = 0
    
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
        
        # Zabezpieczenie: Jeśli próg > 0.0, używamy go. Jeśli 0.0 -> wchodzimy zawsze (RAW).
        # Short
        if event_short:
            if BAYES_THRESHOLD_SHORT == 0.0:
                trade_dir = 'SHORT' # Tryb RAW
            elif p_r1 > BAYES_THRESHOLD_SHORT:
                trade_dir = 'SHORT' # Tryb Bayes
                
        # Long
        elif event_long:
            if bayes_long_threshold == 0.0:
                trade_dir = 'LONG' # Tryb RAW
            elif p_s1 > bayes_long_threshold:
                trade_dir = 'LONG' # Tryb Bayes

        # Egzekucja
        pnl = 0.0
        if trade_dir:
            total_trades += 1
            if trade_dir == 'SHORT':
                raw_pnl_short = c_prev - c_curr
                is_sl_short = (h_curr - c_prev) > current_sl
                real_pnl_short = (-current_sl - SPREAD) if is_sl_short else (raw_pnl_short - SPREAD)
                pnl = real_pnl_short
                if event_short: history_r1.append(1 if real_pnl_short > 0 else 0)
                
            elif trade_dir == 'LONG':
                raw_pnl_long = c_curr - c_prev
                is_sl_long = (c_prev - l_curr) > current_sl
                real_pnl_long = (-current_sl - SPREAD) if is_sl_long else (raw_pnl_long - SPREAD)
                pnl = real_pnl_long
                if event_long: history_s1.append(1 if real_pnl_long > 0 else 0)
            
            current_equity += pnl
            equity_curve.append(current_equity)

    # --- OBLICZANIE DRAWDOWNU ---
    equity_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = peak - equity_arr
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0
    
    # Zwracamy słownik z pełnymi danymi
    return {
        "total_profit": current_equity,
        "max_drawdown": max_dd,
        "total_trades": total_trades
    }

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
        
        # Macierz wyników dla Heatmapy
        results_matrix = np.zeros((len(BAYES_LONG_RANGE), len(SL_RANGE)))
        
        # Lista do tabeli szczegółowej
        detailed_results = []
        
        # 2. Pętla Grid Search
        for i, b_thresh in enumerate(BAYES_LONG_RANGE):
            for j, sl in enumerate(SL_RANGE):
                
                # Uruchomienie symulacji
                res = run_simulation_fast(df, sl, b_thresh)
                
                profit = res['total_profit']
                dd = res['max_drawdown']
                trades = res['total_trades']
                
                # Calmar Ratio (uproszczony: Profit / MaxDD)
                calmar = profit / dd if dd > 0 else 0.0
                
                # Zapis do macierzy (do wykresu)
                results_matrix[i, j] = profit
                
                # Zapis do listy (do tabeli)
                detailed_results.append({
                    "Bayes": b_thresh,
                    "SL": sl,
                    "Profit": profit,
                    "DD": dd,
                    "Trades": trades,
                    "Calmar": calmar
                })
                
                print(f"Testing: Bayes={b_thresh:.2f}, SL={sl*100:.1f}% -> Profit: {profit:8.2f} | DD: {dd:8.2f} | Trades: {trades:4d}")

        # 3. Rysowanie Heatmapy (tylko Zysk)
        plt.figure(figsize=(10, 8))
        xticklabels = [f"{x*100:.1f}%" for x in SL_RANGE]
        yticklabels = [f"{y:.2f}" for y in BAYES_LONG_RANGE]
        
        sns.heatmap(results_matrix[::-1], annot=True, fmt=".0f", cmap="RdYlGn",
                   xticklabels=xticklabels, yticklabels=yticklabels[::-1])
        
        plt.title(f"Profit Heatmap (USD) - {SYMBOL} {START_DATE[:4]}-{END_DATE[:4]}\nShort Threshold={BAYES_THRESHOLD_SHORT}, Buffer={B_PCT*100:.1f}%")
        plt.xlabel("Stop Loss (%)")
        plt.ylabel("Bayes Threshold Long")
        plt.tight_layout()
        plt.savefig("heatmap_profit.png")
        plt.show()
        print("\n[INFO] Wykres zapisany jako heatmap_profit.png")
        
        # 4. Generowanie Tabeli TOP 10 (wg Profitu i wg Calmara)
        df_res = pd.DataFrame(detailed_results)
        
        print("\n" + "="*60)
        print("TOP 10 WYNIKÓW WEDŁUG ZYSKU (PROFIT):")
        print("="*60)
        top_profit = df_res.sort_values(by="Profit", ascending=False).head(10)
        print(top_profit.to_string(index=False))
        
        print("\n" + "="*60)
        print("TOP 10 WYNIKÓW WEDŁUG CALMAR RATIO (BEZPIECZEŃSTWO):")
        print("="*60)
        # Filtrujemy, żeby odrzucić ujemne zyski przy liczeniu Calmara
        top_calmar = df_res[df_res['Profit'] > 0].sort_values(by="Calmar", ascending=False).head(10)
        print(top_calmar.to_string(index=False))
        
    except Exception as e:
        print(f"Błąd krytyczny: {e}")
        import traceback
        traceback.print_exc()
