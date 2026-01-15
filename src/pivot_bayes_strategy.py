"""
pivot_bayes_strategy.py
-----------------------------
Poprawiona wersja strategii z implementacją
Prawdziwej Estymacji Bayesowskiej (Beta-Binomial Conjugate Prior).
"""

from __future__ import annotations
import os
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import loadera
try:
    from data_loader import load_bars
except ImportError:
    print("Błąd: Brak pliku data_loader.py.")
    exit()

# ==========================================
# KONFIGURACJA
# ==========================================
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"
SYMBOL = "BNBUSDT"
INTERVAL = "1h"
SPREAD = 0.05  # Koszt transakcyjny (Spread) w USD

# Parametry Strategii
SL_PCT = 0.015
B_PCT = 0.0040
BAYES_WINDOW = 200
BAYES_MIN_EVENTS = 50  # Zmniejszyłem, bo Bayes działa lepiej na małych próbkach niż średnia

# Progi decyzyjne (Probability Thresholds)
BAYES_THRESHOLD_LONG = 0.0
BAYES_THRESHOLD_SHORT = 0.0

# Parametry Bayesa (Prior - Rozkład A Priori)
# Alpha=1.0, Beta=1.0 to rozkład jednostajny (Uniform Prior) - "brak wiedzy początkowej"
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0

OUTPUT_DIR = "trades"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# FUNKCJE POMOCNICZE
# ==========================================
def parse_date_range(start_str, end_str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    return start.year, start.month, end.year, end.month

def calculate_max_drawdown(equity_curve):
    peak = -999999999.0
    max_dd = 0.0
    for value in equity_curve:
        if value > peak: peak = value
        dd = value - peak
        if dd < max_dd: max_dd = dd
    return max_dd

def calculate_calmar_ratio(total_profit, max_dd):
    if max_dd == 0: return 0.0
    return total_profit / abs(max_dd)

def calculate_global_stats(df: pd.DataFrame):
    """
    Analiza 'A Priori' na całej historii (statystyka klasyczna dla porównania).
    """
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    
    kl, ks = 0, 0
    kr, klr = 0, 0
    ks_s1, kls = 0, 0

    for i in range(1, n):
        h_prev, l_prev, c_prev = highs[i-1], lows[i-1], closes[i-1]
        c_curr = closes[i]
        
        PP = (h_prev + l_prev + c_prev) / 3.0
        R1, S1 = 2*PP - l_prev, 2*PP - h_prev
        
        current_b = c_prev * B_PCT
        
        if c_curr > c_prev: kl += 1
        elif c_curr < c_prev: ks += 1
            
        if c_prev > (R1 - current_b):
            kr += 1
            if c_curr < c_prev: klr += 1
                
        if c_prev < (S1 + current_b):
            ks_s1 += 1
            if c_curr > c_prev: kls += 1

    prob_up = kl / (kl + ks) if (kl+ks) > 0 else 0
    plr_r1 = klr / kr if kr > 0 else 0
    plr_s1 = kls / ks_s1 if ks_s1 > 0 else 0

    print("\n=== ANALIZA GLOBALNA (FREQUENTIST / NAIVE) ===")
    print(f"Zakres: {START_DATE} do {END_DATE}")
    print(f"Bias rynku (Bullish): {prob_up:.2%}")
    print(f"[R1 SHORT] Skuteczność historyczna: {plr_r1:.2%} (na {kr} okazji)")
    print(f"[S1 LONG]  Skuteczność historyczna: {plr_s1:.2%} (na {ks_s1} okazji)")
    print("==============================================\n")

# ==========================================
# STRATEGIA BAYESOWSKA (POPRAWIONA)
# ==========================================
def run_strategy(df: pd.DataFrame):
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    times = df.index
    n = len(df)
    
    equity_curve = [0.0]
    trades_log = []
    
    # Historie wyników (0 = porażka, 1 = sukces)
    history_r1 = deque(maxlen=BAYES_WINDOW)
    history_s1 = deque(maxlen=BAYES_WINDOW)
    current_equity = 0.0
    
    print(f"Start symulacji Bayesowskiej...")

    for i in range(1, n):
        h_prev, l_prev, c_prev = highs[i-1], lows[i-1], closes[i-1]
        o_curr = opens[i]
        h_curr, l_curr, c_curr = highs[i], lows[i], closes[i]
        
        PP = (h_prev + l_prev + c_prev) / 3.0
        R1 = 2*PP - l_prev
        S1 = 2*PP - h_prev
        
        current_b = c_prev * B_PCT
        current_sl = c_prev * SL_PCT
        
        # ---------------------------------------------------------
        # ESTYMACJA BAYESOWSKA (BETA-BINOMIAL)
        # ---------------------------------------------------------
        # Zamiast zwykłej średniej (np.mean), liczymy Wartość Oczekiwaną Rozkładu A Posteriori.
        # Wzór: E[theta | data] = (alpha + k) / (alpha + beta + n)
        # Gdzie: k = liczba sukcesów, n = liczba prób
        
        # Obliczenia dla R1 (Short)
        n_r1 = len(history_r1)
        k_r1 = sum(history_r1) # Liczba wygranych shortów w oknie
        if n_r1 >= BAYES_MIN_EVENTS:
            p_r1 = (PRIOR_ALPHA + k_r1) / (PRIOR_ALPHA + PRIOR_BETA + n_r1)
        else:
            p_r1 = 0.5 # Neutralne 50% przed zebraniem minimalnej próbki

        # Obliczenia dla S1 (Long)
        n_s1 = len(history_s1)
        k_s1 = sum(history_s1) # Liczba wygranych longów w oknie
        if n_s1 >= BAYES_MIN_EVENTS:
            p_s1 = (PRIOR_ALPHA + k_s1) / (PRIOR_ALPHA + PRIOR_BETA + n_s1)
        else:
            p_s1 = 0.5
        # ---------------------------------------------------------
        
        event_short = c_prev > (R1 - current_b)
        event_long = c_prev < (S1 + current_b)
        if event_short and event_long: event_short, event_long = False, False

        trade_dir = None
        # Decyzja na podstawie Prawdopodobieństwa A Posteriori
        if event_short and p_r1 > BAYES_THRESHOLD_SHORT: trade_dir = 'SHORT'
        elif event_long and p_s1 > BAYES_THRESHOLD_LONG: trade_dir = 'LONG'

        # Obliczenia PnL (Open-based)
        raw_pnl_short = o_curr - c_curr
        is_sl_short = h_curr > (o_curr + current_sl)
        real_pnl_short = (-current_sl - SPREAD) if is_sl_short else (raw_pnl_short - SPREAD)
        
        raw_pnl_long = c_curr - o_curr
        is_sl_long = l_curr < (o_curr - current_sl)
        real_pnl_long = (-current_sl - SPREAD) if is_sl_long else (raw_pnl_long - SPREAD)
        
        # Uczenie modelu (Aktualizacja Likelihood)
        if event_short: history_r1.append(1 if real_pnl_short > 0 else 0)
        if event_long: history_s1.append(1 if real_pnl_long > 0 else 0)
            
        if trade_dir == 'SHORT':
            current_equity += real_pnl_short
            trades_log.append({'time': times[i], 'type': 'SHORT', 'price': o_curr, 'pnl': real_pnl_short, 'equity': current_equity, 'prob': p_r1})
        elif trade_dir == 'LONG':
            current_equity += real_pnl_long
            trades_log.append({'time': times[i], 'type': 'LONG', 'price': o_curr, 'pnl': real_pnl_long, 'equity': current_equity, 'prob': p_s1})
            
        equity_curve.append(current_equity)

    return pd.DataFrame(trades_log), equity_curve

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    try:
        sy, sm, ey, em = parse_date_range(START_DATE, END_DATE)
        print(f"Pobieranie: {SYMBOL}...")
        df = load_bars(SYMBOL, INTERVAL, sy, sm, ey, em).loc[START_DATE:END_DATE]
        
        if df.empty:
            print("Brak danych.")
            exit()
            
        calculate_global_stats(df)
        trades_df, equity = run_strategy(df)
        
        if not trades_df.empty:
            total_profit = equity[-1]
            max_dd = calculate_max_drawdown(equity)
            calmar = calculate_calmar_ratio(total_profit, max_dd)         
            trades_df.to_csv(f"trades/trades_{SYMBOL}_RAW.csv")
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
            #print(f"\n=== WYNIK STRATEGII BAYESOWSKIEJ ===")
            print(f"\n=== WYNIK STRATEGII PIVOT POINTS ===")
            print(f"Zakres: {START_DATE} do {END_DATE}")
            print(f"Symbol: {SYMBOL}")
            #print(f"Prior: Beta({PRIOR_ALPHA},{PRIOR_BETA})")
            #print(f"Threshold LONG: {BAYES_THRESHOLD_LONG:.2f}, SHORT: {BAYES_THRESHOLD_SHORT:.2f}")
            print(f"Profit: {total_profit:.2f} USD")
            print(f"MaxDD:  {max_dd:.2f} USD")
            print(f"Calmar: {calmar:.2f}")
            print(f"Liczba transakcji: {total_trades}")
            print(f"Win Rate: {win_rate:.2f}%")
            # ---------------------------------------------
            # WYKRES EQUITY
            # ---------------------------------------------
            equity_plot = equity[1:] 
            min_len = min(len(df)-1, len(equity_plot))
            equity_series = pd.Series(equity_plot[:min_len], index=df.index[1:min_len+1])

            plt.figure(figsize=(14, 6))
            #plt.plot(equity_series.index, equity_series.values, label='Bayesian Equity', color='blue', linewidth=1.5)
            plt.plot(equity_series.index, equity_series.values, label='RAW Equity', color='blue', linewidth=1.5)
            #plt.title(f"Bayesian Strategy Equity (Prior: Beta({PRIOR_ALPHA},{PRIOR_BETA}))")
            plt.title(f"RAW Strategy Equity - {SYMBOL}")
            plt.ylabel("Profit ($)")
            plt.xlabel("Date")
            
            info_text = (f"Profit: ${total_profit:.0f}\n"
                         f"Calmar: {calmar:.2f}")
            plt.annotate(info_text, xy=(0.02, 0.95), xycoords='axes fraction',
                         fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
                         verticalalignment='top')

            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            #plt.savefig("equity_bayes.png")
            plt.savefig("equity_RAW.png")
            #print("Zapisano: equity_bayes.png")
            print("Zapisano: equity_RAW.png")
            plt.show()

    except Exception as e:
        print(f"Błąd: {e}")
