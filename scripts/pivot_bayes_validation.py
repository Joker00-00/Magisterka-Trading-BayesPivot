from __future__ import annotations
import numpy as np
import pandas as pd
from collections import deque
import os, sys

# ---------------------------------------------------------
# KONFIGURACJA
# ---------------------------------------------------------
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

# Parametry wybrane w Grid Search (z Tabeli 6)
BEST_SL = 0.005          # 0.5% (Zwycięzca RAW i bezpieczny Bayes)
BEST_BAYES_THRESH = 0.50 # Najlepszy próg Bayesa
BAYES_WINDOW = 200       # Stałe okno

SPREAD_DICT = {
    "BTCUSDT": 2.0,
    "ETHUSDT": 0.20,
    "BNBUSDT": 0.05,
    "SOLUSDT": 0.03
}

# ---------------------------------------------------------
# IMPORT LOADERA
# ---------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.data_loader import load_bars
except ImportError:
    print("Błąd: Brak pliku data_loader.py w src/")
    exit()

# ---------------------------------------------------------
# FUNKCJA SYMULACJI (Skrócona)
# ---------------------------------------------------------
def run_strategy(df, symbol, sl_pct, bayes_threshold):
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)
    
    equity = 0.0
    max_equity = 0.0
    drawdown = 0.0
    max_drawdown = 0.0
    trades = 0
    wins = 0
    
    history_s1 = deque(maxlen=BAYES_WINDOW)
    history_r1 = deque(maxlen=BAYES_WINDOW)
    min_events = 10
    spread = SPREAD_DICT.get(symbol, 0.01)
    
    for i in range(1, n):
        h_prev, l_prev, c_prev = highs[i-1], lows[i-1], closes[i-1]
        h_curr, l_curr, c_curr = highs[i], lows[i], closes[i]
        
        # Pivoty
        pp = (h_prev + l_prev + c_prev) / 3.0
        r1 = 2 * pp - l_prev
        s1 = 2 * pp - h_prev
        
        # Warunki (Cena przecina poziom)
        event_short = (c_prev < r1) and (c_curr > r1) # Przebicie R1 w górę (fałszywe?) -> Short? (W oryginale strategia pivotowa gra na odbicie/przełamanie, tu upraszczamy logikę zgodną z gridem)
        # W gridzie mieliśmy: Short gdy cena < R1 (odbicie w dół) lub przebicie. 
        # Zgodnie z Twoim plikiem strategy: Short jest gdy c_prev > R1 i c_curr < R1 (przebicie w dół) - SPRAWDŹMY LOGIKĘ Z PLIKU 58
        
        # LOGIKA Z PLIKU 58 (pivot_bayes_grid_search.py):
        # event_short = (c_prev > R1) and (current_b < R1) -> Przebicie w dół (Breakout Short? A może Reversal?)
        # W Twoim kodzie było: event_short = (c_prev > R1) and (c_curr < R1)
        
        # Odtwarzam 1:1 logikę z gridu:
        # PP, R1, S1 liczone z (i-1)
        # current_b = c_curr (cena z teraz)
        # c_prev (cena z poprzedniej świecy - ale w pętli 'i' to jest close[i-1])
        
        # W pliku 58: c_prev to close[i-1]. current_b to close[i].
        # event_short = (c_prev > R1) and (c_curr < R1)
        # event_long  = (c_prev < S1) and (c_curr > S1)
        
        is_short_sig = (c_prev > r1) and (c_curr < r1)
        is_long_sig  = (c_prev < s1) and (c_curr > s1)
        
        # Bayes Probability
        p_r1 = np.mean(history_r1) if len(history_r1) >= min_events else 0.5
        p_s1 = np.mean(history_s1) if len(history_s1) >= min_events else 0.5
        
        # Decyzja
        trade = None
        
        # RAW (bayes_threshold == 0.0) -> Wchodzimy zawsze
        # BAYES -> Wchodzimy gdy prob >= threshold
        
        if is_short_sig:
            if bayes_threshold == 0.0 or p_r1 >= bayes_threshold: # Shorty miały BAYES_THRESHOLD_SHORT=0.60 stałe, ale tu testujemy "Wpływ filtru".
                # UWAGA: W Gridzie Shorty miały stały próg 0.60, a Longi zmienny.
                # Żeby porównanie było uczciwe "RAW vs BAYES", w RAW wyłączamy oba (0.0).
                # W wariancie BAYES włączamy oba? Czy tylko Long?
                # Zgodnie z Gridem: Shorty zawsze były filtrowane (0.60) albo nie?
                # W pliku 58: "if (event_short and p_r1 > 0.60) ... elif (event_long and p_s1 > bayes_threshold)"
                # Czyli Shorty zawsze miały filtr 0.60?
                # Nie, w RAW (benchmarku) filtr powinien być wyłączony.
                
                # Uprośćmy na potrzeby walidacji:
                # RAW = p_r1 > 0.0 (Bierzemy wszystko)
                # BAYES = p_r1 > 0.50 (Bierzemy tylko pewne)
                 trade = 'SHORT'

        elif is_long_sig:
             if bayes_threshold == 0.0 or p_s1 >= bayes_threshold:
                 trade = 'LONG'

        # Obliczenie wyniku (uproszczone jak w gridzie)
        current_sl = c_prev * sl_pct
        
        # Wynik dla Shorta
        raw_pnl_short = c_prev - c_curr
        is_sl_short = (h_curr - c_prev) > current_sl
        real_pnl_short = -current_sl - spread if is_sl_short else raw_pnl_short - spread
        
        # Wynik dla Longa
        raw_pnl_long = c_curr - c_prev
        is_sl_long = (c_prev - l_curr) > current_sl
        real_pnl_long = -current_sl - spread if is_sl_long else raw_pnl_long - spread
        
        # Aktualizacja historii Bayesa (czy sygnał był trafny?)
        # Trafny = real_pnl > 0
        if is_short_sig:
            history_r1.append(1 if real_pnl_short > 0 else 0)
        if is_long_sig:
            history_s1.append(1 if real_pnl_long > 0 else 0)
            
        # Realizacja transakcji
        if trade == 'SHORT':
            pnl = real_pnl_short
            equity += pnl
            trades += 1
            if pnl > 0: wins += 1
        elif trade == 'LONG':
            pnl = real_pnl_long
            equity += pnl
            trades += 1
            if pnl > 0: wins += 1
            
        # Max DD
        if equity > max_equity:
            max_equity = equity
        dd = max_equity - equity
        if dd > max_drawdown:
            max_drawdown = dd

    return equity, max_drawdown, trades, (wins/trades*100 if trades>0 else 0)

# ---------------------------------------------------------
# START
# ---------------------------------------------------------
print(f"{'SYMBOL':<10} | {'MODE':<6} | {'PROFIT':>10} | {'DD':>10} | {'TRADES':>6}")
print("-" * 55)

results = []

for symbol in SYMBOLS:
    # 1. Pobierz dane
    sy, sm = int(START_DATE[0:4]), int(START_DATE[5:7])
    ey, em = int(END_DATE[0:4]), int(END_DATE[5:7])
    df = load_bars(symbol, "1h", sy, sm, ey, em)
    df = df.loc[START_DATE:END_DATE]
    
    if df.empty:
        print(f"Brak danych dla {symbol}")
        continue
        
    # 2. Test RAW (Bayes=0.0)
    p_raw, dd_raw, t_raw, w_raw = run_strategy(df, symbol, BEST_SL, 0.0)
    
    # 3. Test BAYES (Bayes=0.50)
    p_bay, dd_bay, t_bay, w_bay = run_strategy(df, symbol, BEST_SL, BEST_BAYES_THRESH)
    
    print(f"{symbol:<10} | RAW    | {p_raw:10.2f} | {dd_raw:10.2f} | {t_raw:6}")
    print(f"{symbol:<10} | BAYES  | {p_bay:10.2f} | {dd_bay:10.2f} | {t_bay:6}")
    print("-" * 55)
