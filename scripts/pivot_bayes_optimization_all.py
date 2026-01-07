from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
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
# KONFIGURACJA OPTYMALIZACJI (4D)
# ==========================================
SYMBOL = "BTCUSDT"
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"
SPREAD = 2.0
WARMUP = 200 # Sztywno, bo wiemy że działa

# --- SIATKA PARAMETRÓW ---
# 1. SL (6 wariantów)
SL_RANGE = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030] 

# 2. Buffer (4 warianty) - od ciasnego do szerokiego
BUFFER_RANGE = [0.0010, 0.0020, 0.0030, 0.0040] # 0.1% - 0.4%

# 3. Bayes Long (6 wariantów)
BAYES_LONG_RANGE = [0.48, 0.50, 0.52, 0.54, 0.56, 0.58]

# 4. Bayes Short (3 warianty) - sprawdzamy czy 0.51 to na pewno optimum
BAYES_SHORT_RANGE = [0.50, 0.51, 0.52]

# ==========================================
# SILNIK (Ultra Fast)
# ==========================================
def run_simulation(df, sl, buf, b_long, b_short):
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)
    
    current_equity = 0.0
    history_s1 = deque(maxlen=200)
    history_r1 = deque(maxlen=200)
    
    for i in range(1, n):
        c_prev = closes[i-1]
        
        # Szybkie obliczanie tylko potrzebnych rzeczy
        PP = (highs[i-1] + lows[i-1] + c_prev) / 3.0
        
        # Bayes Prob
        p_r1 = np.mean(history_r1) if len(history_r1) >= WARMUP else 0.5
        p_s1 = np.mean(history_s1) if len(history_s1) >= WARMUP else 0.5
        
        # Sygnały
        curr_b = c_prev * buf
        event_short = c_prev > ((2*PP - lows[i-1]) - curr_b)
        event_long = c_prev < ((2*PP - highs[i-1]) + curr_b)
        
        if event_short and event_long: event_short, event_long = False, False

        trade_dir = None
        if event_short and p_r1 > b_short: trade_dir = 'SHORT'
        elif event_long and p_s1 > b_long: trade_dir = 'LONG'

        # Exec
        curr_sl = c_prev * sl
        
        # Short
        r_pnl_s = (-curr_sl - SPREAD) if (highs[i] - c_prev) > curr_sl else (c_prev - closes[i] - SPREAD)
        # Long
        r_pnl_l = (-curr_sl - SPREAD) if (c_prev - lows[i]) > curr_sl else (closes[i] - c_prev - SPREAD)
        
        if event_short: history_r1.append(1 if r_pnl_s > 0 else 0)
        if event_long: history_s1.append(1 if r_pnl_l > 0 else 0)
            
        if trade_dir == 'SHORT': current_equity += r_pnl_s
        elif trade_dir == 'LONG': current_equity += r_pnl_l
            
    return current_equity

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("Ładowanie danych...")
    sy, sm = int(START_DATE[:4]), int(START_DATE[5:7])
    ey, em = int(END_DATE[:4]), int(END_DATE[5:7])
    df = load_bars(SYMBOL, "15m", sy, sm, ey, em).loc[START_DATE:END_DATE]
    
    # Lista wyników
    results = []
    
    total_iter = len(SL_RANGE)*len(BUFFER_RANGE)*len(BAYES_LONG_RANGE)*len(BAYES_SHORT_RANGE)
    print(f"Start Optymalizacji 4D ({total_iter} kombinacji)...")
    
    count = 0
    for sl in SL_RANGE:
        for buf in BUFFER_RANGE:
            for bl in BAYES_LONG_RANGE:
                for bs in BAYES_SHORT_RANGE:
                    pnl = run_simulation(df, sl, buf, bl, bs)
                    results.append({
                        'SL %': sl*100,
                        'Buffer %': buf*100,
                        'Bayes L': bl,
                        'Bayes S': bs,
                        'Profit': pnl
                    })
                    count += 1
                    if count % 20 == 0: print(f"{count}/{total_iter}...", end='\r')

    # Konwersja do DataFrame
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values('Profit', ascending=False)
    
    # Zapis CSV
    res_df.to_csv('optimization_results.csv', index=False)
    print("\n\nTOP 5 WYNIKÓW:")
    print(res_df.head(5))
    
    # --- RYSOWANIE PARALLEL COORDINATES ---
    print("Generowanie wykresu Parallel Coordinates...")
    plt.figure(figsize=(15, 8))
    
    # Kolorowanie: Dzielimy wyniki na grupy (kwartyle) dla czytelności
    # Ale Parallel Coordinates w Pandas wymaga jednej kolumny "klasy" do koloru
    # Zrobimy prosty trik: Przypiszemy 'Profit Class'
    
    def classify_profit(p):
        if p > 30000: return 'Super High (>30k)'
        if p > 15000: return 'High (15k-30k)'
        if p > 0: return 'Positive (0-15k)'
        return 'Negative'

    plot_df = res_df.copy()
    plot_df['Profit Class'] = plot_df['Profit'].apply(classify_profit)
    
    # Sortujemy, żeby zielone linie (najlepsze) były na wierzchu
    plot_df = plot_df.sort_values('Profit', ascending=True)
    
    # Kolory: Czerwony, Żółty, Niebieski, Zielony
    colors = ['red', 'orange', 'blue', 'green']
    
    parallel_coordinates(plot_df[['SL %', 'Buffer %', 'Bayes L', 'Bayes S', 'Profit', 'Profit Class']], 
                         'Profit Class', 
                         color=colors, 
                         alpha=0.4, 
                         linewidth=2)
    
    plt.title("Wielowymiarowa Analiza Parametrów Strategii (Parallel Coordinates)")
    plt.ylabel("Wartość Parametru (Znormalizowana na osiach)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', title="Profit Group")
    plt.tight_layout()
    plt.savefig("parallel_coordinates.png")
    plt.show()
    
    print("Gotowe! Zapisano: parallel_coordinates.png oraz optimization_results.csv")
