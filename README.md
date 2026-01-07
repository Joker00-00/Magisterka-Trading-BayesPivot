# Magisterka – BayesPivot (Pivot Points + filtr Bayes)

Repozytorium zawiera kod do pobierania i ładowania danych OHLCV oraz testowania strategii **Pivot Points** z opcjonalnym filtrem **Bayesowskim**. Dodatkowo dostępne są skrypty do **grid search / optymalizacji** parametrów.

## Struktura projektu

```
src/       # rdzeń: downloader/loader/strategie
scripts/   # uruchamialne grid search / optymalizacje
data/      # dane rynkowe (lokalnie, gitignored)
trades/    # wyniki/logi (lokalnie, gitignored)
```

> `data/`, `trades/` oraz `__pycache__/` są ignorowane przez `.gitignore`.

## Wymagania

* Python **3.11+** (testowane również na 3.13)
* Pakiety: `numpy`, `pandas`, `matplotlib`, `requests` , `seaborn`)


## Uruchamianie

### Main run (strategia v2)

```bash
python src/pivot_bayes_strategy_v2.py
```

### Grid search / optymalizacja

```bash
python scripts/pivot_bayes_grid_search.py
python scripts/pivot_bayes_grid_search_extended.py
python scripts/pivot_bayes_grid_dual.py
python scripts/pivot_bayes_optimization_all.py
```

## Dane

Dane świec są przechowywane lokalnie w `data/raw/...` (nie są wersjonowane w repo).
Źródło danych (dataset): Binance Vision (klines).
