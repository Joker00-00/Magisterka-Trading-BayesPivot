import os
import glob
import pandas as pd

try:
    from data_downloader import download_range
except ModuleNotFoundError:
    from .data_downloader import download_range


def load_bars(symbol: str, interval: str, start_year: int, start_month: int, end_year: int,
              end_month: int) -> pd.DataFrame:
    download_range(
        symbol,
        interval,
        start_year,
        start_month,
        end_year,
        end_month,
    )

    folder = os.path.join("data", "raw", symbol, interval)
    pattern = os.path.join(folder, f"{symbol}-{interval}-*.csv")

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Brak plików w: {folder}")

    dfs = []
    for file in files:
        filename = os.path.basename(file)
        _, _, year, month = filename.replace(".csv", "").split("-")

        year = int(year)
        month = int(month)

        if (year, month) < (start_year, start_month):
            continue
        if (year, month) > (end_year, end_month):
            continue

        df = pd.read_csv(file)

        # wybieramy tylko potrzebne nam kolumny
        df = df[["open_time", "open", "high", "low", "close", "volume"]]

        df.dropna(how="all")
        dfs.append(df)

    if not dfs:
        raise ValueError("Brak plików w podanym zakresie dat.")

    # łączymy w jeden DataFrame
    df = pd.concat(dfs, ignore_index=True)

    # timestamp → datetime
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

    # klucz czasowy jako index
    df.set_index("open_time", inplace=True)

    # konwersja na floaty
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    # sortowanie po czasie
    df.sort_index(inplace=True)

    return df
