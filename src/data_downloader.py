import os
from io import BytesIO
from zipfile import ZipFile

import requests

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"


def download_range(symbol: str, interval: str, start_year: int, start_month: int, end_year: int, end_month: int):
    base_dir = os.path.join("data", "raw", symbol, interval)
    os.makedirs(base_dir, exist_ok=True)

    for year, month in _iter_months(start_year, start_month, end_year, end_month):
        _download_month(symbol, interval, year, month, base_dir)


def _iter_months(start_year: int, start_month: int, end_year: int, end_month: int):
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def _download_month(symbol: str, interval: str, year: int, month: int, out_dir: str):
    fname_zip = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    url = f"{BASE_URL}/{symbol}/{interval}/{fname_zip}"
    out_csv = os.path.join(out_dir, f"{symbol}-{interval}-{year}-{month:02d}.csv")

    if os.path.exists(out_csv):
        print(f"[SKIP] {out_csv} już istnieje")
        return

    print(f"[DL] {url}")
    r = requests.get(url)
    if r.status_code != 200:
        print(f"[MISS] Brak pliku: {url} (status {r.status_code})")
        return

    with ZipFile(BytesIO(r.content)) as z:
        # w archiwum jest jeden plik CSV
        name = z.namelist()[0]
        with z.open(name) as f_in, open(out_csv, "wb") as f_out:
            f_out.write(f_in.read())

    print(f"[OK] {out_csv}")
