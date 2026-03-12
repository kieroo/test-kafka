#!/usr/bin/env python3
"""BTC Dual Investment (Dual Currency) strategy simulator.

This is a risk-controlled reference implementation:
- It does NOT guarantee stable profits.
- It helps evaluate a conservative product-selection and exposure policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import urllib.request
from urllib.error import URLError
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional


@dataclass
class Candle:
    date: datetime
    close: float


@dataclass
class StrategyConfig:
    initial_usdt: float = 10_000.0
    initial_btc: float = 0.0
    max_allocation_ratio: float = 0.35
    trend_window: int = 20
    vol_window: int = 10
    vol_pause_threshold: float = 0.08
    base_apr: float = 0.18
    stress_apr: float = 0.10
    tenor_days: int = 7
    strike_buffer: float = 0.03


@dataclass
class Snapshot:
    date: datetime
    equity_usdt: float


class DualInvestmentSimulator:
    def __init__(self, candles: List[Candle], cfg: StrategyConfig):
        if len(candles) < max(cfg.trend_window, cfg.vol_window) + cfg.tenor_days + 1:
            raise ValueError("Not enough data for selected windows and tenor.")
        self.candles = candles
        self.cfg = cfg
        self.usdt = cfg.initial_usdt
        self.btc = cfg.initial_btc
        self.history: List[Snapshot] = []
        self.trade_count = 0
        self.win_count = 0

    def run(self) -> dict:
        start = max(self.cfg.trend_window, self.cfg.vol_window)
        i = start
        while i + self.cfg.tenor_days < len(self.candles):
            today = self.candles[i]
            spot = today.close
            self.record_equity(today.date, spot)

            trend = self.calc_trend(i)
            vol = self.calc_vol(i)
            if vol > self.cfg.vol_pause_threshold:
                i += 1
                continue

            allocated_value = self.usdt + self.btc * spot
            budget = allocated_value * self.cfg.max_allocation_ratio
            if budget < 50:
                i += 1
                continue

            if trend >= 0:
                did_trade = self.place_sell_high(i, budget)
            else:
                did_trade = self.place_buy_low(i, budget)

            self.trade_count += int(did_trade)
            i += 1

        # final snapshot
        self.record_equity(self.candles[-1].date, self.candles[-1].close)
        return self.metrics()

    def calc_trend(self, idx: int) -> float:
        now = self.candles[idx].close
        past = self.candles[idx - self.cfg.trend_window].close
        return (now / past) - 1.0

    def calc_vol(self, idx: int) -> float:
        rets = []
        for k in range(idx - self.cfg.vol_window + 1, idx + 1):
            p0 = self.candles[k - 1].close
            p1 = self.candles[k].close
            rets.append((p1 / p0) - 1.0)
        m = sum(rets) / len(rets)
        var = sum((r - m) ** 2 for r in rets) / len(rets)
        return math.sqrt(var)

    def est_apr(self, vol: float) -> float:
        if vol > self.cfg.vol_pause_threshold * 0.8:
            return self.cfg.stress_apr
        return self.cfg.base_apr

    def place_sell_high(self, idx: int, budget_usdt: float) -> bool:
        spot = self.candles[idx].close
        alloc_btc = min(self.btc, budget_usdt / spot)
        if alloc_btc <= 0:
            return False

        strike = spot * (1 + self.cfg.strike_buffer)
        settle_idx = idx + self.cfg.tenor_days
        settle_price = self.candles[settle_idx].close
        apr = self.est_apr(self.calc_vol(idx))
        premium_btc = alloc_btc * apr * self.cfg.tenor_days / 365

        # if price >= strike, BTC converted to USDT at strike; otherwise keep BTC
        if settle_price >= strike:
            self.btc -= alloc_btc
            self.usdt += alloc_btc * strike
            self.btc += premium_btc
            # profit relative to hold BTC for this notional
            hold_value = alloc_btc * settle_price
            strategy_value = alloc_btc * strike + premium_btc * settle_price
            if strategy_value >= hold_value:
                self.win_count += 1
        else:
            self.btc += premium_btc
            # baseline hold is alloc_btc, strategy is alloc_btc+premium
            self.win_count += 1

        return True

    def place_buy_low(self, idx: int, budget_usdt: float) -> bool:
        spot = self.candles[idx].close
        alloc_usdt = min(self.usdt, budget_usdt)
        if alloc_usdt <= 0:
            return False

        strike = spot * (1 - self.cfg.strike_buffer)
        settle_idx = idx + self.cfg.tenor_days
        settle_price = self.candles[settle_idx].close
        apr = self.est_apr(self.calc_vol(idx))
        premium_usdt = alloc_usdt * apr * self.cfg.tenor_days / 365

        # reserve funds until settlement
        self.usdt -= alloc_usdt

        if settle_price <= strike:
            got_btc = alloc_usdt / strike
            self.btc += got_btc
            self.usdt += premium_usdt
            # benchmark: keep USDT; strategy gets BTC + premium
            bench = alloc_usdt
            strat = got_btc * settle_price + premium_usdt
            if strat >= bench:
                self.win_count += 1
        else:
            self.usdt += alloc_usdt + premium_usdt
            self.win_count += 1

        return True

    def record_equity(self, date: datetime, spot: float) -> None:
        equity = self.usdt + self.btc * spot
        self.history.append(Snapshot(date, equity))

    def metrics(self) -> dict:
        if len(self.history) < 2:
            return {}
        start = self.history[0].equity_usdt
        end = self.history[-1].equity_usdt
        days = (self.history[-1].date - self.history[0].date).days or 1
        annualized = (end / start) ** (365 / days) - 1

        peak = self.history[0].equity_usdt
        max_dd = 0.0
        for p in self.history:
            peak = max(peak, p.equity_usdt)
            dd = (peak - p.equity_usdt) / peak
            max_dd = max(max_dd, dd)

        return {
            "start_equity": round(start, 2),
            "end_equity": round(end, 2),
            "total_return": round((end / start - 1) * 100, 2),
            "annualized_return": round(annualized * 100, 2),
            "max_drawdown": round(max_dd * 100, 2),
            "trades": self.trade_count,
            "win_rate": round((self.win_count / self.trade_count) * 100, 2)
            if self.trade_count
            else 0.0,
        }


def load_csv(path: str) -> List[Candle]:
    out: List[Candle] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"date", "close"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV must contain columns: date, close")
        for row in reader:
            out.append(
                Candle(
                    date=datetime.strptime(row["date"], "%Y-%m-%d"),
                    close=float(row["close"]),
                )
            )
    out.sort(key=lambda x: x.date)
    return out


def generate_demo(days: int = 365) -> List[Candle]:
    random.seed(42)
    price = 30_000.0
    date = datetime(2023, 1, 1)
    candles: List[Candle] = []
    for _ in range(days):
        drift = 0.0003
        shock = random.gauss(0, 0.02)
        price *= (1 + drift + shock)
        price = max(6_000, price)
        candles.append(Candle(date=date, close=price))
        date += timedelta(days=1)
    return candles


def load_coingecko(days: int = 365, vs_currency: str = "usd") -> List[Candle]:
    """Load BTC daily close data from CoinGecko public API.

    API endpoint returns [timestamp_ms, price] pairs. We keep the last price
    observed per UTC date as the daily close.
    """

    if days <= 0:
        raise ValueError("days must be a positive integer")

    url = (
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        f"?vs_currency={vs_currency}&days={days}&interval=daily"
    )
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "dual-invest-strategy/1.0",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except URLError as e:
        raise ValueError(f"Failed to fetch real data from CoinGecko: {e}") from e

    prices = payload.get("prices")
    if not prices:
        raise ValueError("No prices returned from CoinGecko API")

    daily_closes = {}
    for ts_ms, close in prices:
        date = datetime.utcfromtimestamp(ts_ms / 1000).date()
        daily_closes[date] = float(close)

    candles = [
        Candle(
            date=datetime.combine(d, datetime.min.time()),
            close=price,
        )
        for d, price in sorted(daily_closes.items())
    ]
    return candles


def load_binance(days: int = 365, symbol: str = "BTCUSDT") -> List[Candle]:
    """Load BTC daily close data from Binance Kline API."""

    if days <= 0:
        raise ValueError("days must be a positive integer")
    if days > 1000:
        raise ValueError("Binance source supports up to 1000 days per request")

    url = (
        "https://api.binance.com/api/v3/klines"
        f"?symbol={symbol}&interval=1d&limit={days}"
    )
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "dual-invest-strategy/1.0",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except URLError as e:
        raise ValueError(f"Failed to fetch real data from Binance: {e}") from e

    if not isinstance(payload, list) or not payload:
        raise ValueError("No kline data returned from Binance API")

    candles: List[Candle] = []
    for row in payload:
        open_time_ms = int(row[0])
        close = float(row[4])
        d = datetime.utcfromtimestamp(open_time_ms / 1000).date()
        candles.append(Candle(date=datetime.combine(d, datetime.min.time()), close=close))

    candles.sort(key=lambda x: x.date)
    return candles


def load_okx(days: int = 365, inst_id: str = "BTC-USDT") -> List[Candle]:
    """Load BTC daily close data from OKX Candles API."""

    if days <= 0:
        raise ValueError("days must be a positive integer")
    if days > 300:
        raise ValueError("OKX source supports up to 300 days per request")

    url = f"https://www.okx.com/api/v5/market/history-candles?instId={inst_id}&bar=1D&limit={days}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "dual-invest-strategy/1.0",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except URLError as e:
        raise ValueError(f"Failed to fetch real data from OKX: {e}") from e

    data = payload.get("data") if isinstance(payload, dict) else None
    if not data:
        raise ValueError("No candle data returned from OKX API")

    candles: List[Candle] = []
    for row in data:
        ts_ms = int(row[0])
        close = float(row[4])
        d = datetime.utcfromtimestamp(ts_ms / 1000).date()
        candles.append(Candle(date=datetime.combine(d, datetime.min.time()), close=close))

    candles.sort(key=lambda x: x.date)
    return candles


def load_real_data(days: int, source: str) -> List[Candle]:
    source = source.lower()
    loaders = {
        "coingecko": lambda: load_coingecko(days=days),
        "binance": lambda: load_binance(days=days),
        "okx": lambda: load_okx(days=days),
    }

    if source != "auto":
        if source not in loaders:
            raise ValueError("Unknown real source. Use auto/coingecko/binance/okx")
        return loaders[source]()

    errors = []
    for name in ("okx", "binance", "coingecko"):
        try:
            candles = loaders[name]()
            print(f"[real-data] source={name}, records={len(candles)}")
            return candles
        except ValueError as e:
            errors.append(f"{name}: {e}")

    raise ValueError("Failed to fetch from all real sources. " + " | ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC 双币赢稳健策略模拟器")
    parser.add_argument("--csv", help="CSV path with date,close")
    parser.add_argument("--demo", action="store_true", help="Use generated demo data")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real BTC daily closes from online APIs",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days for --real source (default: 365)",
    )
    parser.add_argument(
        "--real-source",
        default="auto",
        choices=["auto", "okx", "binance", "coingecko"],
        help="Data source for --real (default: auto, try okx->binance->coingecko)",
    )
    args = parser.parse_args()

    selected_sources = int(bool(args.csv)) + int(args.demo) + int(args.real)
    if selected_sources != 1:
        parser.error("Please choose exactly one source: --csv / --demo / --real")

    try:
        if args.csv:
            candles = load_csv(args.csv)
        elif args.real:
            candles = load_real_data(days=args.days, source=args.real_source)
        else:
            candles = generate_demo()
    except ValueError as e:
        parser.error(str(e))

    cfg = StrategyConfig()
    sim = DualInvestmentSimulator(candles, cfg)
    result = sim.run()

    print("=== 回测结果（风险约束策略）===")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
