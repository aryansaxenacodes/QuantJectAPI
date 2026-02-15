import time
import asyncio
import logging
import httpx
import numpy as np
import pandas as pd

from typing import Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [QuantJect] - %(levelname)s - %(message)s",
)
logger = logging.getLogger("QuantJect-Core")

INJECTIVE_REST = "https://api.exchange.injective.network"
INJECTIVE_LCD = "https://lcd.injective.network"
COINGECKO = "https://api.coingecko.com/api/v3"

CHRONOS_POOL = [
    "https://sentry.exchange.grpc-web.injective.network",
    "https://k8s.global.mainnet.chronos.grpc-web.injective.network",
]

COIN_MAP = {
    "INJ": "injective-protocol",
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}

TTL_FRESH_SECONDS = 10
TTL_SOFT_STALE_SECONDS = 45


class DataCoordinator:
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.flights: Dict[str, asyncio.Event] = {}
        self.lock = asyncio.Lock()
        self.client: httpx.AsyncClient | None = None
        self.market_map: Dict[str, str] = {}

    async def resolve_markets(self):
        try:
            r = await self.client.get(
                f"{INJECTIVE_REST}/api/exchange/v1/spot/markets"
            )
            if r.status_code != 200:
                return

            raw = r.json()
            markets = raw.get("markets", [])

            for m in markets:
                market_id = m.get("marketId")
                base_meta = m.get("baseTokenMeta", {})
                symbol = base_meta.get("symbol")

                if not symbol or not market_id:
                    continue

                symbol = symbol.upper()

                if symbol in ["INJ", "BTC", "ETH", "SOL"]:
                    self.market_map[symbol] = market_id

            logger.info(f"Resolved markets: {self.market_map}")

        except Exception as e:
            logger.info(f"Market resolve failed: {e}")

    async def get_market_data(
        self,
        ticker: str,
        days: int,
        bg_tasks: BackgroundTasks,
    ) -> Tuple[pd.Series, str]:

        key = f"{ticker}_{days}"
        now = time.time()

        if key in self.cache:
            entry = self.cache[key]
            age = now - entry["ts"]

            if age < TTL_FRESH_SECONDS:
                return entry["data"], "FRESH"

            if age < TTL_SOFT_STALE_SECONDS:
                if not entry.get("updating", False):
                    entry["updating"] = True
                    bg_tasks.add_task(
                        self._background_refresh,
                        ticker,
                        days,
                        key,
                    )
                return entry["data"], "SOFT_STALE"

        if key in self.flights:
            await self.flights[key].wait()
            if key in self.cache:
                return self.cache[key]["data"], "COALESCED"

        self.flights[key] = asyncio.Event()

        try:
            data = await self._fetch_market_history(ticker, days)

            async with self.lock:
                self.cache[key] = {
                    "data": data,
                    "ts": time.time(),
                    "updating": False,
                }

            return data, "NETWORK_FETCH"

        except Exception as e:
            logger.error(f"Fetch failed: {e}")
            raise HTTPException(502, "Injective data unavailable")

        finally:
            self.flights[key].set()
            del self.flights[key]

    async def _background_refresh(self, ticker, days, key):
        try:
            data = await self._fetch_market_history(ticker, days)
            async with self.lock:
                self.cache[key] = {
                    "data": data,
                    "ts": time.time(),
                    "updating": False,
                }
        except Exception:
            if key in self.cache:
                self.cache[key]["updating"] = False

    async def _layer_exchange_rest(self, ticker, days):
        market_id = self.market_map.get(ticker.upper())
        if not market_id:
            raise IOError("No resolved market id")

        r = await self.client.get(
            f"{INJECTIVE_REST}/api/exchange/v1/spot/markets/{market_id}/candles",
            params={"interval": "1d", "limit": days},
        )

        if r.status_code != 200:
            raise IOError("Injective REST failed")

        raw = r.json()
        candles = raw.get("candles") or raw.get("data") or []

        if not candles:
            raise IOError("Empty candles")

        closes = [float(x["close"]) for x in candles]
        return pd.Series(closes)

    async def _layer_lcd(self, ticker, days):
        market_id = self.market_map.get(ticker.upper())
        if not market_id:
            raise IOError("No market id for lcd")

        r = await self.client.get(
            f"{INJECTIVE_LCD}/injective/exchange/v1beta1/spot/markets/{market_id}/candles",
            params={"interval": "1d", "limit": days},
        )

        if r.status_code != 200:
            raise IOError("Injective LCD failed")

        raw = r.json()
        candles = raw.get("candles", [])

        if not candles:
            raise IOError("LCD empty")

        closes = [float(x["close"]) for x in candles]
        return pd.Series(closes)

    async def _layer_chronos(self, ticker, days):
        market_id = self.market_map.get(ticker.upper())
        if not market_id:
            raise IOError("No market id for chronos")

        end_time = int(time.time())
        start_time = end_time - (days * 86400)

        params = {
            "marketID": market_id,
            "resolution": "1440",
            "from": start_time,
            "to": end_time,
        }

        for node in CHRONOS_POOL:
            r = await self.client.get(
                f"{node}/api/chronos/v1/market/history",
                params=params,
            )

            if r.status_code != 200:
                continue

            raw = r.json()
            hist = raw.get("history", [])

            if not hist:
                continue

            closes = [float(x["c"]) for x in hist]
            return pd.Series(closes)

        raise IOError("Chronos failed")

    async def _layer_coingecko(self, ticker, days):
        coin = COIN_MAP.get(ticker.upper(), "injective-protocol")

        r = await self.client.get(
            f"{COINGECKO}/coins/{coin}/ohlc",
            params={"vs_currency": "usd", "days": days},
        )

        if r.status_code != 200:
            raise IOError("CoinGecko failed")

        data = r.json()

        if not data:
            raise IOError("CoinGecko empty")

        closes = [float(x[4]) for x in data]
        return pd.Series(closes)

    async def _fetch_market_history(self, ticker, days):
        try:
            return await self._layer_exchange_rest(ticker, days)
        except Exception:
            logger.info("Injective Exchange REST failed")

        try:
            return await self._layer_lcd(ticker, days)
        except Exception:
            logger.info("Injective LCD failed")

        try:
            return await self._layer_chronos(ticker, days)
        except Exception:
            logger.info("Chronos failed")

        logger.info("Fallback CoinGecko")
        return await self._layer_coingecko(ticker, days)


coordinator = DataCoordinator()


async def startup_sequence():
    logger.info("Starting QuantJect...")
    coordinator.client = httpx.AsyncClient(
        timeout=15.0,
        headers=HEADERS,
    )
    await coordinator.resolve_markets()
    try:
        await coordinator._background_refresh("INJ", 30, "INJ_30")
        logger.info("Initial cache warmed.")
    except Exception:
        logger.info("Startup warming skipped.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_sequence()
    yield
    await coordinator.client.aclose()


app = FastAPI(
    title="QuantJect Institutional API",
    version="FINAL",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RiskResponse(BaseModel):
    ticker: str
    metric: str
    value: float
    meta: Dict[str, Any]


@app.get("/")
def root():
    return {"status": "Online", "docs_url": "/docs"}


@app.get("/v1/risk/volatility", response_model=RiskResponse)
async def get_volatility(
    bg_tasks: BackgroundTasks,
    ticker: str = "INJ",
    days: int = 30,
):
    t0 = time.time()

    closes, status = await coordinator.get_market_data(
        ticker,
        days,
        bg_tasks,
    )

    if len(closes) < 2:
        raise HTTPException(502, "Insufficient market data")

    log_ret = np.diff(np.log(closes))
    vol = np.std(log_ret) * np.sqrt(365) * 100

    return {
        "ticker": ticker,
        "metric": "Annualized Volatility",
        "value": round(vol, 2),
        "meta": {
            "status": status,
            "latency_ms": int((time.time() - t0) * 1000),
            "timestamp_unix": time.time(),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
