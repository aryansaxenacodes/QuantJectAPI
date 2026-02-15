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
COINGECKO = "https://api.coingecko.com/api/v3"

MARKET_MAP = {
    "INJ": "0xa508cb32923323679f29a032c70342c147c17d0145625922b0ef84e951c8440a",
    "BTC": "0x4ca0f92fc28be0c9761326016b5a1a2177dd6375558365116b5bdda9abc229ce",
    "ETH": "0x90e66cb9159ac39dc3692d43e2621db383d47f9a888c7a6e76860d7031201550",
    "SOL": "0xd32398d57529452b4755a7114e9f5ee667954b60e6118db0d33e506691c9533f",
}

COIN_MAP = {
    "INJ": "injective-protocol",
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}

HEADERS = {"User-Agent": "Mozilla/5.0"}

TTL_FRESH_SECONDS = 10
TTL_SOFT_STALE_SECONDS = 45


class DataCoordinator:
    def __init__(self):
        self.cache = {}
        self.flights = {}
        self.lock = asyncio.Lock()
        self.client = None

    async def get_market_data(self, ticker, days, bg_tasks):
        key = f"{ticker}_{days}"
        now = time.time()

        if key in self.cache:
            entry = self.cache[key]
            age = now - entry["ts"]

            if age < TTL_FRESH_SECONDS:
                return entry["data"], "FRESH"

            if age < TTL_SOFT_STALE_SECONDS:
                if not entry.get("updating"):
                    entry["updating"] = True
                    bg_tasks.add_task(self._background_refresh, ticker, days, key)
                return entry["data"], "SOFT_STALE"

        if key in self.flights:
            await self.flights[key].wait()
            return self.cache[key]["data"], "COALESCED"

        self.flights[key] = asyncio.Event()

        try:
            data = await self._fetch_market_history(ticker, days)
            async with self.lock:
                self.cache[key] = {"data": data, "ts": time.time(), "updating": False}
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
                self.cache[key] = {"data": data, "ts": time.time(), "updating": False}
        except Exception:
            if key in self.cache:
                self.cache[key]["updating"] = False

    async def _fetch_injective_rest(self, ticker, days):
        market_id = MARKET_MAP[ticker]
        url = f"{INJECTIVE_REST}/api/exchange/v1/spot/markets/{market_id}/candles"
        r = await self.client.get(url, params={"interval": "1d", "limit": days})
        if r.status_code != 200:
            raise IOError("injective rest failed")
        data = r.json().get("data", [])
        if not data:
            raise IOError("empty candles")
        closes = [float(c["close"]) for c in data]
        return pd.Series(closes)

    async def _fetch_coingecko(self, ticker, days):
        coin = COIN_MAP[ticker]
        r = await self.client.get(
            f"{COINGECKO}/coins/{coin}/ohlc",
            params={"vs_currency": "usd", "days": days},
        )
        if r.status_code != 200:
            raise IOError("coingecko failed")
        data = r.json()
        closes = [float(x[4]) for x in data]
        return pd.Series(closes)

    async def _fetch_market_history(self, ticker, days):
        ticker = ticker.upper()

        try:
            return await self._fetch_injective_rest(ticker, days)
        except Exception:
            logger.info("Injective REST failed, fallback CoinGecko")

        return await self._fetch_coingecko(ticker, days)


coordinator = DataCoordinator()


async def startup_sequence():
    logger.info("Starting QuantJect...")
    coordinator.client = httpx.AsyncClient(timeout=15.0, headers=HEADERS)
    try:
        await coordinator._background_refresh("INJ", 30, "INJ_30")
        logger.info("Initial cache warmed.")
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_sequence()
    yield
    await coordinator.client.aclose()


app = FastAPI(title="QuantJect Institutional API", version="11.0.0", lifespan=lifespan)

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
async def get_volatility(bg_tasks: BackgroundTasks, ticker="INJ", days=30):
    t0 = time.time()

    closes, status = await coordinator.get_market_data(ticker, days, bg_tasks)

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
