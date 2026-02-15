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


NODE_POOL = [
    "https://sentry.exchange.grpc-web.injective.network",
    "https://k8s.global.mainnet.chronos.grpc-web.injective.network",
]

MARKET_MAP = {
    "INJ": "0xa508cb32923323679f29a032c70342c147c17d0145625922b0ef84e951c8440a",
    "BTC": "0x4ca0f92fc28be0c9761326016b5a1a2177dd6375558365116b5bdda9abc229ce",
    "ETH": "0x90e66cb9159ac39dc3692d43e2621db383d47f9a888c7a6e76860d7031201550",
    "SOL": "0xd32398d57529452b4755a7114e9f5ee667954b60e6118db0d33e506691c9533f",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}

TTL_FRESH_SECONDS = 10
TTL_SOFT_STALE_SECONDS = 45

SAFE_MAX_TS = 1740000000


def safe_now() -> int:
    return min(int(time.time()), SAFE_MAX_TS)


class DataCoordinator:
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.flights: Dict[str, asyncio.Event] = {}
        self.lock = asyncio.Lock()
        self.client: httpx.AsyncClient | None = None

    async def get_market_data(
        self,
        ticker: str,
        days: int,
        bg_tasks: BackgroundTasks,
    ) -> Tuple[pd.Series, str]:

        key = f"{ticker}_{days}"
        now = safe_now()

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
                    "ts": safe_now(),
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
                    "ts": safe_now(),
                    "updating": False,
                }
            logger.info(f"Background refresh success: {ticker}")
        except Exception:
            if key in self.cache:
                self.cache[key]["updating"] = False

    async def _fetch_market_history(self, ticker, days) -> pd.Series:
        market_id = MARKET_MAP.get(ticker.upper(), MARKET_MAP["INJ"])

        end_time = safe_now() - 86400
        start_time = end_time - (days * 86400)

        params = {
            "marketID": market_id,
            "resolution": "1440",
            "from": start_time,
            "to": end_time,
        }

        last_error = None

        for node in NODE_POOL:
            try:
                url = f"{node}/api/chronos/v1/market/history"

                r = await self.client.get(url, params=params)

                if r.status_code != 200:
                    last_error = f"{node} -> {r.status_code}"
                    continue

                raw = r.json()

                if isinstance(raw, dict):
                    hist = raw.get("history", [])
                elif isinstance(raw, list) and raw:
                    hist = raw[0].get("history", [])
                else:
                    hist = []

                if not hist:
                    last_error = f"{node} -> empty history"
                    continue

                closes = [float(x["c"]) for x in hist]
                return pd.Series(closes)

            except Exception as e:
                last_error = str(e)

        raise IOError(last_error or "All nodes failed")


coordinator = DataCoordinator()


async def startup_sequence():
    logger.info("Starting QuantJect...")

    coordinator.client = httpx.AsyncClient(
        timeout=12,
        headers=HEADERS,
    )

    try:
        await coordinator._background_refresh("INJ", 30, "INJ_30")
        logger.info("Initial cache warmed.")
    except Exception:
        logger.info("Startup warm skipped.")


async def warming_loop():
    while True:
        try:
            await asyncio.gather(
                coordinator._background_refresh("INJ", 30, "INJ_30"),
                coordinator._background_refresh("BTC", 30, "BTC_30"),
            )
        except Exception:
            pass
        await asyncio.sleep(15)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_sequence()
    task = asyncio.create_task(warming_loop())
    yield
    task.cancel()
    await coordinator.client.aclose()


app = FastAPI(
    title="QuantJect Institutional API",
    version="8.0.0",
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
            "timestamp_unix": safe_now(),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
