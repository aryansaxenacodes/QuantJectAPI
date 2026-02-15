# QuantJect Institutional API

## Overview

QuantJect Institutional API is a FastAPI-based service designed to provide derived market analytics using Injective market data. The current implementation exposes volatility metrics calculated from historical price data.

This project is designed for cloud deployment and is suitable for environments such as Railway.

---

## Features

- FastAPI REST interface
- Injective Chronos market history integration
- Automatic node failover
- Background cache warming
- Request coalescing to prevent duplicated upstream requests
- Volatility calculation based on log returns
- Cloud-ready architecture

---

## API Endpoints

### Root

```

GET /

````

Response:

```json
{
  "status": "Online",
  "docs_url": "/docs"
}
````

---

### Volatility Endpoint

```
GET /v1/risk/volatility
```

Query Parameters:

| Parameter | Type    | Default |
| --------- | ------- | ------- |
| ticker    | string  | INJ     |
| days      | integer | 30      |

Response Example:

```json
{
  "ticker": "INJ",
  "metric": "Annualized Volatility",
  "value": 62.41,
  "meta": {
    "status": "FRESH",
    "latency_ms": 112,
    "timestamp_unix": 1739620000
  }
}
```

---

## Installation

### Python Version

Python 3.10 or newer is recommended.

### Install Dependencies

```bash
pip install fastapi uvicorn httpx pandas numpy pydantic
```

---

## Running Locally

```bash
uvicorn main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## Railway Deployment

### Start Command

```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Railway automatically injects the port environment variable.

---

## Architecture Notes

* Market data is fetched from Injective Chronos nodes.
* A cache layer reduces upstream requests.
* Background tasks refresh cached data periodically.
* Multiple nodes are used for resilience.

---

## Error Handling

If Injective data cannot be fetched:

```
502 Bad Gateway
```

Response:

```json
{
  "detail": "Injective data unavailable"
}
```