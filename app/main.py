from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
import alpaca_trade_api as tradeapi

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Trading Bot API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Alpaca API
try:
    api = tradeapi.REST(
        key_id=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        base_url=os.getenv('ALPACA_BASE_URL')
    )
except Exception as e:
    print(f"Error initializing Alpaca API: {e}")

@app.get("/")
async def root():
    return {"message": "AI Trading Bot API is running"}

@app.get("/api/account")
async def get_account():
    try:
        account = api.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/positions")
async def get_positions():
    try:
        positions = api.list_positions()
        return [{
            "symbol": position.symbol,
            "qty": float(position.qty),
            "market_value": float(position.market_value),
            "unrealized_pl": float(position.unrealized_pl)
        } for position in positions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 