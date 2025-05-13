from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
from ..services.dashboard_service import DashboardService

app = FastAPI()
dashboard_service = DashboardService()

class TradeData(BaseModel):
    symbol: str
    type: str  # 'BUY' or 'SELL'
    price: float
    quantity: int
    pnl: float
    status: str  # 'open' or 'closed'

class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float

class PortfolioValue(BaseModel):
    value: float
    timestamp: Optional[datetime] = None

@app.post("/api/trade")
async def add_trade(trade: TradeData):
    """Add a new trade to the dashboard"""
    try:
        dashboard_service.add_trade(trade.dict())
        return {"status": "success", "message": "Trade added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model-metrics")
async def update_model_metrics(metrics: ModelMetrics):
    """Update model performance metrics"""
    try:
        dashboard_service.update_model_metrics(metrics.dict())
        return {"status": "success", "message": "Model metrics updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/portfolio-value")
async def update_portfolio_value(portfolio: PortfolioValue):
    """Update portfolio value"""
    try:
        dashboard_service.update_portfolio_value(
            portfolio.value,
            portfolio.timestamp
        )
        return {"status": "success", "message": "Portfolio value updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard-data")
async def get_dashboard_data():
    """Get all dashboard data"""
    try:
        return dashboard_service.get_dashboard_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 