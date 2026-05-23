"""
fastapi server for serving real-time pricing recommendations
this is the production interface for our dynamic pricing system
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np
import joblib
import sys
import os

# add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.rl_agent import PPOAgent

# initialize fastapi app
app = FastAPI(
    title="swiss railway dynamic pricing api",
    description="real-time ticket pricing using ppo reinforcement learning",
    version="2.0.0"
)

# global model objects
demand_model = None
rl_agent = None

# action space mapping (price multipliers)
ACTION_MULTIPLIERS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]


class PriceRequest(BaseModel):
    """request schema for price endpoint"""
    base_price: float = Field(..., description="current base ticket price", example=50.0)
    competitor_price: float = Field(..., description="lowest competitor price", example=45.0)
    days_to_departure: int = Field(..., description="days until travel date", ge=0, le=365, example=3)
    is_weekend: bool = Field(default=False, description="whether travel date is weekend")
    month: int = Field(..., description="month of travel (1-12)", ge=1, le=12, example=6)


class PriceResponse(BaseModel):
    """response schema for price endpoint"""
    original_price: float
    recommended_price: float
    price_multiplier: float
    expected_demand: float
    explanation: str
    timestamp: str


@app.on_event("startup")
async def load_models():
    """load trained models when api server starts"""
    global demand_model, rl_agent
    
    print("loading trained models...")
    
    try:
        # load demand prediction model
        demand_model = joblib.load('models/demand_model.pkl')
        print("demand model loaded successfully")
        
        # load ppo agent
        rl_agent = PPOAgent(state_size=5, action_size=11)
        rl_agent.load('models/ppo_agent.pth')
        print("ppo agent loaded successfully")
        
    except Exception as e:
        print(f"warning: could not load models: {e}")
        print("api will run with fallback pricing logic")


@app.get("/health")
async def health_check():
    """kubernetes health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": demand_model is not None and rl_agent is not None
    }


@app.get("/metrics")
async def get_metrics():
    """basic metrics for monitoring"""
    return {
        "service": "dynamic-pricing",
        "version": "2.0.0",
        "algorithm": "ppo",
        "status": "operational"
    }


@app.post("/price", response_model=PriceResponse)
async def get_dynamic_price(request: PriceRequest):
    """
    get dynamically optimized price based on current conditions
    uses ppo agent to choose optimal price multiplier
    """
    
    try:
        # build state vector for the rl agent
        state = np.array([
            request.base_price / 100.0,  # normalize price
            request.base_price / (request.competitor_price + 0.01),  # price ratio
            min(request.days_to_departure, 30) / 30.0,  # normalized days
            1.0 if request.is_weekend else 0.0,  # weekend indicator
            request.month / 12.0  # normalized month
        ], dtype=np.float32)
        
        # get action from ppo agent (no exploration in production)
        if rl_agent is not None:
            action_idx, _, _ = rl_agent.act(state, training=False)
        else:
            # fallback logic if model not loaded
            if request.days_to_departure < 3:
                action_idx = 7  # 1.2x multiplier
            elif request.competitor_price < request.base_price * 0.9:
                action_idx = 4  # 0.9x multiplier
            else:
                action_idx = 5  # 1.0x multiplier
        
        # get price multiplier from action
        multiplier = ACTION_MULTIPLIERS[action_idx]
        recommended_price = request.base_price * multiplier
        
        # estimate demand at this price (simple elasticity model)
        price_elasticity = -0.4
        demand_change = price_elasticity * (multiplier - 1)
        
        # base demand estimation
        base_demand = 0.6
        if request.is_weekend:
            base_demand += 0.15
        if request.month in [7, 8, 12]:
            base_demand += 0.1
        if request.days_to_departure < 3:
            base_demand += 0.2
        elif request.days_to_departure < 7:
            base_demand += 0.1
        
        expected_demand = max(0.1, min(0.95, base_demand + demand_change))
        
        # generate human-readable explanation
        if multiplier > 1.15:
            explanation = f"premium price due to high demand and close departure ({request.days_to_departure} days away)"
        elif multiplier < 0.85:
            explanation = f"discounted price to compete with alternatives (competitor at chf {request.competitor_price:.2f})"
        else:
            explanation = "standard pricing based on current market conditions"
        
        return PriceResponse(
            original_price=round(request.base_price, 2),
            recommended_price=round(recommended_price, 2),
            price_multiplier=round(multiplier, 2),
            expected_demand=round(expected_demand, 3),
            explanation=explanation,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"pricing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
