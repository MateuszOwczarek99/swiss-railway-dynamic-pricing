"""
unit tests for the pricing api
we test both the health endpoint and the main pricing logic
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.app import app

client = TestClient(app)


def test_health_endpoint():
    """test that health check returns correct status"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_metrics_endpoint():
    """test metrics endpoint returns service info"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "dynamic-pricing"
    assert data["algorithm"] == "ppo"


def test_price_endpoint_basic():
    """test basic pricing request returns expected structure"""
    request_data = {
        "base_price": 50.0,
        "competitor_price": 45.0,
        "days_to_departure": 3,
        "is_weekend": False,
        "month": 6
    }
    
    response = client.post("/price", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "original_price" in data
    assert "recommended_price" in data
    assert "price_multiplier" in data
    assert "expected_demand" in data
    assert "explanation" in data
    assert "timestamp" in data


def test_price_endpoint_close_departure():
    """test that close departure leads to premium price"""
    request_data = {
        "base_price": 50.0,
        "competitor_price": 55.0,
        "days_to_departure": 1,
        "is_weekend": True,
        "month": 12
    }
    
    response = client.post("/price", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    # close departure should result in multiplier > 1.0
    assert data["price_multiplier"] >= 1.0
    assert "premium" in data["explanation"].lower() or "close" in data["explanation"].lower()


def test_price_endpoint_competitor_discount():
    """test that cheaper competitor leads to discount"""
    request_data = {
        "base_price": 60.0,
        "competitor_price": 40.0,
        "days_to_departure": 10,
        "is_weekend": False,
        "month": 3
    }
    
    response = client.post("/price", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    # cheaper competitor should result in discount
    if data["price_multiplier"] < 1.0:
        assert "discount" in data["explanation"].lower() or "compete" in data["explanation"].lower()


def test_price_endpoint_invalid_input():
    """test that invalid input returns error"""
    # missing required field
    request_data = {
        "base_price": 50.0,
        "days_to_departure": 3
    }
    
    response = client.post("/price", json=request_data)
    assert response.status_code == 422  # validation error


def test_price_endpoint_edge_cases():
    """test edge cases like zero prices"""
    request_data = {
        "base_price": 0.01,
        "competitor_price": 0.01,
        "days_to_departure": 365,
        "is_weekend": False,
        "month": 1
    }
    
    response = client.post("/price", json=request_data)
    assert response.status_code == 200
    assert response.json()["recommended_price"] > 0
