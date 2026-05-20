# Swiss Railway Dynamic Pricing

Dynamic pricing system using reinforcement learning (DQN) for Swiss railway tickets.

## Problem

Railways lose revenue with static pricing. We built an RL agent that learns optimal prices based on demand, time to departure, and competitor prices.

## fast start

```bash
# install
make install

# collect data
make collect

# train models
make train

# run api
make run

# test with curl
curl -X POST http://localhost:8000/price \
  -H "Content-Type: application/json" \
  -d '{"origin":"zurich","destination":"geneva","travel_date":"2024-06-15T10:00:00","base_price":50,"competitor_price":45,"days_to_departure":5,"is_weekend":false}'
