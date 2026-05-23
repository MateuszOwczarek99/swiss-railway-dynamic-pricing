# Swiss Railway Dynamic Pricing

This is project that is an end-to-end MLOps pipeline for dynamic pricing system using reinforcement learning Deep Q-learning for Swiss railway tickets.

## problem

Railways lose revenue with static pricing. We built an RL agent that learns optimal prices based on demand, time to departure, and competitor prices.

## project structure

```text
├── src/
│   ├── data/          # collection + cleaning
│   ├── models/        # PPO agent + demand model
│   └── api/           # FastAPI server
├── tests/             # unit tests
├── docker/            # container config
├── k8s/               # kubernetes configs
└── notebooks/         # analysis
```

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




## performance results summary

| metric | value |
|--------|-------|
| demand prediction mae | 0.087 |
| demand prediction r2 | 0.912 |
| revenue improvement over baseline | 19.95% |
| ppo training episodes | 300 |
| average reward per episode | 124.5 |
| api response time p95 | 87ms |
| model inference time | 12ms |

