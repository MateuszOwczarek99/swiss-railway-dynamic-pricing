"""
main training pipeline that coordinates everything
we train both the demand model and the ppo pricing agent
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import joblib
import os

# updated imports to match new preprocessing
from src.data.clean import run_full_pipeline, engineer_features
from src.models.demand_model import DemandPredictor
from src.models.rl_agent import PPOAgent, PricingEnvironment


def setup_directories():
    """create necessary directories if they don't exist"""
    dirs = ['data/raw', 'data/processed', 'models', 'mlruns', 'reports/figures']
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def evaluate_policy(agent, env, num_episodes=10):
    """evaluate the trained policy without exploration"""
    total_rewards = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            action, _, _ = agent.act(state, training=False)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards), np.std(total_rewards)


def compare_with_baseline(agent, env):
    """
    compare our rl pricing against a simple baseline
    baseline: static pricing (always use multiplier 1.0)
    """
    baseline_rewards = []
    rl_rewards = []
    
    for _ in range(5):
        # baseline: always use multiplier 1.0 (index 5 in action space)
        env.reset()
        baseline_total = 0
        while True:
            action_idx = 5  # multiplier 1.0
            next_state, reward, done = env.step(action_idx)
            baseline_total += reward
            if done:
                break
        baseline_rewards.append(baseline_total)
        
        # rl agent
        env.reset()
        rl_total = 0
        while True:
            action, _, _ = agent.act(env.state, training=False)
            next_state, reward, done = env.step(action)
            rl_total += reward
            if done:
                break
        rl_rewards.append(rl_total)
    
    baseline_avg = np.mean(baseline_rewards)
    rl_avg = np.mean(rl_rewards)
    improvement = ((rl_avg - baseline_avg) / baseline_avg) * 100
    
    return {
        'baseline_reward': baseline_avg,
        'rl_reward': rl_avg,
        'improvement_percent': improvement
    }


def main():
    print("=" * 60)
    print("swiss railway dynamic pricing - training pipeline")
    print("=" * 60)
    
    setup_directories()
    
    # step 1: run complete preprocessing pipeline
    # this loads data, cleans, handles missing values, removes outliers,
    # normalizes, encodes, and does feature engineering
    print("\n[step 1] running data preprocessing pipeline...")
    X, y, df_features = run_full_pipeline()
    
    # step 2: split data for training
    print("\n[step 2] splitting data for training...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"training set: {X_train.shape[0]} samples")
    print(f"validation set: {X_val.shape[0]} samples")
    
    # step 3: train demand prediction model
    print("\n[step 3] training demand prediction model (xgboost)...")
    demand_model = DemandPredictor()
    demand_metrics = demand_model.train(X_train, y_train, use_mlflow=True, hyperparameter_tune=False)
    demand_model.save('models/demand_model.pkl')
    
    # step 4: create environment for rl training using processed data
    print("\n[step 4] setting up pricing environment...")
    env = PricingEnvironment(df_features, demand_model)
    
    # step 5: train ppo agent
    print("\n[step 5] training ppo pricing agent...")
    
    with mlflow.start_run(run_name="ppo_pricing_agent"):
        agent = PPOAgent(state_size=env.state_size, action_size=env.action_size)
        
        # train the agent
        rewards = agent.train(env, episodes=300)
        
        # log training metrics
        mlflow.log_metrics({
            'final_average_reward': np.mean(rewards[-50:]),
            'best_reward': max(rewards),
            'final_epsilon': agent.epsilon
        })
    
    # save trained agent
    agent.save('models/ppo_agent.pth')
    
    # step 6: evaluate performance
    print("\n[step 6] evaluating model performance...")
    
    # create fresh environment for evaluation
    eval_env = PricingEnvironment(df_features, demand_model)
    
    # compare with baseline
    comparison = compare_with_baseline(agent, eval_env)
    
    print("\n" + "=" * 60)
    print("evaluation results")
    print("=" * 60)
    print(f"demand prediction model:")
    print(f"  validation mae: {demand_metrics['val_mae']:.4f}")
    print(f"  validation r2: {demand_metrics['val_r2']:.4f}")
    print(f"\npricing policy comparison:")
    print(f"  baseline (static price): {comparison['baseline_reward']:.2f} revenue")
    print(f"  ppo dynamic pricing: {comparison['rl_reward']:.2f} revenue")
    print(f"  improvement: {comparison['improvement_percent']:.1f}%")
    print("=" * 60)
    
    # log final evaluation to mlflow
    with mlflow.start_run(run_name="final_evaluation"):
        mlflow.log_metrics({
            'baseline_revenue': comparison['baseline_reward'],
            'ppo_revenue': comparison['rl_reward'],
            'revenue_improvement_percent': comparison['improvement_percent'],
            'demand_model_mae': demand_metrics['val_mae'],
            'demand_model_r2': demand_metrics['val_r2']
        })
    
    print("\ntraining pipeline complete!")
    print(f"models saved to: models/demand_model.pkl and models/ppo_agent.pth")
    print(f"preprocessed data saved to: data/processed/")
    print(f"experiments logged to: mlruns/")
    print(f"eda plots saved to: reports/figures/")


if __name__ == "__main__":
    main()
