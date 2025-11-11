"""
Simple test script to verify Optuna setup
"""

import torch
import optuna

def simple_objective(trial):
    """Simple test objective"""
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

def main():
    print("OPTUNA TEST")
    print("=" * 40)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test Optuna
    print("Testing Optuna...")
    study = optuna.create_study(direction='minimize')
    study.optimize(simple_objective, n_trials=3)
    
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")
    print("SUCCESS: Optuna working!")

if __name__ == "__main__":
    main()