# 🔍 Hyperparameter Optimization with Optuna

This document explains how to use the automatic hyperparameter tuning feature added to `train_video_improved.py`.

## 📦 Installation

First, install the required dependencies:

```bash
python install_optuna.py
```

Or manually install:
```bash
pip install optuna>=3.0.0 matplotlib>=3.5.0 plotly>=5.0.0 kaleido>=0.2.1
```

## 🚀 Usage

### 1. Run Hyperparameter Optimization

```bash
# Basic optimization (50 trials, no timeout)
python train_video_improved.py optimize

# Custom number of trials
python train_video_improved.py optimize 30

# Custom trials and timeout (in seconds)
python train_video_improved.py optimize 50 3600  # 50 trials, 1 hour timeout
```

### 2. Regular Training (Original Mode)

```bash
# Regular training with default parameters
python train_video_improved.py
```

## 🎯 Optimized Hyperparameters

The optimization will tune the following hyperparameters:

### **Model Architecture:**
- `hidden_size`: [128, 256, 384] - LSTM hidden size
- `num_layers`: [1, 2, 3] - Number of LSTM layers
- `video_embed_dim`: [256, 384, 512] - Video encoder embedding dimension
- `video_num_heads`: [4, 6, 8] - Video transformer attention heads
- `video_num_layers`: [2, 3, 4, 5, 6] - Video transformer layers

### **Training Parameters:**
- `batch_size`: [4, 6, 8] - Training batch size
- `gradient_accumulation_steps`: [1, 2, 4] - Gradient accumulation
- `learning_rate`: [1e-5, 5e-5] (log scale) - Learning rate
- `dropout_rate`: [0.1, 0.5] - Dropout rate
- `weight_decay`: [0.001, 0.1] (log scale) - Weight decay
- `max_grad_norm`: [0.5, 2.0] - Gradient clipping

### **BERT Configuration:**
- `freeze_bert_layers`: [4, 5, 6, 7, 8] - Number of frozen BERT layers
- `bert_lr_ratio`: [0.05, 0.2] - BERT learning rate ratio

### **Loss Function:**
- `use_focal_loss`: [True, False] - Use focal loss for class imbalance
- `focal_alpha`: [0.5, 2.0] - Focal loss alpha parameter
- `focal_gamma`: [1.0, 3.0] - Focal loss gamma parameter

### **Data Processing:**
- `max_dialogue_length`: [8, 9, 10, 11, 12] - Maximum dialogue sequences
- `data_augmentation`: [True, False] - Enable data augmentation
- `warmup_ratio`: [0.05, 0.15] - Learning rate warmup ratio

## 📊 Results

After optimization completes, you'll find:

### **📁 optuna_results/ directory:**
- `study_YYYYMMDD_HHMMSS.pkl` - Complete Optuna study object
- `best_params_YYYYMMDD_HHMMSS.json` - Best hyperparameters in JSON format
- `optimization_history.png` - Optimization progress plot
- `param_importances.png` - Parameter importance analysis
- `parallel_coordinate.png` - Hyperparameter relationship visualization

### **💻 Console Output:**
```
🎉 HYPERPARAMETER OPTIMIZATION COMPLETED!
================================================================================
🏆 Best trial number: 23
🎯 Best validation accuracy: 0.8756 (87.56%)

📊 Best hyperparameters:
   batch_size: 6
   dropout_rate: 0.2854
   hidden_size: 256
   learning_rate: 0.00003247
   ...
```

## 🎯 Optimization Strategy

### **Sampler:**
- **TPE (Tree-structured Parzen Estimator)**: Intelligent hyperparameter sampling
- **Seed**: 42 for reproducible results

### **Pruner:**
- **MedianPruner**: Stops unpromising trials early
- **Startup trials**: 5 trials before pruning starts
- **Warmup steps**: 3 epochs before evaluation
- **Interval**: Check every epoch

### **Objective:**
- **Metric**: Validation accuracy maximization
- **Early stopping**: 4 epochs patience for faster trials
- **Memory management**: Automatic cleanup between trials

## 💡 Tips for Better Optimization

### **1. Start Small:**
```bash
# Quick test with fewer trials
python train_video_improved.py optimize 10 1800  # 10 trials, 30 min
```

### **2. Resource Management:**
- Monitor GPU memory usage
- Use smaller batch sizes if running out of memory
- Consider reducing `max_dialogue_length` for faster trials

### **3. Time Management:**
```bash
# Long optimization session
python train_video_improved.py optimize 100 7200  # 100 trials, 2 hours
```

### **4. Resume Optimization:**
If optimization is interrupted, you can load the study and continue:
```python
import optuna
study = optuna.load_study(study_name="your_study", storage="sqlite:///optuna.db")
study.optimize(objective, n_trials=50)
```

## 🔧 Customization

To modify the search space, edit the `objective()` function in `train_video_improved.py`:

```python
def objective(trial):
    config = {
        # Add or modify hyperparameter suggestions
        'new_param': trial.suggest_float('new_param', 0.1, 1.0),
        'hidden_size': trial.suggest_categorical('hidden_size', [256, 512, 768]),
        # ...
    }
```

## 📈 Performance Expectations

### **Typical Results:**
- **Baseline model**: ~31.89% accuracy
- **After optimization**: 35-45% accuracy (expected improvement)
- **Optimization time**: 2-8 hours depending on trials and hardware

### **GPU Requirements:**
- **Minimum**: RTX 3060 (12GB VRAM)
- **Recommended**: RTX 4070 or better (16GB+ VRAM)
- **Memory usage**: ~10-12GB per trial

## 🐛 Troubleshooting

### **Common Issues:**

1. **CUDA out of memory**:
   ```bash
   # Reduce batch size in search space
   'batch_size': trial.suggest_categorical('batch_size', [2, 4])
   ```

2. **Trials being pruned too early**:
   ```python
   # Increase warmup steps in pruner
   pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
   ```

3. **Installation issues**:
   ```bash
   # Try installing individually
   pip install optuna
   pip install matplotlib plotly kaleido
   ```

## 🎯 Next Steps

After finding optimal hyperparameters:

1. **Train full model** with optimized parameters
2. **Test performance** on test set
3. **Compare results** with baseline
4. **Fine-tune further** if needed

Example workflow:
```bash
# 1. Optimize hyperparameters
python train_video_improved.py optimize 50

# 2. Train with best parameters (when prompted)
# Choose 'y' to automatically train with best params

# 3. Test the optimized model
python test_2.py
```

## 📚 References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [TPE Sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [MedianPruner](https://optuna.readthedocs.io/en/stable/reference/pruners/generated/optuna.pruners.MedianPruner.html)
