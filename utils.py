from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch

def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    return f1, precision, recall, accuracy

def save_metrics_to_file(filepath, metrics_dict):
    with open(filepath, 'w') as f:
        for k, v in metrics_dict.items():
            f.write(f"{k}: {v:.4f}\n")

def save_model(model, path):
    torch.save(model.state_dict(), path)
