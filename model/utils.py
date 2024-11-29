import pandas as pd
from sklearn.metrics import accuracy_score

def calculate_accuracy(labels, preds):
    return accuracy_score(labels, preds)

def save_predictions(predictions, prediction_probs, labels, filename='predictions.csv'):
    prediction_df = pd.DataFrame({
        'predictions': predictions,
        'probabilities': prediction_probs,
        'labels': labels
    })
    prediction_df.to_csv(filename, index=False)
    print(f"Predictions saved to '{filename}'")
