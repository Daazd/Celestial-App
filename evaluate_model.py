import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def custom_tokenizer(text):
    return text.split()

def load_data_and_model(data_path, model_path):
    data = pd.read_csv(data_path)
    model_dict = joblib.load(model_path)
    return data, model_dict['pipeline'], model_dict['label_encoder']

def map_label_to_category(label):
    label = label.lower()
    if any(word in label for word in ['galaxy', 'universe', 'andromeda']):
        return 'Galaxy'
    elif any(word in label for word in ['nebula', 'stellar nursery']):
        return 'Nebula'
    elif any(word in label for word in ['planet', 'earth', 'moon', 'mars', 'jupiter', 'saturn', 'venus', 'mercury', 'uranus', 'neptune', 'pluto']):
        return 'Planet'
    elif any(word in label for word in ['star', 'sun']):
        return 'Star'
    elif any(word in label for word in ['cluster']):
        return 'Star Cluster'
    elif any(word in label for word in ['comet', 'asteroid', 'meteor']):
        return 'Small Bodies'
    elif any(word in label for word in ['black hole', 'neutron star', 'supernova']):
        return 'Extreme Objects'
    else:
        return 'Other Phenomena'

def evaluate_model(X, y, model, label_encoder):
    y_mapped = y.apply(map_label_to_category)
    print("Label distribution after mapping:")
    print(y_mapped.value_counts())
    y_true_encoded = label_encoder.transform(y_mapped)
    try:
        y_pred = model.predict(X)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)

        print("\nPerformance Evaluation:")
        print(classification_report(y_mapped, y_pred_decoded, zero_division=0))
        print(f"Balanced Accuracy: {balanced_accuracy_score(y_true_encoded, y_pred):.4f}")

        cm = confusion_matrix(y_mapped, y_pred_decoded)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

    except Exception as e:
        print(f"An error occurred during prediction: {str(e)}")
        print("This might be due to a mismatch in the feature extraction process.")
        print("Consider retraining the model on the current dataset.")

if __name__ == "__main__":
    data, model, label_encoder = load_data_and_model('celestial_bodies.csv', 'celestial_model.joblib')
    X = data['description']
    y = data['name']

    evaluate_model(X, y, model, label_encoder)

    print("Evaluation complete. Check 'confusion_matrix.png' for visualization.")