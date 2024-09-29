import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.preprocess(text) for text in X]

    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)

def categorize_celestial_body(name):
    name_lower = name.lower()
    if any(keyword in name_lower for keyword in ['star', 'sun', 'nova', 'supernova', 'dwarf']):
        return 'Star'
    elif any(keyword in name_lower for keyword in ['planet', 'mars', 'jupiter', 'saturn', 'venus', 'mercury', 'uranus', 'neptune']):
        return 'Planet'
    elif 'galaxy' in name_lower or 'milky way' in name_lower:
        return 'Galaxy'
    elif 'nebula' in name_lower:
        return 'Nebula'
    elif any(keyword in name_lower for keyword in ['asteroid', 'meteor', 'comet', 'kuiper', 'oort']):
        return 'Small Bodies'
    elif 'cluster' in name_lower:
        return 'Star Cluster'
    elif any(keyword in name_lower for keyword in ['black hole', 'pulsar', 'quasar']):
        return 'Extreme Objects'
    elif 'weather' in name_lower:
        return 'Mars Weather'
    else:
        return 'Other Phenomena'

def load_and_preprocess_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    
    # Extract features (X) and target variable (y)
    X = data['description']
    
    # Create target variable based on 'name' only
    # This prevents leakage from 'description' to the target variable
    y = data['name'].apply(categorize_celestial_body)
    
    print("Class distribution after categorization:")
    print(y.value_counts())
    
    return X, y, data

def extract_astronomical_features(text):
    features = []
    lower_text = text.lower()
    
    # Star-specific features
    features.append(1 if any(word in lower_text for word in ['luminous', 'brightness', 'magnitude', 'spectral type']) else 0)
    features.append(1 if 'temperature' in lower_text else 0)
    
    # Planet-specific features
    features.append(1 if any(word in lower_text for word in ['orbit', 'atmosphere', 'terrestrial', 'gas giant']) else 0)
    
    # Galaxy-specific features
    features.append(1 if any(word in lower_text for word in ['spiral', 'elliptical', 'irregular', 'redshift']) else 0)
    
    # Nebula-specific features
    features.append(1 if any(word in lower_text for word in ['dust', 'gas', 'emission', 'reflection']) else 0)
    
    # General features
    features.append(1 if 'telescope' in lower_text else 0)
    features.append(1 if 'observation' in lower_text else 0)
    features.append(1 if any(word in lower_text for word in ['fusion', 'hydrogen', 'helium', 'stellar', 'supernova']) else 0)
    features.append(1 if 'main sequence' in lower_text else 0)
    features.append(1 if any(word in lower_text for word in ['red giant', 'white dwarf', 'neutron star', 'binary']) else 0)
    
    return features

def create_custom_features(X):
    # Example of simple feature engineering
    custom_features = np.column_stack([
        [len(x) for x in X],  # Length of description
        [x.count(' ') + 1 for x in X],  # Word count
        [len(set(x.split())) for x in X],  # Unique word count
    ])
    return custom_features

class CustomFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=5000, ngram_range=(1, 3)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, analyzer='char_wb')

    def fit(self, X, y=None):
        self.tfidf.fit(X)
        return self

    def transform(self, X):
        tfidf_features = self.tfidf.transform(X).toarray()
        custom_features = np.column_stack([
            [len(x) for x in X],  # Length of description
            [x.count(' ') + 1 for x in X],  # Word count
            [len(set(x.split())) for x in X],  # Unique word count
        ])
        return np.hstack((tfidf_features, custom_features))

def create_and_train_model(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Define the imbalanced pipeline
    imb_pipeline = ImbPipeline([
        ('features', CustomFeatureExtractor()),
        ('sampler', RandomOverSampler(random_state=42)),
        ('clf', GradientBoostingClassifier(random_state=42))
    ])

    # Define parameters for randomized search
    param_dist = {
        'features__max_features': [5000, 10000, None],
        'features__ngram_range': [(1, 2), (1, 3), (1, 4)],
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [3, 5, 7, None],
        'clf__learning_rate': [0.01, 0.1, 0.3],
        'clf__subsample': [0.8, 0.9, 1.0],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4]
    }

    # Perform randomized search
    random_search = RandomizedSearchCV(imb_pipeline, param_distributions=param_dist, n_iter=50, cv=3, n_jobs=-1, verbose=1, scoring='f1_macro')
    
    try:
        random_search.fit(X_train, y_train)
        print("Best parameters:", random_search.best_params_)

        # Get the best pipeline
        best_pipeline = random_search.best_estimator_

        # Evaluate on test set
        y_pred = best_pipeline.predict(X_test)
        print("Model performance on test set:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        # Perform cross-validation on the entire dataset
        cv_scores = cross_val_score(best_pipeline, X, y_encoded, cv=5, scoring='f1_macro')
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return best_pipeline, le
    
    except Exception as e:
        print(f"An error occurred during model training: {str(e)}")
        return None, le

def save_model(pipeline, label_encoder, file_path):
    joblib.dump({
        'pipeline': pipeline,
        'label_encoder': label_encoder
    }, file_path)
    print(f"Model saved to {file_path}")
    
# def load_model(file_path):
#     loaded = joblib.load(file_path)
#     return loaded['model'], loaded['isolation_forest'], loaded['label_encoder']

if __name__ == '__main__':
    X, y, celestial_data = load_and_preprocess_data('celestial_bodies.csv')
    pipeline, label_encoder = create_and_train_model(X, y)
    save_model(pipeline, label_encoder, 'celestial_model.joblib')