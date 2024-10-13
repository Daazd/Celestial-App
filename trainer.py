################################# CELESTIAL BODIES TRAINING ONLY #################################
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import classification_report, accuracy_score, make_scorer, balanced_accuracy_score
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.base import BaseEstimator, TransformerMixin
# from imblearn.combine import SMOTEENN
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.pipeline import Pipeline as ImbPipeline
# import joblib
# import re
# from scipy.stats import randint, uniform
# import nltk
# from textblob import TextBlob
# import logging


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)

# def preprocess_text(text):
#     text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
#     tokens = nltk.word_tokenize(text)
#     stop_words = set(nltk.corpus.stopwords.words('english'))
#     tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
#     return ' '.join(tokens)

# def extract_features(text):
#     blob = TextBlob(text)
#     return {
#         'word_count': len(text.split()),
#         'sentiment_polarity': blob.sentiment.polarity,
#         'sentiment_subjectivity': blob.sentiment.subjectivity,
#         'noun_phrase_count': len(list(blob.noun_phrases)),
#         'adjective_count': len([word for word, pos in blob.tags if pos.startswith('JJ')]),
#         'verb_count': len([word for word, pos in blob.tags if pos.startswith('VB')])
#     }


# class FeatureCombiner(BaseEstimator, TransformerMixin):
#     def __init__(self, text_vectorizer, numeric_scaler):
#         self.text_vectorizer = text_vectorizer
#         self.numeric_scaler = numeric_scaler

#     def fit(self, X, y=None):
#         self.text_vectorizer.fit(X.iloc[:, 0])
#         self.numeric_scaler.fit(X.iloc[:, 1:])
#         return self

#     def transform(self, X):
#         X_text_vectorized = self.text_vectorizer.transform(X.iloc[:, 0])
#         X_numeric_scaled = self.numeric_scaler.transform(X.iloc[:, 1:])
#         return np.hstack((X_text_vectorized.toarray(), X_numeric_scaled))
    
    
# def categorize_celestial_body(name):
#     name_lower = name.lower()
#     if any(keyword in name_lower for keyword in ['star', 'sun', 'nova', 'supernova', 'dwarf']):
#         return 'Star'
#     elif any(keyword in name_lower for keyword in ['planet', 'mars', 'jupiter', 'saturn', 'venus', 'mercury', 'uranus', 'neptune']):
#         return 'Planet'
#     elif 'galaxy' in name_lower or 'milky way' in name_lower:
#         return 'Galaxy'
#     elif 'nebula' in name_lower:
#         return 'Nebula'
#     elif any(keyword in name_lower for keyword in ['asteroid', 'meteor', 'comet', 'kuiper', 'oort']):
#         return 'Small Bodies'
#     elif 'cluster' in name_lower:
#         return 'Star Cluster'
#     elif any(keyword in name_lower for keyword in ['black hole', 'pulsar', 'quasar']):
#         return 'Extreme Objects'
#     elif 'weather' in name_lower:
#         return 'Mars Weather'
#     else:
#         return 'Other Phenomena'

# def custom_tokenizer(text):
#     return text.split()

# def load_and_preprocess_data(file_path):
#     data = pd.read_csv(file_path)
#     X = data['description'].apply(preprocess_text)
#     y = data['name'].apply(categorize_celestial_body)
    
#     logging.info("Data shape: %s", data.shape)
#     logging.info("X shape: %s", X.shape)
#     logging.info("y shape: %s", y.shape)
#     logging.info("Class distribution:")
#     logging.info(y.value_counts())
    
#     return X, y, data
# # def analyze_clusters(X_features, clusters, y):
# #     pca = TruncatedSVD(n_components=2)
# #     X_pca = pca.fit_transform(X_features)

# #     plt.figure(figsize=(10, 8))
# #     plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5)
# #     plt.title('Clusters Visualization (PCA-reduced)')
# #     plt.xlabel('PCA Component 1')
# #     plt.ylabel('PCA Component 2')
# #     plt.colorbar(label='Cluster Label')
# #     plt.show()

# #     cluster_counts = np.bincount(clusters)
# #     for cluster in range(len(cluster_counts)):
# #         print(f"Cluster {cluster}: {cluster_counts[cluster]} samples")
# #         corresponding_classes = np.unique(y[clusters == cluster])
# #         print(f"Corresponding classes: {corresponding_classes}")

# def create_and_train_model(X, y, quick_mode=True):
#     le = LabelEncoder()
#     y_encoded = le.fit_transform(y)

#     # Calculate class weights
#     class_weights = dict(zip(range(len(le.classes_)), 
#                              [1 / (len(y_encoded) / (len(np.unique(y_encoded)) * sum(y_encoded == c))) for c in range(len(np.unique(y_encoded)))]))

#     pipeline = ImbPipeline([
#         ('tfidf', TfidfVectorizer(tokenizer=custom_tokenizer)),
#         ('oversampler', RandomOverSampler(random_state=42)),
#         ('clf', RandomForestClassifier(random_state=42, n_jobs=-1, class_weight=class_weights))
#     ])

#     param_dist = {
#         'tfidf__max_features': randint(3000, 10000),
#         'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
#         'clf__n_estimators': randint(100, 500),
#         'clf__max_depth': randint(10, 100),
#         'clf__min_samples_split': randint(2, 20),
#         'clf__min_samples_leaf': randint(1, 10)
#     }

#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     n_iter = 20 if quick_mode else 50
    
#     scorer = make_scorer(balanced_accuracy_score)
    
#     random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, 
#                                        n_iter=n_iter, cv=cv, n_jobs=-1, 
#                                        verbose=2, scoring=scorer)

#     try:
#         logging.info("Starting Randomized Search...")
#         random_search.fit(X, y_encoded)
#         logging.info("Best parameters: %s", random_search.best_params_)

#         # Perform cross-validation
#         cv_scores = cross_val_score(random_search.best_estimator_, X, y_encoded, cv=5, scoring=scorer)
#         logging.info("Cross-validation scores: %s", cv_scores)
#         logging.info("Mean CV score: %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std() * 2)

#         # Final evaluation on the entire dataset
#         y_pred = random_search.predict(X)
#         accuracy = accuracy_score(y_encoded, y_pred)
#         balanced_accuracy = balanced_accuracy_score(y_encoded, y_pred)
#         logging.info("Model accuracy on entire dataset: %.4f", accuracy)
#         logging.info("Balanced accuracy on entire dataset: %.4f", balanced_accuracy)
#         logging.info("Classification Report:\n%s", classification_report(y_encoded, y_pred, target_names=le.classes_))

#         return random_search.best_estimator_, le, balanced_accuracy

#     except Exception as e:
#         logging.error("An error occurred during model training: %s", str(e))
#         return None, le, 0

# def save_model(pipeline, label_encoder, file_path):
#     joblib.dump({
#         'pipeline': pipeline,
#         'label_encoder': label_encoder
#     }, file_path)
#     logging.info("Model saved to %s", file_path)

# if __name__ == '__main__':
#     X, y, celestial_data = load_and_preprocess_data('celestial_bodies.csv')
#     pipeline, label_encoder, balanced_accuracy = create_and_train_model(X, y, quick_mode=False)
    
#     if balanced_accuracy >= 0.75:
#         save_model(pipeline, label_encoder, 'celestial_model.joblib')
#         logging.info("Model with %.2f%% balanced accuracy saved successfully.", balanced_accuracy * 100)
#     else:
#         logging.info("Model balanced accuracy (%.2f%%) is below the 75%% threshold. Model not saved.")

################################# CELESTIAL BODIES TRAINING AND HYG DATA #################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, make_scorer, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import randint, uniform
import joblib
import re
import nltk
from textblob import TextBlob
import logging
from memory_profiler import profile
import os
import ast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def categorize_celestial_body(name):
    name_lower = str(name).lower()
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

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    return ' '.join(tokens)

def extract_features(row):
    features = {}
    
    # Extract weather data
    if 'description' in row and 'InSight Weather' in str(row.get('name', '')):
        try:
            weather_data = ast.literal_eval(row['description'])
            features.update({
                'avg_temp': weather_data['AT']['av'],
                'min_temp': weather_data['AT']['mn'],
                'max_temp': weather_data['AT']['mx'],
                'avg_wind_speed': weather_data['HWS']['av'],
                'avg_pressure': weather_data['PRE']['av']
            })
        except:
            pass
    
    # Extract APOD 
    features.update({
        'text_length': len(str(row.get('description', ''))),
        'keyword_count': len(str(row.get('keywords', '')).split(',')) if pd.notna(row.get('keywords')) else 0,
        'has_image': pd.notna(row.get('image_url'))
    })
    
    # Extract HYG features if available
    if 'mag' in row and 'dist' in row and 'ci' in row:
        features.update({
            'magnitude': row['mag'],
            'distance': row['dist'],
            'color_index': row['ci']
        })
    features['text'] = preprocess_text(row.get('description', '') or row.get('proper', ''))
    return pd.Series(features)

def load_and_preprocess_data(celestial_file_path, hyg_file_path, max_samples_per_class=1000):
    celestial_data = pd.read_csv(celestial_file_path)
    celestial_features = celestial_data.apply(extract_features, axis=1)
    
    hyg_data = pd.read_csv(hyg_file_path)
    hyg_features = hyg_data.apply(extract_features, axis=1)
    
    X_combined = pd.concat([celestial_features, hyg_features], ignore_index=True)
    y_combined = pd.concat([
        celestial_data['name'].apply(categorize_celestial_body),
        hyg_data['proper'].fillna('').apply(categorize_celestial_body)
    ], ignore_index=True)
    
    X_combined = X_combined.fillna(0)
    
    class_counts = y_combined.value_counts()
    sampling_strategy = {
        label: min(count, max_samples_per_class)
        for label, count in class_counts.items()
        if count > max_samples_per_class
    }
    
    if sampling_strategy:
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_combined, y_combined)
    else:
        X_resampled, y_resampled = X_combined, y_combined

    logging.info("Final data shape after balanced sampling: %s", X_resampled.shape)
    logging.info("Final class distribution after balanced sampling:")
    logging.info(y_resampled.value_counts())

    return X_resampled, y_resampled

def analyze_class_distribution(y):
    class_distribution = y.value_counts()
    logging.info("Class distribution:")
    logging.info(class_distribution)
    logging.info("Number of classes: %d", len(class_distribution))
    
    if len(class_distribution) < 3:
        logging.warning("Less than 3 classes in the dataset. This might lead to poor model performance.")
    
    imbalance_ratio = class_distribution.max() / class_distribution.min()
    logging.info("Imbalance ratio (max/min): %.2f", imbalance_ratio)
    
    return class_distribution

def create_and_train_model(X, y, quick_mode=True):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    numeric_features = [col for col in X.columns if col != 'text']
    text_feature = 'text'

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('text', TfidfVectorizer(max_features=5000), text_feature)
        ])

    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'))
    ])

    param_dist = {
        'preprocessor__text__max_features': randint(1000, 10000),
        'preprocessor__text__ngram_range': [(1, 1), (1, 2)],
        'clf__n_estimators': randint(100, 300),
        'clf__max_depth': randint(10, 100),
        'clf__min_samples_split': randint(2, 20),
        'clf__min_samples_leaf': randint(1, 10)
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    n_iter = 10 if quick_mode else 30

    random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, 
                                       n_iter=n_iter, cv=cv, n_jobs=-1, 
                                       verbose=2, scoring='balanced_accuracy')

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    try:
        logging.info("Starting Randomized Search...")
        random_search.fit(X_train, y_train)
        logging.info("Best parameters: %s", random_search.best_params_)

        y_pred = random_search.predict(X_test)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        logging.info("Balanced accuracy on test set: %.4f", balanced_accuracy)
        
        class_names = le.classes_
        logging.info("Classification Report:\n%s", classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

        return random_search.best_estimator_, le, balanced_accuracy

    except Exception as e:
        logging.error("An error occurred during model training: %s", str(e))
        return None, le, 0

def save_model(pipeline, label_encoder, file_path):
    joblib.dump({
        'pipeline': pipeline,
        'label_encoder': label_encoder
    }, file_path)
    logging.info("Model saved to %s", file_path)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        X, y = load_and_preprocess_data('celestial_bodies.csv', 'hygdata_v3.csv', max_samples_per_class=1000)
        pipeline, label_encoder, balanced_accuracy = create_and_train_model(X, y, quick_mode=False)
        
        if balanced_accuracy >= 0.75:
            joblib.dump({'pipeline': pipeline, 'label_encoder': label_encoder}, 'celestial_model.joblib')
            logging.info("Model with %.2f%% balanced accuracy saved successfully.", balanced_accuracy * 100)
        else:
            logging.info("Model balanced accuracy (%.2f%%) is below the 75%% threshold. Model not saved.", balanced_accuracy * 100)
    
    except Exception as e:
        logging.error(f"An error occurred during execution: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
