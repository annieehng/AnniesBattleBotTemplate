from teams_classes import DetectionMark
from abc_classes import ADetector
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from xgboost import XGBClassifier



class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.column].values.astype("U")  # ensure unicode strings

class NumericSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.column]]

class Detector(ADetector):
    def __init__(self):
        self.ml_model = None

    def extract_features(self, session_data):
        data = []
        for user in session_data.users:
            user_id = user.get("id")
            posts = [p.get("text", "") for p in session_data.posts if p.get("author_id") == user_id]
            aggregated_text = " ".join(posts)
            data.append({
                "user_id": user_id,
                "text": aggregated_text,
                "tweet_count": user.get("tweet_count", 0),
                "z_score": user.get("z_score", 0),
                "username": user.get("username", ""),
                "name": user.get("name", ""),
                "description": user.get("description", ""),
                "location": user.get("location", ""),
                "label": user.get("label")
            })
        return pd.DataFrame(data)

    def train_ml_model(self, training_session_data):
        df = self.extract_features(training_session_data)
        df = df[df['label'].notnull()]
        df.fillna(0, inplace=True)

        X = df.drop(columns=["user_id", "label"])
        y = df["label"].astype(int)

        # Test with only TF-IDF features
        feature_union = FeatureUnion([
            ("tfidf", Pipeline([
                ("selector", ColumnSelector("text")),
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)))
            ]))
        ])

        classifier = XGBClassifier(
            n_estimators=300,  # Increase estimators
            max_depth=10,
            learning_rate=0.05,  # Reduce learning rate
            scale_pos_weight=len(y[y == 0]) / len(y[y == 1]),
            random_state=42
        )

        self.ml_model = Pipeline([
            ("features", feature_union),
            ("classifier", classifier)
        ])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='binary'),
            'recall': make_scorer(recall_score, average='binary'),
            'f1': make_scorer(f1_score, average='binary')
        }
        
        cv_results = cross_validate(self.ml_model, X, y, cv=skf, scoring=scoring)
        print("Recall Scores:", cv_results['test_recall'])
        print("Mean Recall:", cv_results['test_recall'].mean())
        print("F1 Scores:", cv_results['test_f1'])
        print("Mean F1 Score:", cv_results['test_f1'].mean())

        self.ml_model.fit(X, y)
        print("Model trained.")

    def detect_bot(self, session_data):
        if self.ml_model is None:
            print("ML model is not trained!")
            return []
        df = self.extract_features(session_data)
        X = df.drop(columns=["user_id", "label"], errors='ignore')
        predictions = self.ml_model.predict(X)
        probabilities = self.ml_model.predict_proba(X)[:, 1]
        marked_accounts = []
        bot_count = 0
        for idx, row in df.iterrows():
            is_bot = predictions[idx] == 1
            if is_bot:
                bot_count += 1
            confidence = int(probabilities[idx] * 100)
            if row["user_id"] is None:
                print(f"Warning: User ID is None for row {idx}, skipping...")
                continue
            # print(f"ML Detection -> User {row['username']}: Confidence: {confidence}, Is_Bot: {is_bot}")
            marked_accounts.append(DetectionMark(user_id=row["user_id"], confidence=confidence, bot=is_bot))
        print(f"ML Detection Summary: Detected {bot_count} bots out of {len(df)} users.")
        return marked_accounts