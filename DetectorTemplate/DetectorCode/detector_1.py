from teams_classes import DetectionMark
from abc_classes import ADetector
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# download French stopwords (if not already downloaded)
nltk.download('stopwords')
french_stop_words = stopwords.words('french')

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
        text_feature = 'text'
        
        feature_union = FeatureUnion([
            ("text", Pipeline([
                ("selector", ColumnSelector(text_feature)),
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words=french_stop_words))
            ])),
            ("tweet_count", Pipeline([
                ("selector", NumericSelector("tweet_count")),
                ("scaler", StandardScaler())
            ])),
            ("z_score", Pipeline([
                ("selector", NumericSelector("z_score")),
                ("scaler", StandardScaler())
            ]))
        ])
        
        classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        self.ml_model = Pipeline([
            ("features", feature_union),
            ("classifier", classifier)
        ])
        
        X = df.drop(columns=["user_id", "label"])
        y = df["label"].astype(int)
        self.ml_model.fit(X, y)
        print("Machine learning model trained with RandomForestClassifier for French data")

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
            print(f"ML Detection -> User {row['username']}: Confidence: {confidence}, Is_Bot: {is_bot}")
            marked_accounts.append(DetectionMark(user_id=row["user_id"], confidence=confidence, bot=is_bot))
        print(f"ML Detection Summary: Detected {bot_count} bots out of {len(df)} users.")
        return marked_accounts

