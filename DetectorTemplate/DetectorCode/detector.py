from teams_classes import DetectionMark
from abc_classes import ADetector
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# transformer function to select a column from a datafr
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.column].values.astype("U")  # convert to unicode string

# transformer function to select numeric columns
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
        
        # def feature union with TF-IDF features and numeric features
        feature_union = FeatureUnion([
            ("text", Pipeline([
                ("selector", ColumnSelector(text_feature)),
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)))
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
        
        # RandomForestClassifier
        classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced'
        )
        
        self.ml_model = Pipeline([
            ("features", feature_union),
            ("classifier", classifier)
        ])
        
        # prep features and labels
        X = df.drop(columns=["user_id", "label"])
        y = df["label"].astype(int)
        
        # perform stratified k-fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.ml_model, X, y, cv=skf, scoring='accuracy')
        print("Stratified K-Fold CV Accuracy Scores: ", cv_scores)
        print("Mean CV Accuracy: {:.2f}".format(cv_scores.mean()))
        
        # fit the model on entire training set
        self.ml_model.fit(X, y)
        print("Machine learning model trained with RandomForestClassifier.")

    def detect_bot(self, session_data):
        if self.ml_model is None:
            print("ML model is not trained!")
            return []
        df = self.extract_features(session_data)
        X = df.drop(columns=["user_id", "label"], errors='ignore')
        predictions = self.ml_model.predict(X)
        probabilities = self.ml_model.predict_proba(X)[:, 1]  # prob of being bot
        marked_accounts = []
        bot_count = 0
        for idx, row in df.iterrows():
            is_bot = predictions[idx] == 1
            if is_bot:
                bot_count += 1
            confidence = int(probabilities[idx] * 100)
            # print(f"ML Detection -> User {row['username']}: Confidence: {confidence}, Is_Bot: {is_bot}")
            marked_accounts.append(DetectionMark(user_id=row["user_id"], confidence=confidence, bot=is_bot))
        # print(f"ML Detection Summary: Detected {bot_count} bots out of {len(df)} users.")
        return marked_accounts