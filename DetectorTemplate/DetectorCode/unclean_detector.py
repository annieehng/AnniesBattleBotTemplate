from teams_classes import DetectionMark
from abc_classes import ADetector
import datetime
import math
import random
import language_tool_python
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Define a simple transformer to select a column from a DataFrame.
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.column].values.astype("U")  # convert to unicode string

# Transformer to select numeric columns.
class NumericSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.column]]

class Detector(ADetector):
    def __init__(
        self,
        openai_api_key,
        # weights for each content-based signal
        similarity_weight=0.3,
        sentiment_weight=0.2,
        grammar_weight=0.2,
        openai_weight=0.25,  # increased from 0.15 to 0.25 for higher influence
        timeseries_weight=0.1,  # weight for time-series signal
        # additional scaling factors for content and profile signals
        content_scale=1.2,
        profile_scale=1.1,  # slightly reduced profile influence
        # weights for final combination: content vs. profile signals
        content_weight=0.6,
        profile_weight=0.4,
        # grammar error threshold for scoring
        grammar_error_threshold=0.01,
        # classification threshold (lowered to reduce false positives)
        classification_threshold=45  
    ):
        self.openai_api_key = openai_api_key
        self.similarity_weight = similarity_weight
        self.sentiment_weight = sentiment_weight
        self.grammar_weight = grammar_weight
        self.openai_weight = openai_weight
        self.timeseries_weight = timeseries_weight
        self.content_scale = content_scale
        self.profile_scale = profile_scale
        self.content_weight = content_weight
        self.profile_weight = profile_weight
        self.grammar_error_threshold = grammar_error_threshold
        self.classification_threshold = classification_threshold

        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )
        self.grammar_tool = language_tool_python.LanguageTool('en-US')
        self.ml_model = None  # Will be set during training

    def compute_similarity_score(self, texts):
        if len(texts) < 2:
            return 0
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        cosim = cosine_similarity(tfidf_matrix)
        n = cosim.shape[0]
        sum_sim = np.sum(cosim) - n  # subtract diagonal ones (which are 1)
        count = n * (n - 1)
        return sum_sim / count if count > 0 else 0

    def compute_sentiment_score(self, texts):
        if not texts:
            return 0
        scores = []
        for text in texts:
            try:
                result = self.sentiment_analyzer(text)[0]
                scores.append(0 if result['label'].lower() == 'neutral' else result['score'])
            except Exception:
                scores.append(0)
        return np.mean(scores)

    def compute_grammar_score(self, text):
        words = text.split()
        if not words:
            return 0
        matches = self.grammar_tool.check(text)
        error_rate = len(matches) / len(words)
        return max(0, 1 - error_rate / self.grammar_error_threshold) * 100

    def compute_lexical_diversity(self, texts):
        ratios = []
        for text in texts:
            words = text.split()
            if not words:
                continue
            ratio = len(set(words)) / len(words)
            ratios.append(ratio)
        return np.mean(ratios) if ratios else 0

    def compute_timeseries_score(self, posts):
        if len(posts) < 2:
            return 0
        times = []
        for post in posts:
            created_at = post.get("created_at")
            if created_at:
                try:
                    dt = datetime.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    times.append(dt)
                except Exception:
                    continue
        if len(times) < 2:
            return 0
        times.sort()
        intervals = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]
        if not intervals:
            return 0
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        cv = std_interval / mean_interval if mean_interval > 0 else 0
        num_posts = len(times)
        if num_posts >= 10 and cv < 0.2:
            return min(100, 100 * (0.2 - cv) / 0.2)
        else:
            return 0

    def query_openai(self, text):
        if text is None:
            text = ""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in detecting subtle signs of automated social media content. "
                    "Based on the following guidelines, rate the text on a scale from 0 to 100, where:\n"
                    "  0-30: Very human-like – posts are casual, Gen Z style, typically all lower-case, informal, and rarely use hashtags\n"
                    "  31-60: Possibly mildly automated\n"
                    "  61-100: Highly likely bot-generated – posts are more formal, follow proper capitalization, structured grammar, and frequently include hashtags\n\n"
                    "Examples from our datasets:\n\n"
                    "Example 1 (Human - Gen Z style):\n"
                    "   'what movie should i watch tn (reason why i wanna watch it)\n"
                    "    you can submit others in comments'\n"
                    "   -> Answer: 15\n\n"
                    "Example 2 (Bot - Millennial style):\n"
                    "   'Tar Heels forever! Born a Tar Heel, sty a Tar Heel, can’t shake off the pride! #GoHeels'\n"
                    "   -> Answer: 75\n\n"
                    "Example 3 (Bot - Cheesy, overly formal style):\n"
                    "   'Found a hilarious note on my door this morning. Fun fact: My neighbors apparently have a PhD in passive aggression. 😄 Might start leaving them thank you notes for the laughs. #JustNeighborThings'\n"
                    "   -> Answer: 80\n\n"
                    "Note: Hashtags and emojis are uncommon in genuine human posts nowadays, but bot posts tend to include them for emphasis or promotional reasons.\n\n"
                    "Now, evaluate the following text and return only the number (do not include any additional commentary):"
                )
            },
            {
                "role": "user",
                "content": f"Text: \"{text}\"\nAnswer:"
            }
        ]
        try:
            client = OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=5,
                temperature=0.0
            )
            rating_str = response.choices[0].message.content.strip()
            print("query_openai: Raw response:", rating_str)
            return float(rating_str)
        except Exception as e:
            print("query_openai query failed:", str(e))
            return 0

    # Machine Learning Pipeline Methods
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
        """
        Train the machine learning model using a stratified train-test split and stratified k-fold cross-validation
        to ensure balanced proportions of bots/humans and to check generalizability.
        We replace Logistic Regression with RandomForestClassifier for increased robustness.
        """
        df = self.extract_features(training_session_data)
        df = df[df['label'].notnull()]
        text_feature = 'text'
        
        # Define feature union with TF-IDF features and numeric features.
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
        
        # Use a RandomForestClassifier.
        classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced'
        )
        
        self.ml_model = Pipeline([
            ("features", feature_union),
            ("classifier", classifier)
        ])
        
        # Prepare features and labels.
        X = df.drop(columns=["user_id", "label"])
        y = df["label"].astype(int)
        
        # Perform Stratified K-Fold Cross-Validation.
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.ml_model, X, y, cv=skf, scoring='accuracy')
        print("Stratified K-Fold CV Accuracy Scores: ", cv_scores)
        print("Mean CV Accuracy: {:.2f}".format(cv_scores.mean()))
        
        # Fit the model on the entire training set.
        self.ml_model.fit(X, y)
        print("Machine learning model trained with RandomForestClassifier.")

    # ML detection method
    def detect_bot(self, session_data):
        if self.ml_model is None:
            print("ML model is not trained!")
            return []
        df = self.extract_features(session_data)
        X = df.drop(columns=["user_id", "label"], errors='ignore')
        predictions = self.ml_model.predict(X)
        probabilities = self.ml_model.predict_proba(X)[:, 1]  # Probability of being bot.
        marked_accounts = []
        bot_count = 0
        for idx, row in df.iterrows():
            is_bot = predictions[idx] == 1
            if is_bot:
                bot_count += 1
            confidence = int(probabilities[idx] * 100)
            print(f"ML Detection -> User {row['username']}: Confidence: {confidence}, Is_Bot: {is_bot}")
            marked_accounts.append(DetectionMark(user_id=row["user_id"], confidence=confidence, bot=is_bot))
        print(f"ML Detection Summary: Detected {bot_count} bots out of {len(df)} users.")
        return marked_accounts
