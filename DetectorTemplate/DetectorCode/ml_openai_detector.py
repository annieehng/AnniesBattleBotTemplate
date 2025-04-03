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
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

# Define a simple transformer to select a column from a DataFrame.
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.column].values.astype("U")  # convert to unicode string

# Transformer to select numeric columns
class NumericSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.column]]

# Original Detector class with various signal methods
class Detector(ADetector):
    def __init__(
        self,
        openai_api_key,
        similarity_weight=0.4,
        sentiment_weight=0.2,
        grammar_weight=0.2,
        openai_weight=0.2,  # increased for higher influence
        timeseries_weight=0.1,
        content_scale=1.2,
        profile_scale=1.1,
        content_weight=0.3,
        profile_weight=0.2,
        grammar_error_threshold=0.01,
        classification_threshold=40  
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
        self.ml_model = None  # To be set after training

    def compute_similarity_score(self, texts):
        if len(texts) < 2:
            return 0
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        cosim = cosine_similarity(tfidf_matrix)
        n = cosim.shape[0]
        sum_sim = np.sum(cosim) - n  # subtract self-similarity (diagonals)
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
        # Higher score means fewer errors.
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
        """
        Uses GPT-4 to evaluate the aggregated posts for overall stylistic consistency.
        The prompt instructs the model to consider grammar, tone, punctuation, formality,
        hashtag usage, and formatting to determine how likely the posts are bot-generated.
        """
        if text is None:
            text = ""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in detecting subtle signs of automated social media content. "
                    "Instead of evaluating individual posts, consider the aggregated posts as a whole. "
                    "Based on the following guidelines, assess the overall stylistic consistency of the text by evaluating: "
                    "1) Grammar consistency, 2) Tone, 3) Punctuation, 4) Formality, 5) Hashtag usage, and 6) General formatting. \n\n"
                    "Please rate the aggregated posts on a scale from 0 to 100, where:\n"
                    "  0-30: Very human-like – posts are casual, informal, diverse in tone and structure, often all lower-case and spontaneous.\n"
                    "  31-60: Possibly mildly automated – posts show some uniformity in style yet retain natural variability.\n"
                    "  61-100: Highly likely bot-generated – posts are overly consistent, formal, with structured grammar, proper punctuation, and frequent or unnatural hashtag usage.\n\n"
                    "Examples:\n"
                    "Example 1 (Human): \"what movie should i watch tn (reason why i wanna watch it) you can submit others in comments\" -> Answer: 15\n"
                    "Example 2 (Bot): \"Tar Heels forever! Born a Tar Heel, sty a Tar Heel, can’t shake off the pride! #GoHeels\" -> Answer: 75\n"
                    "Example 3 (Bot): \"Found a hilarious note on my door this morning. Fun fact: My neighbors apparently have a PhD in passive aggression. 😄 Might start leaving them thank you notes for the laughs. #JustNeighborThings\" -> Answer: 80\n\n"
                    "Now, based on the aggregated posts provided below, evaluate the overall stylistic consistency across the multiple dimensions mentioned, "
                    "and return only the final numeric score (do not include any additional commentary):"
                )
            },
            {
                "role": "user",
                "content": f"Aggregated Posts: \"{text}\""
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
        
        self.ml_model = Pipeline([
            ("features", feature_union),
            ("classifier", LogisticRegression(max_iter=1000))
        ])
        
        X = df.drop(columns=["user_id", "label"])
        y = df["label"].astype(int)
        self.ml_model.fit(X, y)
        print("Machine learning model trained.")

    # ML detection method (existing)
    def detect_bot(self, session_data):
        if self.ml_model is None:
            print("ML model is not trained!")
            return []
        df = self.extract_features(session_data)
        X = df.drop(columns=["user_id", "label"], errors='ignore')
        predictions = self.ml_model.predict(X)
        probabilities = self.ml_model.predict_proba(X)[:, 1]  # Probability of being bot
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

# New Ensemble Detector that combines multiple signals
class EnsembleDetector(Detector):
    def __init__(
        self,
        openai_api_key,
        # Inherit base parameters as needed...modify as necessary
        ml_weight=0.5,
        openai_weight=0.2,
        timeseries_weight=0.05,
        similarity_weight=0.15,
        grammar_weight=0.05,
        ensemble_threshold=45,  # threshold for final decision on 0-100 scale
        **kwargs
    ):
        super().__init__(openai_api_key, **kwargs)
        # Define weights for each signal; they should sum to 1 ideally, but the absolute values work if threshold is adjusted.
        self.ensemble_weights = {
            'ml': ml_weight,
            'openai': openai_weight,
            'timeseries': timeseries_weight,
            'similarity': similarity_weight,
            'grammar': grammar_weight
        }
        self.ensemble_threshold = ensemble_threshold

    def detect_bot_ensemble(self, session_data):
        # Extract features for aggregation (for ML and basic info)
        df = self.extract_features(session_data)
        marked_accounts = []
        
        # Group posts by user for per-user signal calculation
        posts_by_user = {}
        for post in session_data.posts:
            author_id = post.get("author_id")
            posts_by_user.setdefault(author_id, []).append(post)
        
        for idx, row in df.iterrows():
            user_id = row['user_id']
            username = row.get("username", "unknown")
            # Retrieve all posts for this user
            user_posts = posts_by_user.get(user_id, [])
            texts = [p.get("text", "") for p in user_posts]
            aggregated_text = row['text']  # already the concatenated posts
            
            # ML Signal: use the ML model probability if available.
            ml_score = 0
            if self.ml_model is not None:
                temp_df = pd.DataFrame([row])
                ml_prob = self.ml_model.predict_proba(temp_df.drop(columns=["user_id", "label"], errors='ignore'))[0][1]
                ml_score = ml_prob * 100  # scale to 0-100
            # OpenAI Signal: prompt-engineered score on aggregated text.
            openai_score = self.query_openai(aggregated_text)
            # Timeseries Signal: evaluate posting regularity.
            timeseries_score = self.compute_timeseries_score(user_posts)
            # Similarity Signal: high similarity among posts can be a bot indicator.
            similarity_score = 0
            if len(texts) >= 2:
                similarity = self.compute_similarity_score(texts)
                similarity_score = similarity * 100  # scale to 0-100
            # Grammar Signal: average grammar score over all posts.
            grammar_scores = [self.compute_grammar_score(text) for text in texts if text.strip()]
            grammar_score = np.mean(grammar_scores) if grammar_scores else 0

            # Combine weighted signals
            final_score = (
                self.ensemble_weights['ml'] * ml_score +
                self.ensemble_weights['openai'] * openai_score +
                self.ensemble_weights['timeseries'] * timeseries_score +
                self.ensemble_weights['similarity'] * similarity_score +
                self.ensemble_weights['grammar'] * grammar_score
            )
            
            is_bot = final_score >= self.ensemble_threshold
            print(
                f"Ensemble Detection -> User {username}: "
                f"ML: {ml_score:.1f}, OpenAI: {openai_score:.1f}, Timeseries: {timeseries_score:.1f}, "
                f"Similarity: {similarity_score:.1f}, Grammar: {grammar_score:.1f} => Final: {final_score:.1f} | Bot: {is_bot}"
            )
            marked_accounts.append(DetectionMark(user_id=user_id, confidence=int(final_score), bot=is_bot))
        
        # Print summary: number of bots detected out of total users processed.
        bot_count = sum(1 for account in marked_accounts if account.bot)
        total_users = len(df)
        print(f"Ensemble Detection Summary: Detected {bot_count} bots out of {total_users} users.")
        
        return marked_accounts
