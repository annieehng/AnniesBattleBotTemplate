import datetime
import math
import random
import language_tool_python
import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from teams_classes import DetectionMark
from abc_classes import ADetector


class Detector(ADetector):
    def __init__(
        self,
        openai_api_key,
        # weights for each content-based signal
        similarity_weight=0.3,
        sentiment_weight=0.2,
        grammar_weight=0.2,
        openai_weight=0.25,  # increased for higher influence
        timeseries_weight=0.1,  # weight for time-series signal
        # additional scaling factors for content and profile signals
        content_scale=1.2,
        profile_scale=1.1,
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

        # Use a sentiment analysis model that supports French.
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        # Initialize the grammar tool for French.
        self.grammar_tool = language_tool_python.LanguageTool('fr')

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
                # Depending on the model, you might need to adjust how you interpret the score.
                scores.append(float(result['score']))
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
        query_text = text
        messages = [
            {
                "role": "system",
                "content": (
                    "Vous êtes un expert en détection de contenu automatisé sur les réseaux sociaux. "
                    "Évaluez le texte suivant sur une échelle de 0 à 100, où :\n"
                    "  0-30 : Très humain – style décontracté, informel, typiquement tout en minuscule et peu d'hashtags\n"
                    "  31-60 : Possiblement légèrement automatisé\n"
                    "  61-100 : Très probablement généré par un bot – texte plus formel, bonne capitalisation, grammaire structurée et usage fréquent des hashtags\n\n"
                    "Exemples :\n"
                    "Exemple 1 (Humain) : 'coucou ça va ?'\n"
                    "   -> Réponse : 15\n\n"
                    "Exemple 2 (Bot) : 'Bonjour à tous, veuillez trouver ci-joint les dernières mises à jour. #Information'\n"
                    "   -> Réponse : 70\n\n"
                    "Maintenant, évaluez le texte suivant et répondez uniquement par un nombre :"
                )
            },
            {
                "role": "user",
                "content": f"Texte : \"{query_text}\"\nRéponse:"
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
            try:
                return float(rating_str)
            except ValueError:
                return 0
        except Exception as e:
            return 0

    def detect_bot(self, session_data):
        # Group posts by user id.
        user_posts = {}
        for post in session_data.posts:
            author_id = post.get("author_id")
            if author_id not in user_posts:
                user_posts[author_id] = []
            user_posts[author_id].append(post)

        marked_accounts = []
        for user in session_data.users:
            user_id = user["id"]
            posts = user_posts.get(user_id, [])
            texts = [p.get("text", "") for p in posts if p.get("text")]

            sim_score = self.compute_similarity_score(texts) * 100
            sentiment_score = self.compute_sentiment_score(texts) * 100
            grammar_scores = [self.compute_grammar_score(t) for t in texts]
            avg_grammar_score = np.mean(grammar_scores) if grammar_scores else 0
            openai_raw = self.query_openai(texts[0]) if texts else 0
            try:
                openai_score = float(openai_raw)
            except Exception:
                openai_score = 0

            lex_diversity = self.compute_lexical_diversity(texts)
            diversity_bonus = 20 if lex_diversity < 0.4 else 0
            timeseries_score = self.compute_timeseries_score(posts)

            final_content_confidence = (
                self.similarity_weight * sim_score +
                self.sentiment_weight * sentiment_score +
                self.grammar_weight * avg_grammar_score +
                self.openai_weight * openai_score +
                diversity_bonus +
                self.timeseries_weight * timeseries_score
            )
            final_content_confidence *= self.content_scale

            # PROFILE-BASED SIGNALS
            profile_confidence = 0
            tweet_count = user.get("tweet_count", 0)
            z_score = user.get("z_score", 0)
            username = (user.get("username") or "").strip()
            name = (user.get("name") or "").strip()
            description = (user.get("description") or "").strip()
            location = (user.get("location") or "").strip()

            if tweet_count > 300:
                profile_confidence += 30

            if z_score > 5:
                profile_confidence += 50
            elif z_score > 3:
                profile_confidence += 25
            elif z_score > 2:
                profile_confidence += 10

            username_lower = username.lower()
            if "bot" in username_lower:
                profile_confidence += 50
            elif username.isalnum() and username != username_lower:
                profile_confidence += 30
            else:
                num_digits = sum(c.isdigit() for c in username)
                if num_digits >= 4:
                    profile_confidence += 30
                elif any(ch in username for ch in "_-.") and num_digits >= 2:
                    profile_confidence += 20

            if " " in name and name == name.title():
                profile_confidence += 20

            if not description:
                profile_confidence += 20
            else:
                if "#" in description:
                    profile_confidence += 30
                if description.count(". ") >= 1:
                    profile_confidence += 10
                if description != description.lower():
                    profile_confidence += 10
                else:
                    profile_confidence -= 10

            if not location:
                profile_confidence += 10

            profile_confidence *= self.profile_scale

            final_confidence = ((self.content_weight * final_content_confidence) + (self.profile_weight * profile_confidence)) / (self.content_weight + self.profile_weight)
            is_bot = final_confidence >= self.classification_threshold

            print(f"User {username}: Content Score={final_content_confidence:.2f}, Profile Score={profile_confidence:.2f}, Final Score={final_confidence:.2f}, Is_Bot={is_bot}")

            marked_accounts.append(DetectionMark(user_id=user_id, confidence=int(final_confidence), bot=is_bot))
            
        return marked_accounts
