from abc_classes import ADetector
from teams_classes import DetectionMark
import random
from openai import OpenAI

import datetime
import language_tool_python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import math


class Detector(ADetector):
    def __init__(
        self,
        openai_api_key,
        # weights for each content-based signal
        similarity_weight=0.3,
        sentiment_weight=0.2,
        grammar_weight=0.2,
        openai_weight=0.15,  # reduced weight
        timeseries_weight=0.1,  # new weight for time-series signal
        # additional scaling factors for content and profile signals
        content_scale=1.2,
        profile_scale=1.1,  # slightly reduced profile influence
        # weights for final combination: content vs. profile signals
        content_weight=0.6,
        profile_weight=0.4,
        # grammar error threshold for scoring
        grammar_error_threshold=0.01,
        # classification threshold (lowered to reduce false positives)
        classification_threshold=45  # lowered threshold
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

        # French sentiment analyzer using Allociné sentiment model
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="tblard/tf-allocine-sentiment",
            tokenizer="tblard/tf-allocine-sentiment"
        )
        # French grammar tool
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
                # for Allociné model, treat 'neutral' as 0; otherwise, use the confidence
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
        """
        Compute a score based on the regularity (burstiness) of posts.
        If an author posts many messages with very similar time intervals (low CV), it may be suspicious

        Returns a score from 0 to 100.
        """
        if len(posts) < 2:
            return 0
        times = []
        for post in posts:
            created_at = post.get("created_at")
            if created_at:
                try:
                    # ISO 8601 format; adjust if necessary
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
        # if there are many posts and the time intervals are very similar, assign a high score
        if num_posts >= 10 and cv < 0.2:
            return min(100, 100 * (0.2 - cv) / 0.2)
        else:
            return 0

    def query_openai(self, text):
        # french prompt for evaluating the likelihood of automation
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in detecting subtle signs of automated social media content. "
                    "In a competitive setting, human tweets are naturally varied, even if short, while bot-generated tweets "
                    "tend to show overly consistent phrasing, minimal personalization, and repetition. "
                    "Based solely on the text content, return a single integer rating from 0 to 100, where:\n"
                    "  0-30: Very human-like\n"
                    "  31-60: Possibly mildly automated\n"
                    "  61-100: Highly likely bot-generated\n"
                    "If the text seems natural and varied, output a low number. Do not include any commentary—only the number.\n\n"
                    "Examples:\n"
                    "Texte : 'J'ai passé une merveilleuse matinée à me promener dans le parc avec mon chien.' -> Réponse : 15\n"
                    "Texte : 'Pris mon café. Prêt pour le match. Hâte d'y être.' -> Réponse : 25\n"
                    "Texte : 'Pris mon café. Pris mon café. Pris mon café.' -> Réponse : 90\n\n"
                    "Now, evaluate the following text and return only the number:"
                )
            },
            {
                "role": "user",
                "content": f"Texte : \"{text}\"\nRéponse :"
            }
        ]
        try:
            client = OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=5,
                temperature=0.0
            )
            rating_str = response.choices[0].message.content.strip()
            print("OpenAI response:", rating_str)
            return float(rating_str)
        except Exception as e:
            print("OpenAI query failed:", str(e))
            return 0

    def detect_bot(self, session_data):
        # group posts by author; keep the full post for time info.
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
            texts = [p.get("text", "") for p in posts]

            sim_score = self.compute_similarity_score(texts) * 100
            sentiment_score = self.compute_sentiment_score(texts) * 100
            grammar_scores = [self.compute_grammar_score(t) for t in texts]
            avg_grammar_score = np.mean(grammar_scores) if grammar_scores else 0
            openai_raw = self.query_openai(texts[0]) if texts else 0

            try:
                openai_score = float(openai_raw)
            except Exception:
                print(f"OpenAI query returned non-numeric value for user {user_id}: {openai_raw}")
                openai_score = 0

            print(f"OpenAI answer for user {user_id}: {openai_score}")

            lex_diversity = self.compute_lexical_diversity(texts)
            diversity_bonus = 20 if lex_diversity < 0.4 else 0

            timeseries_score = self.compute_timeseries_score(posts)

            # combine content signals using a weighted sum
            final_content_confidence = (
                self.similarity_weight * sim_score +
                self.sentiment_weight * sentiment_score +
                self.grammar_weight * avg_grammar_score +
                self.openai_weight * openai_score +
                diversity_bonus +
                self.timeseries_weight * timeseries_score
            )
            final_content_confidence *= self.content_scale

            # profile signals
            profile_confidence = 0
            tweet_count = user.get("tweet_count", 0)
            z_score = user.get("z_score", 0)
            username = (user.get("username") or "").lower()
            description = (user.get("description") or "").lower()
            location = user.get("location", "")

            if tweet_count > 5000:
                profile_confidence += 50
            elif tweet_count > 1000:
                profile_confidence += 30
            elif tweet_count < 5:
                profile_confidence += 20

            if z_score > 5:
                profile_confidence += 50
            elif z_score > 3:
                profile_confidence += 25
            elif z_score > 2:
                profile_confidence += 10

            num_count = sum(c.isdigit() for c in username)
            if "bot" in username:
                profile_confidence += 50
            elif username.isalnum():
                profile_confidence += 40
            elif num_count >= 4:
                profile_confidence += 30
            elif any(ch in username for ch in "_-.") and num_count >= 2:
                profile_confidence += 20

            if not description.strip():
                profile_confidence += 20
            if not location:
                profile_confidence += 10

            # adjust spam and generic keyword heuristics to French terms
            spam_keywords = ["cliquez ici", "suivez moi", "argent gratuit", "crypto", "cadeau", "réduction", "xxx", "adulte", "onlyfans", "dernières nouvelles"]
            if any(spam_kw in description for spam_kw in spam_keywords):
                profile_confidence += 40
            generic_phrases = ["manifestation", "ondes positives", "reconnaissant", "rêver grand", "boulot", "motivation"]
            if any(phrase in description for phrase in generic_phrases):
                profile_confidence += 20

            profile_confidence *= self.profile_scale

            # combine final signals with a weighted average
            final_confidence = ((self.content_weight * final_content_confidence) + (self.profile_weight * profile_confidence)) / (self.content_weight + self.profile_weight)

            is_bot = final_confidence >= self.classification_threshold

            # print("RESULT = user:", username, "is_bot:", is_bot)

            marked_accounts.append(DetectionMark(user_id=user_id, confidence=int(final_confidence), bot=is_bot))

        return marked_accounts
