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
        openai_weight=0.25,  # increased from 0.15 to 0.25 for higher influence
        timeseries_weight=0.1,  # weight for time-series signal
        # scaling factors for content and profile signals
        content_scale=1.2,
        profile_scale=1.1,  # slightly reduced profile influence
        # final combination weights: content vs. profile signals
        content_weight=0.6,
        profile_weight=0.4,
        # grammar error threshold
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
        self.grammar_tool = language_tool_python.LanguageTool('fr-FR')

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
                # treat 'neutral' as 0; otherwise, use the confidence score
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
        Compute a score based on the regularity (burstiness) of posting.
        If a user posts many messages with very similar intervals (low coefficient of variation), that is more suspicious.
        Returns a score from 0 to 100.
        """
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
        messages = [
            {
                "role": "system",
                "content": (
                    "Vous êtes un expert dans la détection des signes subtils de contenu automatisé sur les réseaux sociaux. "
                    "Sur la base des directives suivantes, évaluez le texte sur une échelle de 0 à 100, où :\n"
                    "  0-30 : Très humain – les publications sont décontractées, style Génération Z, généralement en minuscules, informelles et utilisent rarement des hashtags.\n"
                    "  31-60 : Possiblement légèrement automatisé.\n"
                    "  61-100 : Très probablement généré par un bot – les publications sont plus formelles, avec une capitalisation appropriée, une grammaire structurée et utilisent fréquemment des hashtags.\n\n"
                    "Exemples (extraits de nos données) :\n\n"
                    "Exemple 1 (Humain, style Gén Z) :\n"
                    "   \"ko y'a rien à voir\"\n"
                    "   -> Réponse : 15\n\n"
                    "Exemple 2 (Bot, style formel) :\n"
                    "   \"Les supporters de l'équipe FC Paris sont fiers de leur club ! #FCParis #Victoire\"\n"
                    "   -> Réponse : 70\n\n"
                    "Exemple 3 (Bot, très formel et verbeux) :\n"
                    "   \"Découvrez nos offres exclusives pour une expérience client inégalée. Profitez dès maintenant de remises exceptionnelles. #OffreSpéciale\"\n"
                    "   -> Réponse : 80\n\n"
                    "Note : Les hashtags et emojis sont rares chez les utilisateurs authentiques.\n\n"
                    "Évaluez le texte suivant et renvoyez uniquement le nombre (sans commentaire) :"
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
                model="gpt-4",  # Using GPT-4 for optimal language processing
                messages=messages,
                max_tokens=5,
                temperature=0.0
            )
            rating_str = response.choices[0].message.content.strip()
            return float(rating_str)
        except Exception as e:
            print("OpenAI query failed:", str(e))
            return 0

    def query_openai_profile(self, profile):
        # Build a structured string that includes all profile data
        profile_info = (
            f"Tweet Count: {profile.get('tweet_count', 'N/A')}\n"
            f"Z-Score: {profile.get('z_score', 'N/A')}\n"
            f"Username: {profile.get('username', 'N/A')}\n"
            f"Name: {profile.get('name', 'N/A')}\n"
            f"Description: {profile.get('description', 'N/A')}\n"
            f"Location: {profile.get('location', 'N/A')}\n"
        )
        
        messages = [
            {
                "role": "system",
                "content": (
                    "Vous êtes un expert dans la détection des signes subtils dans les données de profil automatisé sur les réseaux sociaux. "
                    "Sur la base des directives suivantes, évaluez le profil sur une échelle de 0 à 100, où :\n"
                    "  0-30 : Très humain – les profils ont généralement un nombre moyen ou faible de tweets, des noms d'utilisateur créatifs et informels, et des descriptions rédigées dans un style décontracté (souvent en minuscules, avec du slang) sans utilisation excessive de hashtags.\n"
                    "  61-100 : Très probablement généré par un bot – les profils utilisent souvent de vrais noms, affichent une grammaire formelle, une capitalisation correcte, des phrases complètes et incluent des hashtags trop travaillés.\n\n"
                    "Exemples :\n\n"
                    "Exemple 1 (Humain) :\n"
                    "   Tweet Count: 25\n"
                    "   Z-Score: -0.5\n"
                    "   Username: jeSuisCool\n"
                    "   Name: jeSuisCool\n"
                    "   Description: \"passionné de musique, amateur de vidéos spontanées et de commentaires décontractés\"\n"
                    "   Location: \"Paris\"\n"
                    "   -> Réponse : environ 15\n\n"
                    "Exemple 2 (Bot) :\n"
                    "   Tweet Count: 0\n"
                    "   Z-Score: -1.2\n"
                    "   Username: Elite_Paris\n"
                    "   Name: Elite Paris\n"
                    "   Description: \"Expert en technologie et innovations, dédié à la qualité et à l'excellence. #Innovation #Tech\"\n"
                    "   Location: \"Paris\"\n"
                    "   -> Réponse : environ 80\n\n"
                    "Évaluez les données de profil suivantes et renvoyez uniquement le nombre (sans commentaire) :\n\n"
                    f"{profile_info}"
                )
            },
            {
                "role": "user",
                "content": "Évaluation du profil :\n" + profile_info + "\nRéponse :"
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
            return float(rating_str)
        except Exception as e:
            print("OpenAI profile query failed:", str(e))
            return 0

    def detect_bot(self, session_data):
        # Group posts by author id
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
            # Extract texts for content analysis
            texts = [p.get("text", "") for p in posts if p.get("text")]

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

            lex_diversity = self.compute_lexical_diversity(texts)
            diversity_bonus = 20 if lex_diversity < 0.4 else 0
            timeseries_score = self.compute_timeseries_score(posts)

            # Add bonus if grammar score is very high (over 80)
            if avg_grammar_score > 80:
                grammar_bonus = (avg_grammar_score - 80) * 0.5
            else:
                grammar_bonus = 0

            # Combine content signals
            final_content_confidence = (
                self.similarity_weight * sim_score +
                self.sentiment_weight * sentiment_score +
                self.grammar_weight * avg_grammar_score +
                self.openai_weight * openai_score +
                diversity_bonus +
                self.timeseries_weight * timeseries_score +
                grammar_bonus
            )
            final_content_confidence *= self.content_scale

            # ------------------- PROFILE-BASED SIGNALS -------------------
            profile_confidence = 0
            tweet_count = user.get("tweet_count", 0)
            z_score = user.get("z_score", 0)
            username = (user.get("username") or "").strip()
            name = (user.get("name") or "").strip()
            description = (user.get("description") or "").strip()
            location = (user.get("location") or "").strip()

            # Tweet count heuristic: extreme counts indicate bots
            if tweet_count > 400:
                profile_confidence += 30

            # Z-score heuristic: extreme values indicate abnormal behavior
            if z_score > 5:
                profile_confidence += 50
            elif z_score > 3:
                profile_confidence += 25
            elif z_score > 2:
                profile_confidence += 10

            # Username and name heuristics: check capitalization and formatting
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

            # Name: if the name is a proper full name, add signal
            if " " in name and name == name.title():
                profile_confidence += 20

            # Description analysis
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

            # Missing location increases suspicion
            if not location:
                profile_confidence += 10

            # Use AI evaluation on full profile data
            profile_data = {
                "tweet_count": tweet_count,
                "z_score": z_score,
                "username": username,
                "name": name,
                "description": description,
                "location": location
            }
            profile_ai_score = self.query_openai_profile(profile_data)
            # Incorporate AI score at 50% weight
            profile_confidence += 0.5 * profile_ai_score

            profile_confidence *= self.profile_scale
            # ------------------------------------------------------------------

            # Combine content and profile signals via weighted average
            final_confidence = ((self.content_weight * final_content_confidence) + (self.profile_weight * profile_confidence)) / (self.content_weight + self.profile_weight)
            is_bot = final_confidence >= self.classification_threshold

            # Optional debug print:
            # print(f"User {username}: Content Score = {final_content_confidence:.2f}, Profile Score = {profile_confidence:.2f}, Final Score = {final_confidence:.2f}")

            marked_accounts.append(DetectionMark(user_id=user_id, confidence=int(final_confidence), bot=is_bot))
            
        return marked_accounts
