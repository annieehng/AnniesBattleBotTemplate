import datetime
import math
import random

import language_tool_python
import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from teams_classes import DetectionMark
from transformers import pipeline
from abc_classes import ADetector


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
        If a user posts many messages with very similar intervals (i.e. low coefficient of variation), that is more suspicious.
        
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
                model="gpt-4",  # Consider switching to ChatGPT o3-mini-high for a lightweight alternative
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
                    "You are an expert in detecting subtle signs of automated social media profile data. "
                    "Based on the following guidelines, rate the profile on a scale from 0 to 100, where:\n"
                    "  0-30: Very human-like – profiles typically have average or low tweet counts, casual and creative usernames, informal names, and descriptions that are written in a relaxed, Gen Z style (often all lower-case, with slang and little formal punctuation) with little or no use of hashtags.\n"
                    "  61-100: Highly likely bot-generated – profiles often use full real names, display formal grammar, proper capitalization, full sentences, and include cheesy hashtags and polished language.\n\n"
                    "Examples:\n\n"
                    "Example 1 (Human):\n"
                    "   Tweet Count: 22\n"
                    "   Z-Score: -0.2142360868\n"
                    "   Username: Coscorrodrift\n"
                    "   Name: coscorrodrift (streaming on YT at 16:30CET)\n"
                    "   Description: 'youtube video watcher, commenter and maker (100/100 vids)\n"
                    "                 now streaming at 16:30CET/10:30EST on youtube\n"
                    "                 hmu if you need help with youtube or writing for $'\n"
                    "   Location: spiritually in SF\n"
                    "   -> Answer: around 15\n\n"
                    "Example 2 (Bot):\n"
                    "   Tweet Count: 0\n"
                    "   Z-Score: -1.2616125109\n"
                    "   Username: UrbanWanderlust\n"
                    "   Name: Maya Rivers\n"
                    "   Description: 'Explorer of cityscapes and green spaces | Photography Enthusiast | Sharing local gems from around the world | NYC native | #TravelBlogger #UrbanAdventures'\n"
                    "   Location: N/A\n"
                    "   -> Answer: around 80\n\n"
                    "Now, evaluate the following profile data and return only the number (do not include any commentary):\n\n"
                    f"{profile_info}"
                )
            },
            {
                "role": "user",
                "content": "Profile Evaluation:\n" + profile_info + "\nAnswer:"
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
        # Group posts by user id
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

            # Combine content signals
            final_content_confidence = (
                self.similarity_weight * sim_score +
                self.sentiment_weight * sentiment_score +
                self.grammar_weight * avg_grammar_score +
                self.openai_weight * openai_score +
                diversity_bonus +
                self.timeseries_weight * timeseries_score
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

            # Tweet count heuristic: extreme counts can indicate bots
            if tweet_count > 400:
                profile_confidence += 30

            # Z-score heuristic: extreme z-scores indicate abnormal behavior
            if z_score > 5:
                profile_confidence += 50
            elif z_score > 3:
                profile_confidence += 25
            elif z_score > 2:
                profile_confidence += 10

            # Username and Name heuristics: check for proper capitalization and formatting
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

            # Name: if the name is a full proper name (e.g., first and last), this can be a bot signal
            if " " in name and name == name.title():
                profile_confidence += 20

            # Description analysis:
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

            # Location missing increases suspicion
            if not location:
                profile_confidence += 10

            # Use AI evaluation on the full profile data (username, name, description, tweet_count, z_score, location)
            profile_data = {
                "tweet_count": tweet_count,
                "z_score": z_score,
                "username": username,
                "name": name,
                "description": description,
                "location": location
            }
            profile_ai_score = self.query_openai_profile(profile_data)  # AI score for the full profile data
            # Incorporate the AI score at 50% weight into the overall profile confidence (increased from 30%)
            profile_confidence += 0.5 * profile_ai_score

            profile_confidence *= self.profile_scale
            # ------------------------------------------------------------------

            # Combine content and profile signals using weighted average
            final_confidence = ((self.content_weight * final_content_confidence) + (self.profile_weight * profile_confidence)) / (self.content_weight + self.profile_weight)
            is_bot = final_confidence >= self.classification_threshold

            # Debug print (optional)
            # print(f"User {username}: Content Score={final_content_confidence:.2f}, Profile Score={profile_confidence:.2f}, Final Score={final_confidence:.2f}")

            marked_accounts.append(DetectionMark(user_id=user_id, confidence=int(final_confidence), bot=is_bot))
            
        return marked_accounts

"""
first contest code:

        marked_account = []
        
        for user in session_data.users:
            user_id = user["id"]  
            username = user["username"]

            confidence = 0
            is_bot = False
            
            if "bot" in username.lower():
                confidence = 100  # full confidence its a bot
                is_bot = True

            marked_account.append(DetectionMark(user_id=user_id, confidence=confidence, bot=is_bot))

        return marked_account

"""

"""
second competition code:

def detect_bot(self, session_data):
        # todo logic
        # Example: trying basic logic heuristics 
        marked_account = []

        for user in session_data.users:
            user_id = user["id"]
            username = user["username"]
            tweet_count = user.get("tweet_count", 0)
            z_score = user.get("z_score", 0)
            description = user.get("description", "")
            if description:
                description = description.strip().lower()
            location = user.get("location", "")
            if location:
                location = location.strip()

            confidence = 0
            is_bot = False

            # high tweet count 
            if tweet_count > 5000:
                confidence += 50
            elif tweet_count > 1000:
                confidence += 30
            elif tweet_count < 5:  
                confidence += 20  

            # z-score 
            if z_score > 5: # extremely high activity
                confidence += 50  
            elif z_score > 3:
                confidence += 25
            elif z_score > 2:
                confidence += 10  

            # judging the username 
            num_count = sum(c.isdigit() for c in username)
            if "bot" in username:
                confidence += 40  
            elif num_count >= 4:  
                confidence += 30  
            elif any(char in username for char in "_-.") and num_count >= 2:
                confidence += 20  

            # checking their profile 
            if not description:
                confidence += 20  
            if not location:
                confidence += 10  # 

            # looking for suspicious keywords in bio description
            spam_keywords = ["click here", "follow me", "free money", "crypto", "giveaway", "discount", "xxx", "adult", "onlyfans", "breaking news"]
            if any(word in description for word in spam_keywords):
                confidence += 40  

            generic_phrases = ["manifesting", "positive vibes", "grateful", "dream big", "hustle", "motivation"]
            if any(phrase in description for phrase in generic_phrases):
                confidence += 20  

            if confidence >= 40:
                is_bot = True  

            # print("RESULT = user: " + username + " is_bot: " + str(is_bot))  

            marked_account.append(DetectionMark(user_id=user_id, confidence=confidence, bot=is_bot))

        return marked_account

third session code:
class Detector(ADetector):
    def __init__(self, openai_api_key, similarity_weight=0.3, sentiment_weight=0.2, grammar_weight=0.2, openai_weight=0.3, grammar_error_threshold=0.01, content_weight=0.6, profile_weight=0.4):
        # set the OpenAI key
        openai.api_key = openai_api_key

        # weights to combine content signals
        self.similarity_weight = similarity_weight
        self.sentiment_weight = sentiment_weight
        self.grammar_weight = grammar_weight
        self.openai_weight = openai_weight
        self.grammar_error_threshold = grammar_error_threshold

        # weights for final combination -> content vs. profile-based signals
        self.content_weight = content_weight
        self.profile_weight = profile_weight
        
        # init sentiment analysis pipeline 
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
        # init grammar tool
        self.grammar_tool = language_tool_python.LanguageTool('en-US') # will modify once french tweets are implemented

    def compute_similarity_score(self, texts):
        if len(texts) < 2:
            return 0
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        cosine_sim = cosine_similarity(tfidf_matrix)

        # compute average of off-diagonal similarity scores
        n = cosine_sim.shape[0]
        sum_sim = np.sum(cosine_sim) - n  # subtract the diagonal ones (which are 1)
        count = n * (n - 1)

        return sum_sim / count if count > 0 else 0
        
    def compute_sentiment_score(self, texts):
        if not texts:
            return 0
        scores = []
        for text in texts:
            try:
                result = self.sentiment_analyzer(text)[0]
                label = result['label']
                score = result['score']
                if label.lower() == 'neutral':
                    scores.append(0)
                else:
                    scores.append(score)

            except Exception:
                scores.append(0)

        return np.mean(scores)
        
    def compute_grammar_score(self, text):
        words = text.split()
        if len(words) == 0:
            return 0
        matches = self.grammar_tool.check(text)
        error_rate = len(matches) / len(words)
        # a low error rate (i.e. fewer errors) may indicate auto-generation
        return max(0, 1 - error_rate / self.grammar_error_threshold) * 100 # convert score on a 0–100 scale
        
    def query_openai(self, text):
        try:
            # keep improving prompt for next competition
            prompt = ( 
                f"Rate the following text on how likely it is bot-generated on a scale of 0 to 100. "
                f"Just provide a number.\nText: \"{text}\"\nAnswer:"
            )
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=5,
                temperature=0
            )
            rating_str = response.choices[0].text.strip()
            rating = float(rating_str)
            return rating
        
        except Exception:
            return 0
        
    def detect_bot(self, session_data):
        # todo logic: combine profile heuristics with signals using imported libraries, as well as OpenAI prompting

        # group posts by user id
        user_posts = {}
        for post in session_data.posts:
            author_id = post.get("author_id")
            if author_id not in user_posts:
                user_posts[author_id] = []
            user_posts[author_id].append(post.get("text", ""))
        
        marked_account = []
        for user in session_data.users:
            user_id = user["id"]
            texts = user_posts.get(user_id, [])
            
            # signals
            sim_score = self.compute_similarity_score(texts) * 100
            sentiment_score = self.compute_sentiment_score(texts) * 100
            grammar_scores = [self.compute_grammar_score(text) for text in texts]
            avg_grammar_score = np.mean(grammar_scores) if grammar_scores else 0
            openai_score = self.query_openai(texts[0]) if texts else 0
            
            final_content_confidence = (
                self.similarity_weight * sim_score +
                self.sentiment_weight * sentiment_score +
                self.grammar_weight * avg_grammar_score +
                self.openai_weight * openai_score
            )
            
            # profile data
            profile_confidence = 0
            tweet_count = user.get("tweet_count", 0)
            z_score = user.get("z_score", 0)
            username = user.get("username", "").lower()
            description = (user.get("description") or "").strip()
            location = user.get("location")
            
            # tweet count heuristics
            if tweet_count > 5000:
                profile_confidence += 50
            elif tweet_count > 1000:
                profile_confidence += 30
            elif tweet_count < 5:
                profile_confidence += 20
            
            # z-score heuristics
            if z_score > 5:
                profile_confidence += 50
            elif z_score > 3:
                profile_confidence += 25
            elif z_score > 2:
                profile_confidence += 10
            
            # username analysis
            num_count = sum(c.isdigit() for c in username)
            if "bot" in username:
                profile_confidence += 50  # increased bonus
            elif num_count >= 4:
                profile_confidence += 30
            elif any(char in username for char in "_-.") and num_count >= 2:
                profile_confidence += 20
            
            # profile completeness
            if not description:
                profile_confidence += 20
            if not location:
                profile_confidence += 10
            
            # description spam/generic keyword heuristics
            spam_keywords = ["click here", "follow me", "free money", "crypto", "giveaway", "discount", "xxx", "adult", "onlyfans", "breaking news"]
            if any(spam_kw in description.lower() for spam_kw in spam_keywords):
                profile_confidence += 40
            generic_phrases = ["manifesting", "positive vibes", "grateful", "dream big", "hustle", "motivation"]
            if any(phrase in description.lower() for phrase in generic_phrases):
                profile_confidence += 20
            
            # -- FINAL COMBINATION USING WEIGHTED RMS --
            final_confidence = math.sqrt(
                (self.content_weight * (final_content_confidence ** 2) + self.profile_weight * (profile_confidence ** 2))
                / (self.content_weight + self.profile_weight)
            )

            is_bot = final_confidence >= 50

            print("RESULT = user: " + username + " is_bot: " + str(is_bot))
            
            marked_account.append(DetectionMark(user_id=user_id, confidence=int(final_confidence), bot=is_bot))
            
        return marked_account

        
session 4 code:

class Detector(ADetector):
    def __init__(
        self,
        openai_api_key,
        # weights for each content-based signal
        similarity_weight=0.3,
        sentiment_weight=0.2,
        grammar_weight=0.2,
        openai_weight=0.3,
        # additional scaling factor to amplify content signals
        content_scale=1.2,
        # profile signals 
        profile_scale=1.3,
        # weighted combination of content vs. profile signals
        content_weight=0.6,
        profile_weight=0.4,
        # grammar threshold
        grammar_error_threshold=0.01,
        # classification threshold
        classification_threshold=50
    ):
        openai.api_key = openai_api_key
        self.similarity_weight = similarity_weight
        self.sentiment_weight = sentiment_weight
        self.grammar_weight = grammar_weight
        self.openai_weight = openai_weight
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

    def compute_similarity_score(self, texts):
        if len(texts) < 2:
            return 0
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        cosim = cosine_similarity(tfidf_matrix)
        n = cosim.shape[0]
        sum_sim = np.sum(cosim) - n  # subtract diagonal (1.0 each)
        count = n * (n - 1)
        return sum_sim / count if count > 0 else 0

    def compute_sentiment_score(self, texts):
        if not texts:
            return 0
        scores = []
        for text in texts:
            try:
                result = self.sentiment_analyzer(text)[0]
                # ff neutral, treat it as 0
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
        # fewer errors => higher grammar_score
        return max(0, 1 - error_rate / self.grammar_error_threshold) * 100

    def query_openai(self, text):
        # better prompt
        prompt = f"
        You are a specialized social media bot detection system. 
        Analyze the following text and decide how likely it is generated by a social media bot. 
        Focus on unnatural phrasing, spam-like patterns, or generic style. 
        Return a single integer from 0 (definitely human) to 100 (definitely a bot), 
        and do not include any explanation.

        Text: "{text}"
        Answer:
        "
        
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=5,
                temperature=0
            )
            rating_str = response.choices[0].text.strip()
            return float(rating_str)
        except:
            return 0

    def detect_bot(self, session_data):
        # group posts by user
        user_posts = {}
        for post in session_data.posts:
            author_id = post.get("author_id")
            if author_id not in user_posts:
                user_posts[author_id] = []
            user_posts[author_id].append(post.get("text", ""))

        marked_accounts = []
        for user in session_data.users:
            user_id = user["id"]
            texts = user_posts.get(user_id, [])

            # content signals
            sim_score = self.compute_similarity_score(texts) * 100
            sentiment_score = self.compute_sentiment_score(texts) * 100
            grammar_scores = [self.compute_grammar_score(t) for t in texts]
            avg_grammar_score = np.mean(grammar_scores) if grammar_scores else 0
            openai_score = self.query_openai(texts[0]) if texts else 0

            content_confidence = (
                self.similarity_weight * sim_score +
                self.sentiment_weight * sentiment_score +
                self.grammar_weight * avg_grammar_score +
                self.openai_weight * openai_score
            )
            # scale up content signals
            content_confidence *= self.content_scale

            # profile signals
            profile_confidence = 0
            tweet_count = user.get("tweet_count", 0)
            z_score = user.get("z_score", 0)
            username = (user.get("username") or "").lower()
            description = (user.get("description") or "").lower()
            location = user.get("location", "")

            # tweet_count
            if tweet_count > 5000:
                profile_confidence += 50
            elif tweet_count > 1000:
                profile_confidence += 30
            elif tweet_count < 5:
                profile_confidence += 20

            # z_score
            if z_score > 5:
                profile_confidence += 50
            elif z_score > 3:
                profile_confidence += 25
            elif z_score > 2:
                profile_confidence += 10

            # username analysis
            num_count = sum(c.isdigit() for c in username)
            if "bot" in username:
                profile_confidence += 50
            elif username.isalnum():
                profile_confidence += 40
            elif num_count >= 4:
                profile_confidence += 30
            elif any(ch in username for ch in "_-.") and num_count >= 2:
                profile_confidence += 20

            # profile completeness
            if not description.strip():
                profile_confidence += 20
            if not location:
                profile_confidence += 10

            spam_keywords = [
                "click here", "follow me", "free money", "crypto", 
                "giveaway", "discount", "xxx", "adult", "onlyfans", 
                "breaking news"
            ]
            if any(word in description for word in spam_keywords):
                profile_confidence += 40

            generic_phrases = [
                "manifesting", "positive vibes", "grateful", 
                "dream big", "hustle", "motivation"
            ]
            if any(phrase in description for phrase in generic_phrases):
                profile_confidence += 20

            profile_confidence *= self.profile_scale

            # based on the user's content and profile
            final_confidence = (
                (self.content_weight * content_confidence) +
                (self.profile_weight * profile_confidence)
            ) / (self.content_weight + self.profile_weight)

            # mark the account as bot or not
            is_bot = final_confidence >= self.classification_threshold

            # print("RESULT = user: " + username + " is_bot: " + str(is_bot))
            
            marked_accounts.append(DetectionMark(user_id=user_id, confidence=int(final_confidence), bot=is_bot))

        return marked_accounts

"""

