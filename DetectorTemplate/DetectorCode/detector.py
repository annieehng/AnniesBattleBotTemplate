from abc_classes import ADetector
from teams_classes import DetectionMark
import random

class Detector(ADetector):
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