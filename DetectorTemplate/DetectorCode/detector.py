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
            description = user["description"] or ""
            location = user["location"] or ""

            confidence = 0
            is_bot = False

            # heuristics
            if tweet_count > 1000:
                confidence += 40
            if z_score > 3:
                confidence += 30
            if "bot" in username.lower():
                confidence += 20
            if not description: # account has no profile desciption
                confidence += 10
            if not location:
                confidence += 5
            
            # check the confidence 
            if confidence >= 50:
                is_bot = True

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