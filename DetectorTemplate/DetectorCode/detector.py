from abc_classes import ADetector
from teams_classes import DetectionMark
import random

class Detector(ADetector):
    def detect_bot(self, session_data):
        # todo logic
        # Example: if username contains "bot", flag it as a bot
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
