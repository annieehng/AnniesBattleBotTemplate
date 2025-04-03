import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import requests
from DetectorTemplate.DetectorCode.detector import Detector
import logging
import signal
from pydantic import ValidationError
from teams_classes import DetectionMark
from api_requests import get_session_data, submit_detection
import json
import openai
import glob

# For plotting and performance evaluation.
import matplotlib.pyplot as plt
import pandas as pd

# Competition Environment Variables (normally set via env variables)
session_id = int(os.getenv('SESSION_ID'))
code_max_time = int(os.getenv('MAX_TIME'))
openai_api_key = os.getenv("ENV_VAR1")
# in registration, put the api key in the string value

# print(f"DEBUG: env_var1 = {openai_api_key}")


# Testing Environment Variables
# session_id = 18
# code_max_time = 3601 
# openai_api_key = ""

# print("session_id:", session_id)
# print("Current working directory:", os.getcwd())

openai.log = "warning"  # or "error"

# remove any existing logging handlers to force a fresh configuration
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename='run.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class IgnoreHTTPRequestFilter(logging.Filter):
    def filter(self, record):
        return "HTTP Request:" not in record.getMessage()

root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.addFilter(IgnoreHTTPRequestFilter())

class TimeoutError(Exception):
    """Custom exception for timeout errors."""
    pass

class MarkingMissingUsers(Exception):
    """Custom exception for marking missing users errors."""
    pass

class MultipleDetectionForUser(Exception):
    """Custom exception for multiple detection for user errors."""
    pass

def handler(signum, frame):
    raise TimeoutError("Timeout Error:")

logging.info(f"START SESSION {session_id}")

# function to plot results
def plot_detected_bots(detections):
    """
    Plots a bar chart of detected bots and their confidence scores.
    
    Parameters:
      detections (list): List of DetectionMark objects, each with attributes:
                         - user_id (or username)
                         - confidence (an integer 0-100)
                         - bot (boolean indicating if detected as bot)
    """
    # Filter to keep only detected bots.
    bots = [d for d in detections if d.bot]
    
    if not bots:
        print("No bots detected.")
        return

    # Create a DataFrame from the detected bots.
    bot_data = pd.DataFrame({
        "User": [d.user_id for d in bots],
        "Confidence": [d.confidence for d in bots]
    })

    # Create a bar plot.
    plt.figure(figsize=(10, 6))
    plt.bar(bot_data["User"], bot_data["Confidence"], color="red")
    plt.xlabel("User ID")
    plt.ylabel("Confidence Score")
    plt.title("Detected Bots and Their Confidence Scores")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

try:

    # create detector object 
    
    detector = Detector()

    # detector = EnsembleDetector(openai_api_key=openai_api_key)

    get_session_response, session_dataset = get_session_data()
    
    all_id_set = set()
    for user in session_dataset.users:
        all_id_set.add(user['id'])
    
    get_session_response.raise_for_status()

    logging.info(f"Get Session response status code: {get_session_response.status_code}")
    print("Get Session response status code:", get_session_response.status_code)
    # print("Get Session response content:", json.dumps(session_dataset.__dict__, indent=4))

    # ------------------ NEW: Train ML Model Using Local Data ------------------
    # Define the local folders (assumed to be in the same directory as main_detector.py)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    human_folder = os.path.join(base_dir, "human_data")
    bot_folder = os.path.join(base_dir, "bot_data")
    
    # Helper function to load and combine JSON files from a folder using a pattern
    def load_combined_json(folder, pattern):
        combined = []
        file_pattern = os.path.join(folder, pattern)
        for fp in glob.glob(file_pattern):
            with open(fp, 'r') as f:
                data = json.load(f)
            combined.extend(data)
        return combined

    # Load all JSON files from the human and bot folders
    human_posts = load_combined_json(human_folder, "session_*_human_posts.json")
    human_users = load_combined_json(human_folder, "session_*_human_users.json")
    bot_posts = load_combined_json(bot_folder, "session_*_bot_posts.json")
    bot_users = load_combined_json(bot_folder, "session_*_bot_users.json")
    
    # Label users and ensure they have an "id" key
    for user in human_users:
        user["id"] = user["user_id"]
        user["label"] = 0  # human
    for user in bot_users:
        user["id"] = user["user_id"]
        user["label"] = 1  # bot

    # Create a combined training dataset
    class SessionData:
        pass

    training_session_data = SessionData()
    training_session_data.posts = human_posts + bot_posts
    training_session_data.users = human_users + bot_users

    # Train the machine learning model on the combined training data
    detector.train_ml_model(training_session_data)
    logging.info("Machine learning model trained.")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(code_max_time)
    try:

        # detector call here 

        marked_account = detector.detect_bot(session_dataset)
        # marked_account = detector.detect_bot_ensemble(session_dataset)


        if not isinstance(marked_account[0], DetectionMark):  # Check each element is a DetectionMark
            raise TypeError(f"Expected DetectionMark instance, got {type(marked_account[0])}.")
        
        marked_id_set = set()
        for account in marked_account:
            marked_id_set.add(account.user_id)
        
        if len(marked_account) == 0:  # Empty submission
            detections_submission = []
        elif not len(marked_account) == len(marked_id_set):
            raise MultipleDetectionForUser("Every user must have exactly one DetectionMark.")
        elif not all_id_set == marked_id_set:
            raise MarkingMissingUsers("DetectionMark missing for some users.")
        else:
            detections_submission = [user.to_dict() for user in marked_account]
    except TimeoutError as exc:
        logging.error(f"{exc} Code took more than one hour. Proceeding with empty submission.")
        print(f"{exc} Code took more than one hour. Proceeding with empty submission.")
        detections_submission = []

    submission_confirmation = submit_detection(detections_submission)
    
    submission_confirmation.raise_for_status()
    
    logging.info(f"Detection Submission response status code: {submission_confirmation.status_code}")
    print("Detection Submission response status code:", submission_confirmation.status_code)

    # Plot performance evaluation based on detection results.
    # This assumes that session_dataset.users include ground truth labels.
    plot_detected_bots(marked_account)
    
    signal.alarm(0)
    logging.info(f"END SESSION {session_id}")

except (requests.exceptions.RequestException, ValidationError, TypeError, MarkingMissingUsers, MultipleDetectionForUser) as exc:
    if isinstance(exc, requests.exceptions.RequestException):
        logging.error(f"An error occurred: {exc}")
        print("An error occurred:", exc)
    elif isinstance(exc, ValidationError):
        if exc.errors()[0]['type'] == 'int_from_float':
            logging.error(f"DetectionMark Object Error: Confidence should be an int between 0 and 100. Error: {exc.errors()}.")
            print(f"DetectionMark Object Error: Confidence should be an int between 0 and 100. Error: {exc.errors()}.")
        else:
            logging.error(f"DetectionMark Object Error: {exc.errors()}.")
            print(f"DetectionMark Object Error: {exc.errors()}.")
    elif isinstance(exc, (TypeError, MarkingMissingUsers, MultipleDetectionForUser)):
        logging.error(exc)
        print(exc)

# At the very end, flush and shutdown logging, then print the log file content for verification.
for handler in logging.getLogger().handlers:
    handler.flush()
logging.shutdown()

"""
try:
    with open("run.log", "r") as f:
        print("Contents of run.log:")
        print(f.read())
except Exception as e:
    print("Error reading run.log:", e)
"""
