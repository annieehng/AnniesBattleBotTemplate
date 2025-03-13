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

# Competition Environment Variables (normally set via env variables)
session_id = int(os.getenv('SESSION_ID'))
code_max_time = int(os.getenv('MAX_TIME'))
openai_api_key = os.getenv("ENV_VAR1")
# in registration, put the api key in the string value

print(f"DEBUG: env_var1 = {openai_api_key}")


# Testing Environment Variables
# session_id = 6
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

try:
    detector = Detector(openai_api_key=openai_api_key)
    get_session_response, session_dataset = get_session_data()
    
    all_id_set = set()
    for user in session_dataset.users:
        all_id_set.add(user['id'])
    
    get_session_response.raise_for_status()

    logging.info(f"Get Session response status code: {get_session_response.status_code}")
    print("Get Session response status code:", get_session_response.status_code)
    # print("Get Session response content:", json.dumps(session_dataset.__dict__, indent=4))

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(code_max_time)
    try:
        marked_account = detector.detect_bot(session_dataset)
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
