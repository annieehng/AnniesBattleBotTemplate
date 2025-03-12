import requests
import json

base_url = 'http://3.92.68.65:3000/api'
team_type = "detector"
try:
    authentication_token = requests.post('http://3.92.68.65:3000/api/auth/login', headers={'Content-Type': 'application/json'}, data=json.dumps({"team_name": "anniedetector1", "team_password": "ItsBubz123"}))
    authentication_token.raise_for_status()
except(requests.exceptions.RequestException) as error:
    error_details = error.response.json()
    print(f"An error occurred: {error}. Error Message: {error_details.get('message', 'No message available')}")
session_id = 15

header = {'Authorization': 'bearer ' + authentication_token.text, 'Content-Type': 'application/json'}

try:
    registration = requests.post(base_url + '/' + str(team_type) + '/session/' + str(session_id) + '/register', headers=header, data=json.dumps({"github_url": "https://github.com/annieehng/AnniesBattleBotTemplate", "github_branch": "main", "env_var1": "", "env_var2": ""}))
    registration.raise_for_status()
    print(registration.status_code)
except(requests.exceptions.RequestException) as error:
    error_details = error.response.json()
    print(f"An error occurred: {error}. Error Message: {error_details.get('message', 'No message available')}")