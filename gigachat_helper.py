import requests
import json
import config

import datetime

def check_token_expiration(expires_at_ms):
    expires_at_s = expires_at_ms / 1000

    expiration_date = datetime.datetime.fromtimestamp(expires_at_s)

    current_date = datetime.datetime.now()

    if expiration_date > current_date:
        return True
    else:
        return False


token_expires_at = 0
token = None

url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"


class Message:
    def __init__(self, content, role):
        super().__init__()
        self.content = content
        self.role = role

    def to_json(self):
        return {
            'content': self.content,
            'role': self.role
        }


def get_token():
    token_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    token_payload = 'scope=GIGACHAT_API_PERS'
    token_headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': '6f0b1291-c7f3-43c6-bb2e-9f3efb2dc98e',
        'Authorization': f'Basic {config.API_KEY}'
    }

    return requests.request("POST", token_url, headers=token_headers, data=token_payload)

def check_for_token():
    global token, token_expires_at
    if token is None:
        result = get_token()
        token = result.json()['access_token']
        token_expires_at = result.json()['expires_at']

    elif not check_token_expiration(token_expires_at):
        result = get_token()
        token = result.json()['access_token']
        token_expires_at = result.json()['expires_at']



def request(messages):
    check_for_token()

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }


    payload = json.dumps({
        "model": "GigaChat-preview",
        "messages": messages,
        "temperature": 1,
        "top_p": 0.1,
        "n": 1,
        "stream": False,
        "max_tokens": 1024,
        "repetition_penalty": 1
    })

    return requests.request("POST", url, headers=headers, data=payload)