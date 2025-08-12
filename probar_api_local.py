import requests
import json

url = "http://localhost:1234/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "itlwas/mistral-7b-instruct-v0.1",
    "messages": [
        {"role": "system", "content": "Always answer in rhymes. Today is Thursday"},
        {"role": "user", "content": "What day is it today?"}
    ],
    "temperature": 0.7,
    "max_tokens": -1,
    "stream": False
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
