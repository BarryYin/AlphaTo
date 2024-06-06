import requests
import json

url = 'https://internlm-chat.intern-ai.org.cn/puyu/api/v1/chat/completions'
header = {
    'Content-Type':
    'application/json',
    "Authorization":
    "Bearer eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiIwMTgzMTgiLCJyb2wiOiJST0xFX1JFR0lTVEVSIiwiaXNzIjoiT3BlblhMYWIiLCJpYXQiOjE3MTc1NzQzOTEsImNsaWVudElkIjoiZWJtcnZvZDZ5bzBubHphZWsxeXAiLCJwaG9uZSI6IjE2NzYzMzI2OTY2IiwidXVpZCI6IjBlMWZlMzUxLTM0ZjktNGRhNi05OGIwLWYwMDY3NjViN2MzNiIsImVtYWlsIjoiMTUwOTAwODA2MEBxcS5jb20iLCJleHAiOjE3MzMxMjYzOTF9.9PLiMepoXJLetlfQhKunPySb01FI0mj1CJHbGwzGSyGnVpFwECT8RijTbZEFnnuVZykQ-G-1leo8Hrn1cjJDVw"
}
data = {
    "model": "internlm2-latest",  
    "messages": [{
        "role": "user",
        "text": "你好~"
    }],
    "temperature": 0.8,
    "top_p": 0.9
}

res = requests.post(url, headers=header, data=json.dumps(data))
print(res)
print(res.status_code)
print(res.json())
print(res.json()["choices"][0]["message"]["content"])