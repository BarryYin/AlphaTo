# -*- coding: utf-8 -*-
"""A simple example for conversation between user and assistant agent."""
import agentscope
from agentscope.agents import DialogAgent
from agentscope.agents.user_agent import UserAgent
from agentscope.pipelines.functional import sequentialpipeline


def main() -> None:
    """A basic conversation demo"""

    agentscope.init(
        model_configs=[
            {
                "config_name": "InternLM2",
                "model_type": "InternLM2_chat",
                #"model_name": "internlm2-latest",
                "model_name": "internlm2",
                #"api_key": "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiIwMTgzMTgiLCJyb2wiOiJST0xFX1JFR0lTVEVSIiwiaXNzIjoiT3BlblhMYWIiLCJpYXQiOjE3MTc1NzQzOTEsImNsaWVudElkIjoiZWJtcnZvZDZ5bzBubHphZWsxeXAiLCJwaG9uZSI6IjE2NzYzMzI2OTY2IiwidXVpZCI6IjBlMWZlMzUxLTM0ZjktNGRhNi05OGIwLWYwMDY3NjViN2MzNiIsImVtYWlsIjoiMTUwOTAwODA2MEBxcS5jb20iLCJleHAiOjE3MzMxMjYzOTF9.9PLiMepoXJLetlfQhKunPySb01FI0mj1CJHbGwzGSyGnVpFwECT8RijTbZEFnnuVZykQ-G-1leo8Hrn1cjJDVw",
                "api_key": 'YOUR_API_KEY',
                "client_args": {
                    #"base_url": "https://internlm-chat.intern-ai.org.cn/puyu/api/v1/"
                    "base_url": "http://localhost:23333/v1/"
                },
                "generate_args": {
                    "temperature": 0.7
                }
            },
            {
                "model_type": "post_api_chat",
                "config_name": "internLM",
                #"api_url" : 'https://internlm-chat.intern-ai.org.cn/puyu/api/v1/chat/completions',
                "api_url" : 'https://localhost:23333/v1/chat/completions',
                "header" : {
                    'Content-Type':
                    'application/json',
                    "Authorization":
                    "Bearer eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiIwMTgzMTgiLCJyb2wiOiJST0xFX1JFR0lTVEVSIiwiaXNzIjoiT3BlblhMYWIiLCJpYXQiOjE3MTc1NzQzOTEsImNsaWVudElkIjoiZWJtcnZvZDZ5bzBubHphZWsxeXAiLCJwaG9uZSI6IjE2NzYzMzI2OTY2IiwidXVpZCI6IjBlMWZlMzUxLTM0ZjktNGRhNi05OGIwLWYwMDY3NjViN2MzNiIsImVtYWlsIjoiMTUwOTAwODA2MEBxcS5jb20iLCJleHAiOjE3MzMxMjYzOTF9.9PLiMepoXJLetlfQhKunPySb01FI0mj1CJHbGwzGSyGnVpFwECT8RijTbZEFnnuVZykQ-G-1leo8Hrn1cjJDVw",
                },
                "data" : {
                    "model": "internlm2-latest",
                    "messages": [{
                            "role": "user",
                            "text": "你好~"
                        }],
                    "temperature": 0.8,
                    "top_p": 0.9
                },
            },
            
        ],
    )

    # Init two agents
    dialog_agent = DialogAgent(
        name="Assistant",
        sys_prompt="你是一位画家，等待用户的提问.",
        model_config_name="InternLM2",  # replace by your model config name
    )
    user_agent = UserAgent()

    # start the conversation between user and assistant
    x = None
    while x is None or x.content != "exit":
        x = sequentialpipeline([dialog_agent,user_agent], x)


if __name__ == "__main__":
    main()
