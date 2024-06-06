import os
import random
import shutil
import traceback

import gradio as gr
import agentscope
import json
#import modelscope_studio as mgr
#from config_utils import get_avatar_image, get_ci_dir, parse_configuration
#from gradio_utils import format_cover_html, format_cover_html_author
#from modelscope_agent.schemas import Message
#from modelscope_agent.utils.logger import agent_logger as logger
#from modelscope_studio.components.Chatbot.llm_thinking_presets import qwen
#from user_core import init_user_chatbot_agent
from agents.text2audio_agent import Text2AudioAgent
from agents.text2image_agent import Text2ImageAgent

model_configs = json.load(open('model_configs.json', 'r'))
#model_configs[0]["api_key"] = os.environ["DASHSCOPE_API_KEY"]
model_configs[0]["api_key"] = "sk-ebf86b67058945fa827863a3742df0b0" 
model_configs[1]["api_key"] = "sk-ebf86b67058945fa827863a3742df0b0" 
DASHSCOPE_API_KEY = "sk-ebf86b67058945fa827863a3742df0b0"

agentscope.init(model_configs=model_configs)


text2image = Text2ImageAgent(
    name="image",
    model_config_name="qwen",
    sys_prompt="你是一个图片助手",
)

text2audio = Text2AudioAgent(
    name="audio",
    model_config_name="qwen",
    sys_prompt="你是一个音频助手",
)

def output():
            response = "可乐，在山间"
            image_result = text2image(input=response)
            tts_result = text2audio(input=response)

            print(image_result)
            print(tts_result)


#main启动
if __name__ == "__main__":
    output()