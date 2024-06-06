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
            #response = "在一座图书馆中，一名著名小说家被发现死于书房内。现场门窗紧闭，无明显外力破坏痕迹。小说家坐在书桌前，手中紧握一支钢笔，面部表情痛苦，颈部有一处明显的勒痕。桌面上摆放着未完成的手稿，其内容似乎与他生前最后一部作品的主题有关。书房内还发现一只破碎的花瓶，一地散落的玫瑰花瓣，以及一张被揉皱丢弃的照片，照片上是小说家与一位陌生女子的合影。"
            
            responses = [
                    "小说家端坐于书桌前，面部扭曲表情痛苦，显现出死亡前的挣扎；他的手中紧握着钢笔，身后的窗户紧闭且无破损，强调房间内发生的事件非外部入侵所致。",
                    "未完成的手稿铺陈开来，钢笔停留在纸张上，墨水尚未干涸，暗示小说家在去世前正在书写与他最后一部作品紧密相关的篇章。",
                    "一张皱巴巴的照片被遗弃在角落或者地上，照片上清晰显示小说家与一位陌生女子的合影，这个神秘女子的身份引发了更多关于小说家生前经历和死亡原因的猜测。",
                    # 添加更多的response...
                ]

            image_results = []
            for i, response in enumerate(responses):
                # 获取当前时间戳
                #timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                # 使用时间戳和索引作为文件名的一部分
                #output_path = os.path.join(output_dir, f"output_image_{i}_{timestamp}.png")
                image_result = text2image(input=response)
                image_results.append(image_result)


            #image_result = text2image(input=response)
            tts_result = text2audio(input=response)

            print(image_results)
            print(tts_result)


#main启动
if __name__ == "__main__":
    output()