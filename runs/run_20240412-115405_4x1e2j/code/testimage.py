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
from agentscope.agents.dict_dialog_agent import DictDialogAgent
from agentscope.message import Msg
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





sys_img_prompt =  """
# Background B:
The text describes a complex scene or event, involving multiple elements and details, potentially suitable for creating a sketch image.
The target audience includes directors, animators, graphic designers, and other visual artists.
# Role R:
You are a content creation consultant specializing in converting text into visual content, assisting visual artists in understanding and realizing the visual elements in the text.
#Objective O:
To dismantle the original text into easily understandable visual shots, facilitating the creation of visual works.
#Key Result KR:
Each shot describes the visual elements of the scene in detail, including character actions, background settings, and emotional atmospheres.
The shot descriptions support visual artists in capturing the emotions and storyline of the text.
Sufficient details are provided to guide the actual visual production process.
#Steps S:
Carefully read the original text, identifying key scenes and elements that can be visualized.
Create a storyboard script for each key scene, including character actions, expressions, environments, and other important visual elements.
Use descriptive and specific language to write the visual details of each scene, ensuring that visual artists can create based on these descriptions.
Ensure that the shot descriptions align with the emotions and storyline of the text, enhancing the coherence between the visual and textual elements.
Conduct repeated revisions and fine-tuning to ensure the accuracy and practicality of the shot descriptions.
You should respond in the json format, which can be loaded by json in Python(pleas don't show 'json'):
{
    "thought": "analyze the present situation, and what move you should make",
    "scripts": [script1, script2,script3,script4,script5]
}
Please ensure that your response follows the specified format and includes the necessary fields.
# forbitten
this format is forbitten, don't show 'json'
"""  

imgmaker = DictDialogAgent(
            name="imgmaker1",
            model_config_name="qwen",
            #model_config_name="qwen",
            use_memory=True,
            sys_prompt=sys_img_prompt,
        )


def parse_func_haiguitang(response):
    print("1234")
    match = re.search(r'```json\n(.*?)```', response, re.DOTALL)
    if match is not None:
        json_text = match.group(1)  # group(1)返回第一个括号中匹配的部分
        json_text = json_text.replace('\\', '')  # 移除转义字符
        res_dict = json.loads(json_text) 
        return res_dict
    elif(match is None):
        res_dict = json.loads(response)
        return res_dict
    else:
        raise ValueError(
            f"Invalid response format in parse_func_wodi "
            f"with response: {response}",
        )
    

def output():
            response = "在一座图书馆中，一名著名小说家被发现死于书房内。现场门窗紧闭，无明显外力破坏痕迹。小说家坐在书桌前，手中紧握一支钢笔，面部表情痛苦，颈部有一处明显的勒痕。桌面上摆放着未完成的手稿，其内容似乎与他生前最后一部作品的主题有关。书房内还发现一只破碎的花瓶，一地散落的玫瑰花瓣，以及一张被揉皱丢弃的照片，照片上是小说家与一位陌生女子的合影。"
            

            #responses = [
            #        "小说家端坐于书桌前，面部扭曲表情痛苦，显现出死亡前的挣扎；他的手中紧握着钢笔，身后的窗户紧闭且无破损，强调房间内发生的事件非外部入侵所致。",
            #        "未完成的手稿铺陈开来，钢笔停留在纸张上，墨水尚未干涸，暗示小说家在去世前正在书写与他最后一部作品紧密相关的篇章。",
            #        "一张皱巴巴的照片被遗弃在角落或者地上，照片上清晰显示小说家与一位陌生女子的合影，这个神秘女子的身份引发了更多关于小说家生前经历和死亡原因的猜测。",
                    
             #   ]

            msg = Msg(name="system",content=response,role="assistant")
            msg = imgmaker(msg)
            json_str = json.dumps(msg.content)  #获得字典，这里可以拆分为独立函数，只要能记住词就行
            json_dict = parse_func_haiguitang(json_str)
            json_dict = json.loads(json_dict)
            scripts_list = json_dict["scripts"]  #scripts

            image_results = []
            for i, response in enumerate(scripts_list):
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