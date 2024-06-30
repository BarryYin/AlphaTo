# -*- coding: utf-8 -*-
# @Date     : 2024/04/06
# @Author   : ZixiaHu
# @File     : text2image_agent.py
# @微信公众号 : AI Freedom
# @知乎      : RedHerring


from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.service import ServiceFactory
import os
from modelscope_agent.tools import dashscope_tools
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import re
import json
import requests
from datetime import datetime

output_dir = "./ci_workspace/"
os.makedirs(output_dir, exist_ok=True)



#output_path2 = os.path.join(output_dir, "output_image.png")

def text_to_image(image_url, text, output_path, font_path='./assets/SimHei.ttf', font_size=30,
                        text_color=(255, 255, 255), position=(50, 50)):
    """
    将文字添加到图像上
    参数:
    image_url (str): 输入图像的链接
    text (str): 要添加到图像上的文字
    output_path (str): 输出图像的路径
    font_path (str, optional): 字体文件的路径, 默认为'path/to/font.ttf'
    font_size (int, optional): 字体大小, 默认为30
    text_color (tuple, optional): 文字颜色 (R, G, B), 默认为白色
    position (tuple, optional): 文字在图像上的位置 (x, y), 默认为(50, 50)
    """
    # 下载图像
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    
    
    # 创建绘图对象
    draw = ImageDraw.Draw(image)
    
    # 加载字体
    font = ImageFont.truetype(font_path, font_size)
    
    # 在图像上添加文字
    draw.text((position[0], position[1]), text, font=font, fill=text_color)
    
    # 保存图像
    image.save(output_path)




class Text2ImageAgent(AgentBase):
    def __init__(
            self,
            name: str,
            sys_prompt: str,
            model_config_name: str,
            use_memory: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            use_memory=use_memory
        )
        self.memory.add(Msg("image_system", sys_prompt, role="system"))

    def reply(self, x: dict = None, input: str = None) -> dict:

        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # 使用时间戳作为文件名的一部分
        output_path2 = os.path.join(output_dir, f"output_image_{timestamp}.png")
        tool = dashscope_tools.TextToImageTool()
        result = tool.call({'text': input, 'resolution': '1024*1024'})

        # result = "![IMAGEGEN](https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/1d/5d/20240406/8d820c8d/b5ea70d8-dd42-4797-8b43-de0258afce9b-1.png?Expires=1712499404&OSSAccessKeyId=LTAI5tQZd8AEcZX6KZV4G8qL&Signature=K%2BJGlpmmyfcwSaIW0UUheGWywLg%3D)"
        
        image_url = re.search(r'\((.*?)\)', result).group(1)
        print(image_url)

        func, func_json = ServiceFactory.get(text_to_image, image_url=image_url, text=input, output_path=output_path2)
        

        print("===func, func_json====")
        print(func)
        print(json.dumps(func_json, indent=4))

        result = func()
        result = f"ci_workspace/output_image_{timestamp}.png"
        #result = f'<img src={output_path2}>'
        print(result)

        return result

