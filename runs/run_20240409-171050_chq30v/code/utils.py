import base64
from http import HTTPStatus
import requests

def covert_image_to_base64(image_path):
    # 获得文件后缀名
    ext = image_path.split(".")[-1]
    if ext not in ["gif", "jpeg", "png"]:
        ext = "jpeg"

    with open(image_path, "rb") as image_file:
        # Read the file
        encoded_string = base64.b64encode(image_file.read())

        # Convert bytes to string
        base64_data = encoded_string.decode("utf-8")

        # 生成base64编码的地址
        base64_url = f"data:image/{ext};base64,{base64_data}"
        return base64_url


def generate_image_from_prompt(des):
    import dashscope
    from dashscope.common.error import InvalidTask
    dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY") or dashscope.api_key
    assert dashscope.api_key
    # generate image
    prompt = """根据下面描述:{desc}，生成一张风景图。""".format(des)
    try:
        rsp = dashscope.ImageSynthesis.call(
            model='stable-diffusion-xl', #'wanx-lite',
            prompt=prompt,
            n=1,
            size='256*256')
        # save file to current directory
        if rsp.status_code == HTTPStatus.OK:
            for result in rsp.output.results:
                with open('assets/desc.jpg', 'wb+') as f:
                    f.write(requests.get(result.url).content)
        else:
            print('Failed, status_code: %s, code: %s, message: %s' %
                (rsp.status_code, rsp.code, rsp.message))
    except InvalidTask as e:
        print(e)


def format_cover_html():
    image_src = covert_image_to_base64('assets/wechat.png')
    return f"""
<div class="bot_cover">
    <div class="bot_avatar">
        <img src={image_src} alt="玩心眼logo">
    </div>
    <div class="bot_name">{"和Agent玩心眼"}</div>
    <div class="bot_desp">{"正在参加黑客松，请点个小心心，联系我，发你源码，wx:16763326966😊"}</div>
</div>
"""


def format_desc_html():
    return f"""
<div class="bot_cover">
    <div class="bot_rule">{"游戏介绍"}</div>
    <div class="bot_desp">{"游戏主要剧情："}
        <ul>
            <li>海龟汤：推理悬疑桌游，主持人AI出题和答疑、辅助AI参与推理、每轮都有人机互动，进行信息共享和评分</li>
            <li>谁是卧底：1个人类和3个AI斗心眼，谁是卧底大闯关，到底是AI欺骗了人类，还是人类欺骗了AI</li>
            <li>五子棋：和AI进行棋盘对弈，体验AI阿法狗的棋艺</li>
            <li>猜谜语：比智力，比心眼，AI也会进化成智力大师</li>
        </ul>
    </div>
"""


def format_welcome_html():
    config = {
        'name': "和Agent玩心眼",
        'description': '这是一款由多个大模型Agent驱动的体验Agent智力的游戏-玩心眼，快来体验吧😊',
        'introduction_label': "<br>角色介绍",
        'rule_label': "<br>规则介绍",
        'char1': '海龟汤：推理悬疑桌游，主持人AI出题和答疑、辅助AI参与推理、每轮都有人机互动，进行信息共享和评分',
        'char2': '谁是卧底：1个人类和3个AI斗心眼，谁是卧底大闯关，到底是AI欺骗了人类，还是人类欺骗了AI。',
        'char3': '五子棋：和AI进行棋盘对弈，体验AI阿法狗的棋艺',
        'char4': '猜谜语：比智力，比心眼，AI也会进化成智力大师',
        'rule1': '和1个AI比猜谜底；',
        'rule2': '和1个AI比下五子棋；',
        'rule3': '和3个AI一起玩谁是卧底；',
        'rule4': '和2个AI一起玩海龟汤；',
    }
    image_src = covert_image_to_base64('assets/wechat.png')
    return f"""
<div class="bot_cover">
    <div class="bot_avatar">
        <img src={image_src} />
    </div>
    <div class="bot_name">{config.get("name")}</div>
    <div class="bot_desc">{config.get("description")}</div>
    <div class="bot_intro_label">{config.get("introduction_label")}</div>
    <div class="bot_intro_ctx">
        <ul>
            <li>{config.get("char1")}</li>
            <li>{config.get("char2")}</li>
            <li>{config.get("char3")}</li>
        </ul>
    </div>
    <div class="bot_intro_label">{config.get("rule_label")}</div>
    <div class="bot_intro_ctx">
        <ul>
            <li>{config.get("rule1")}</li>
            <li>{config.get("rule2")}</li>
            <li>{config.get("rule3")}</li>
        </ul>
    </div>
</div>
"""
