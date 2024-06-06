# -*- coding: utf-8 -*-
# @Date     : 2024/04/06
# @Author   : ZixiaHu
# @File     : text2audio_agent.py
# @微信公众号 : AI Freedom
# @知乎      : RedHerring


from agentscope.agents import AgentBase
from agentscope.message import Msg
from modelscope_agent.tools import dashscope_tools


class Text2AudioAgent(AgentBase):
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

        tool = dashscope_tools.SambertTtsTool()
        result = tool.call({'text': input})

        # result = """<audio src="/tmp/ci_workspace/sambert_tts_audio.wav"/>"""

        print(result)
        return result
