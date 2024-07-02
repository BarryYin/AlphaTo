# -*- coding: utf-8 -*-
"""A Gomoku agent that can play the game with another agent."""

from typing import Optional

import json

from agentscope.message import Msg
from agentscope.agents import AgentBase
from agentscope.models import ModelResponse
import re


HINT_PROMPT = """
You should respond in the json format, which can be loaded by json in Python(pleas don't show 'json'):
{
    "thought": "analyze the present situation, and what move you should make",
    "move": [row index, column index]
}
few show like this:
{
    "thought": "I should aim to create a line of my pieces while also blocking my opponent from forming a line. Currently, there is a potential threat from the 'o' piece at [5, 6] which could lead to a vertical line. To counter this threat and potentially create my own line, I should place my piece at [6, 6]. This move will block the opponent's vertical line and create a potential horizontal line for me.",
    "move": [
        7,
        7
    ]
}
Please ensure that your response follows the specified format and includes the necessary fields.
# forbitten
this format is forbitten, don't show 'json'
```json\n{ "thought": "", "move": [ 7, 7 ] }```
"""  
# noqa

def parse_func(response: ModelResponse) -> ModelResponse:
    """Parse the response from the model into a dict with "move" and "thought"
    keys."""
    print(response.text)
    match = re.search(r'```json\n(.*?)```', response.text, re.DOTALL)
    if match is not None:
        json_text = match.group(1)  # group(1)返回第一个括号中匹配的部分
        res_dict = json.loads(json_text)
        if "move" in res_dict and "thought" in res_dict:
            return ModelResponse(raw=res_dict)
    elif(match is None):
        res_dict = json.loads(response.text)
        if "move" in res_dict and "thought" in res_dict:
            return ModelResponse(raw=res_dict)
    else:
        raise ValueError(
            f"Invalid response format in parse_func "
            f"with response: {response.text}",
        )


class GomokuAgent(AgentBase):
    """A Gomoku agent that can play the game with another agent."""

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
    ) -> None:
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
        )

        self.memory.add(Msg("system", sys_prompt, role="system"))

    def reply(self, x: Optional[dict] = None) -> dict:
        if self.memory:
            self.memory.add(x)

        msg_hint = Msg("system", HINT_PROMPT, role="system")

        prompt = self.model.format(
            self.memory.get_memory(),
            msg_hint,
        )

        response = self.model(
            prompt,
            parse_func=parse_func,
            max_retries=3,
        ).raw

        # For better presentation, we print the response proceeded by
        # json.dumps, this msg won't be recorded in memory
        self.speak(
            Msg(
                self.name,
                json.dumps(response, indent=4, ensure_ascii=False),
                role="assistant",
            ),
        )

        if self.memory:
            self.memory.add(Msg(self.name, response, role="assistant"))

        # Hide thought from the response
        return Msg(self.name, response["move"], role="assistant")
