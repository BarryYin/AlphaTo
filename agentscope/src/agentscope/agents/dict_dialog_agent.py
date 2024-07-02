# -*- coding: utf-8 -*-
"""A dict dialog agent that using `parse_func` and `fault_handler` to
parse the model response."""
import json
from typing import Any, Optional, Callable
from loguru import logger

from ..message import Msg
from .agent import AgentBase
from ..models.model import ModelResponse
from ..prompt import PromptType
from ..utils.tools import _convert_to_str


def parse_dict(response: ModelResponse) -> ModelResponse:
    """Parse function for DictDialogAgent"""
    try:
        response_dict = json.loads(response.text)
    except json.decoder.JSONDecodeError:
        # Sometimes LLM may return a response with single quotes, which is not
        # a valid JSON format. We replace single quotes with double quotes and
        # try to load it again.
        # TODO: maybe using a more robust json library to handle this case
        response_dict = json.loads(response.text.replace("'", '"'))

    return ModelResponse(raw=response_dict)


def default_response(response: ModelResponse) -> ModelResponse:
    """The default response of fault_handler"""
    return ModelResponse(raw={"speak": response.text})


class DictDialogAgent(AgentBase):
    """An agent that generates response in a dict format, where user can
    specify the required fields in the response via prompt, e.g.

    .. code-block:: python

        prompt = "... Response in the following format that can be loaded by
        python json.loads()
        {
            "thought": "thought",
            "speak": "thoughts summary to say to others",
            # ...
        }"

    This agent class is an example for using `parse_func` and `fault_handler`
    to parse the output from the model, and handling the fault when parsing
    fails. We take "speak" as a required field in the response, and print
    the speak field as the output response.

    For usage example, please refer to the example of werewolf in
    `examples/game_werewolf`"""

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
        use_memory: bool = True,
        memory_config: Optional[dict] = None,
        parse_func: Optional[Callable[..., Any]] = parse_dict,
        fault_handler: Optional[Callable[..., Any]] = default_response,
        max_retries: Optional[int] = 3,
        prompt_type: Optional[PromptType] = None,
    ) -> None:
        """Initialize the dict dialog agent.

        Arguments:
            name (`str`):
                The name of the agent.
            sys_prompt (`Optional[str]`, defaults to `None`):
                The system prompt of the agent, which can be passed by args
                or hard-coded in the agent.
            model_config_name (`str`, defaults to None):
                The name of the model config, which is used to load model from
                configuration.
            use_memory (`bool`, defaults to `True`):
                Whether the agent has memory.
            memory_config (`Optional[dict]`, defaults to `None`):
                The config of memory.
            parse_func (`Optional[Callable[..., Any]]`,
            defaults to `parse_dict`):
                The function used to parse the model output,
                e.g. `json.loads`, which is used to extract json from the
                output.
            fault_handler (`Optional[Callable[..., Any]]`,
            defaults to `default_response`):
                The function used to handle the fault when parse_func fails
                to parse the model output.
            max_retries (`Optional[int]`, defaults to `None`):
                The maximum number of retries when failed to parse the model
                output.
            prompt_type (`Optional[PromptType]`, defaults to
            `PromptType.LIST`):
                The type of the prompt organization, chosen from
                `PromptType.LIST` or `PromptType.STRING`.
        """
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            use_memory=use_memory,
            memory_config=memory_config,
        )

        # record the func and handler for parsing and handling faults
        self.parse_func = parse_func
        self.fault_handler = fault_handler
        self.max_retries = max_retries

        if prompt_type is not None:
            logger.warning(
                "The argument `prompt_type` is deprecated and "
                "will be removed in the future.",
            )

    def reply(self, x: dict = None) -> dict:
        """Reply function of the agent.
        Processes the input data, generates a prompt using the current
        dialogue memory and system prompt, and invokes the language
        model to produce a response. The response is then formatted
        and added to the dialogue memory.

        Args:
            x (`dict`, defaults to `None`):
                A dictionary representing the user's input to the agent.
                This input is added to the dialogue memory if provided.
        Returns:
            A dictionary representing the message generated by the agent in
            response to the user's input. It contains at least a 'speak' key
            with the textual response and may include other keys such as
            'agreement' if provided by the language model.

        Raises:
            `json.decoder.JSONDecodeError`:
                If the response from the language model is not valid JSON,
                it defaults to treating the response as plain text.
        """
        # record the input if needed
        if self.memory:
            self.memory.add(x)

        # prepare prompt
        prompt = self.model.format(
            Msg("system", self.sys_prompt, role="system"),
            self.memory and self.memory.get_memory(),  # type: ignore[arg-type]
        )

        # call llm
        response = self.model(
            prompt,
            parse_func=self.parse_func,
            fault_handler=self.fault_handler,
            max_retries=self.max_retries,
        ).raw

        # logging raw messages in debug mode
        logger.debug(json.dumps(response, indent=4, ensure_ascii=False))

        # In this agent, if the response is a dict, we treat "speak" as a
        # special key, which represents the text to be spoken
        if isinstance(response, dict) and "speak" in response:
            msg = Msg(
                self.name,
                response["speak"],
                role="assistant",
                **response,
            )
        else:
            msg = Msg(self.name, response, role="assistant")

        # Print/speak the message in this agent's voice
        self.speak(msg)

        # record to memory
        if self.memory:
            # Convert the response dict into a string to store in memory
            msg_memory = Msg(
                name=self.name,
                content=_convert_to_str(response),
                role="assistant",
            )
            self.memory.add(msg_memory)

        return msg
