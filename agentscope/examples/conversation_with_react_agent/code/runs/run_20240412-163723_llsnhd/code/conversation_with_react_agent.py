# -*- coding: utf-8 -*-
"""An example of a conversation with a ReAct agent."""
import sys
import io

from agentscope.agents import UserAgent
from agentscope.agents.react_agent import ReActAgent
from agentscope.service import (
    bing_search,  # or google_search,
    read_text_file,
    write_text_file,
    ServiceFactory,
    ServiceResponse,
    ServiceExecStatus,
)
import agentscope

# Prepare the Bing API key and model configuration
BING_API_KEY = "76991fd9dc2e4f62a052ec90c5f58657"

YOUR_MODEL_CONFIGURATION_NAME = "gpt-4"
YOUR_MODEL_CONFIGURATION = {
        "model_type": "dashscope_chat",
        "config_name": "qwen",
        "model_name": "qwen-max",
        "api_key": "sk-ebf86b67058945fa827863a3742df0b0",
        "generate_args": {
            "temperature": 0.5
        }
}

# Prepare a new tool function
def execute_python_code(code: str) -> ServiceResponse:  # pylint: disable=C0301
    """
    Execute Python code and capture the output. Note you must `print` the output to get the result.
    Args:
        code (`str`):
            The Python code to be executed.
    """  # noqa

    # Create a StringIO object to capture the output
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        # Using `exec` to execute code
        exec(code)
    except Exception as e:
        # If an exception occurs, capture the exception information
        output = str(e)
        status = ServiceExecStatus.ERROR
    else:
        # If the execution is successful, capture the output
        output = new_stdout.getvalue()
        status = ServiceExecStatus.SUCCESS
    finally:
        # Recover the standard output
        sys.stdout = old_stdout

    # Wrap the output and status into a ServiceResponse object
    return ServiceResponse(status, output)


# Prepare the tools for the agent
tools = [
    ServiceFactory.get(bing_search, api_key=BING_API_KEY, num_results=3),
    ServiceFactory.get(execute_python_code),
    ServiceFactory.get(read_text_file),
    ServiceFactory.get(write_text_file),
]

agentscope.init(model_configs=YOUR_MODEL_CONFIGURATION)

# Create agents
agent = ReActAgent(
    name="assistant",
    model_config_name=YOUR_MODEL_CONFIGURATION_NAME,
    verbose=True,
    tools=tools,
)
user = UserAgent(name="User")

# Build
x = None
while True:
    x = user(x)
    if x.content == "exit":
        break
    x = agent(x)
