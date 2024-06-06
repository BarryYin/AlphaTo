import os
import json
import gradio as gr
from gradio.components import Chatbot
import threading
import agentscope
from agentscope.agents import DialogAgent
from agentscope.agents.user_agent import UserAgent
from agentscope.message import Msg
from utils import format_welcome_html
from dotenv import find_dotenv, load_dotenv
_ = load_dotenv(find_dotenv()) 
from agentscope.agents import AgentBase
import numpy as np
import time
from agentscope.models import ModelResponse
from typing import Optional
import re
import base64
import matplotlib.pyplot as plt
import matplotlib.patches as patches

uid = threading.current_thread().name
host_avatar = 'assets/host_image.png'
user_avatar = 'assets/parti_image.png'
judge_avatar = 'assets/judge_image.png'
judge_AI_avatar = 'assets/judge_image.png'
parti_avatar = 'assets/ai.png'

def init_game(state):
    model_configs = json.load(open('model_configs.json', 'r'))
    os.environ["DASHSCOPE_API_KEY"] = "sk-ebf86b67058945fa827863a3742df0b0"
    
    model_configs[0]["api_key"] = "sk-ebf86b67058945fa827863a3742df0b0"
    
    agents = agentscope.init(
        model_configs=model_configs,
        agent_configs="agent_configs.json",
    )
    state['host_agent'] = agents[0]
    state['judge_agent'] = agents[1]
    state['judge_AI_agent'] = agents[2]
    state['parti_agent'] = agents[3]
    state['user_agent'] = UserAgent()
    return state

def is_valid_input(input):
    pattern = r'^\[\d+,\d+\]$'
    if re.match(pattern, input):
        numbers = [int(num) for num in re.findall(r'\d+', input)]
        if numbers[0] > 14 or numbers[1] > 14:
            return False
        else:
            return True
    else:
        return False

def gomoku_init_user(state):
    # seed = state.get('session_seed', random.randint(0, 1000000000))
    user_agent = UserAgent()
    state['user_agent'] = user_agent

    
    return state

pre_host_key = ''

def board2img(board: np.ndarray, save_path: str)->str:
    size = board.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, size - 1)
    ax.set_ylim(0, size - 1)
    
    for i in range(size):
        ax.axhline(i, color='black', linewidth=1)
        ax.axvline(i, color='black', linewidth=1)
        
    for y in range(size):
        for x in range(size):
            if board[y, x] == NAME_TO_PIECE[NAME_WHITE]:  # white player
                circle = patches.Circle((x, y), 0.45, 
                                        edgecolor='black', 
                                        facecolor='black',
                                        zorder=10)
                ax.add_patch(circle)
            elif board[y, x] == NAME_TO_PIECE[NAME_BLACK]:  # black player
                circle = patches.Circle((x, y), 0.45, 
                                        edgecolor='black', 
                                        facecolor='white',
                                        zorder=10)
                ax.add_patch(circle)
    # Hide the axes and invert the y-axis
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(range(size))
    ax.set_yticklabels(range(size))
    ax.invert_yaxis()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)  # Close the figure to free memory
    return save_path


#board
CURRENT_BOARD_PROMPT_TEMPLATE = """The current board is as follows:
{board}
{player}, it's your turn."""

NAME_BLACK = "Alice"
NAME_WHITE = "Bob"

# The mapping from name to piece
NAME_TO_PIECE = {
    NAME_BLACK: "o",
    NAME_WHITE: "x",
}

EMPTY_PIECE = "0"



class BoardAgent(AgentBase):
    """A board agent that can host a Gomoku game."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name, use_memory=False)

        # Init the board
        self.size = 15
        self.board = np.full((self.size, self.size), EMPTY_PIECE)

        # Record the status of the game
        self.game_end = False

    def reply(self, x: dict = None) -> dict:
        if x is None:
            # Beginning of the game
            content = (
                "Welcome to the Gomoku game! Black player goes "
                "first. Please make your move."
            )
        else:
            #row, col = x["content"]
            if(type(x["content"]) == list):
                    row, col = x["content"]
            else:
                    row, col = json.loads(x["content"])

            self.assert_valid_move(row, col)

            if self.check_win(row, col, NAME_TO_PIECE[x["name"]]):
                content = f"The game ends, {x['name']} wins!"
                self.game_end = True
            else:
                # change the board
                self.board[row, col] = NAME_TO_PIECE[x["name"]]

                # check if the game ends
                if self.check_draw():
                    content = "The game ends in a draw!"
                    self.game_end = True
                else:
                    next_player_name = (
                        NAME_BLACK if x["name"] == NAME_WHITE else NAME_WHITE
                    )
                    content = CURRENT_BOARD_PROMPT_TEMPLATE.format(
                        board=self.board2text(),
                        player=next_player_name,
                    )

        msg_host = Msg(self.name, content, role="assistant")
        self.speak(msg_host)

        # Note: we disable the image display here to avoid too many images
        img = plt.imread(board2img(self.board, 'assets/current_board.png'))
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()

        return msg_host

    def assert_valid_move(self, x: int, y: int) -> None:
        """Check if the move is valid."""
        if not (0 <= x < self.size and 0 <= y < self.size):
            try:
                raise RuntimeError(f"Invalid move: {[x, y]} out of board range.")
            except RuntimeError as e:
                 print(e)
        if not self.board[x, y] == EMPTY_PIECE:
            try:
                raise RuntimeError(
                    f"Invalid move: {[x, y]} is already "
                    f"occupied by {self.board[x, y]}.",
                )
            except RuntimeError as e:
                 print(e)

    def check_win(self, x: int, y: int, piece: str) -> bool:
        """Check if the player wins the game."""
        xline = self._check_line(self.board[x, :], piece)
        yline = self._check_line(self.board[:, y], piece)
        diag1 = self._check_line(np.diag(self.board, y - x), piece)
        diag2 = self._check_line(
            np.diag(np.fliplr(self.board), self.size - 1 - x - y),
            piece,
        )
        return xline or yline or diag1 or diag2

    def check_draw(self) -> bool:
        """Check if the game ends in a draw."""
        return np.all(self.board != EMPTY_PIECE)

    def board2text(self) -> str:
        """Convert the board to a text representation."""
        return "\n".join(
            [
                str(_)[1:-1].replace("'", "").replace(" ", "")
                for _ in self.board
            ],
        )

    def _check_line(self, line: np.ndarray, piece: str) -> bool:
        """Check if the player wins in a line."""
        count = 0
        for i in line:
            if i == piece:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0
        return False


#agents


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



def parse_func1(response: ModelResponse) -> ModelResponse:
    """Parse the response from the model into a dict with "move" and "thought"
    keys."""
    print(response.text)
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
        print("reply")
        print(response)

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

#gamestart
    
"""The main script to start a Gomoku game between two agents and a board
agent."""

#from board_agent import NAME_TO_PIECE, NAME_BLACK, NAME_WHITE, BoardAgent
#from gomoku_agent import GomokuAgent

from agentscope import msghub

import agentscope


MAX_STEPS = 30

SYS_PROMPT_TEMPLATE = """
You're a skillful Gomoku player. You should play against your opponent according to the following rules:

Game Rules:
1. This Gomoku board is a 15*15 grid. Moves are made by specifying row and column indexes, with [0, 0] marking the top-left corner and [14, 14] indicating the bottom-right corner.
2. The goal is to be the first player to form an unbroken line of your pieces horizontally, vertically, or diagonally.
3. If the board is completely filled with pieces and no player has formed a row of five, the game is declared a draw.

Note:
1. Your pieces are represented by '{}', your opponent's by '{}'. 0 represents an empty spot on the board.
2. You should think carefully about your strategy and moves, considering both your and your opponent's subsequent moves.
3. Make sure you don't place your piece on a spot that has already been occupied. like spot [5,6] is already occupied, you can't place your piece on it.
4. Only an unbroken line of five same pieces will win the game. For example, "xxxoxx" won't be considered a win.
5. Note the unbroken line can be formed in any direction: horizontal, vertical, or diagonal.
"""  # noqa

# Prepare the model configuration


YOUR_MODEL_CONFIGURATION_NAME = "qwen"
YOUR_MODEL_CONFIGURATION = {
        "model_type": "dashscope_chat",
        "config_name": YOUR_MODEL_CONFIGURATION_NAME,
        "model_name": "qwen-max",
        "api_key": "sk-ebf86b67058945fa827863a3742df0b0",
        "generate_args": {
            "temperature": 0.1
        }
}


# Initialize the agents
agentscope.init(model_configs=YOUR_MODEL_CONFIGURATION)

piece_black = NAME_TO_PIECE[NAME_BLACK]
piece_white = NAME_TO_PIECE[NAME_WHITE]

black = GomokuAgent(
    NAME_BLACK,
    model_config_name=YOUR_MODEL_CONFIGURATION_NAME,
    sys_prompt=SYS_PROMPT_TEMPLATE.format(piece_black, piece_white),
)

white = GomokuAgent(
    NAME_WHITE,
    model_config_name=YOUR_MODEL_CONFIGURATION_NAME,
    sys_prompt=SYS_PROMPT_TEMPLATE.format(piece_white, piece_black),
)

board = BoardAgent(name="Host")


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
    

i = 0
j=0
msg = None
def gomoku_send_message(chatbot, input, _state):
        global j 
        global msg
        if j == 0:
            if input == '开始':
                # 发送任何消息
                # 将发送的消息添加到聊天历史
                user_agent = _state['user_agent']
                chatbot.append(("用户输入：" + input, None))
                yield {
                    user_chatbot1: chatbot,
                    preview_chat_input1: ''
                }
                time.sleep(1)
                #msg = Msg(name="system", content="系统指挥官：开始五子棋比赛。下面有请主持人。",role="system")
                chatbot.append((None, f'系统指挥官：开始五子棋比赛。下面有请主持人。'))
                yield {
                        user_chatbot1: chatbot,
                        preview_chat_input1: '',
                    }
                time.sleep(1)
                global i
                msg = board(msg)
                chatbot.append((None, f"当前的棋局是{msg.content}"))
                yield {
                                        user_chatbot1: chatbot,
                                        preview_chat_input1: '',
                                }
                chatbot.append((None, f'系统指挥官：请用户先输入，注意只能输入如下格式[5,7]。'))
                yield {
                        user_chatbot1: chatbot,
                        preview_chat_input1: '',
                    }
                j = j + 1
                img = "assets/current_board.png"
            else:
                chatbot.append((None,'系统指挥官：请输入开始'))
                yield {
                        user_chatbot1: chatbot,
                        preview_chat_input1: '',
                    }
                img = "assets/current_board.png"
        else:
            with msghub(participants=[black, white, board]):
                if not board.game_end and i < MAX_STEPS:
                #while not board.game_end and i < MAX_STEPS:
                        #for player in [black, white]:
                #receive the move from the player, judge if the game ends and
                # remind the player to make a move
                            #msg = input
                            chatbot.append(("用户输入：" + input, None))
                            yield {
                            user_chatbot1: chatbot,
                                preview_chat_input1: ''
                            }
                            time.sleep(1)
                            #msg = black(msg)
                            #if player == black :
                            chatbot.append((None, f"Alice出手是{input}"))
                            yield {
                                            user_chatbot1: chatbot,
                                            preview_chat_input1: '',
                                    }
                            time.sleep(1)
                            if is_valid_input(input):
                                msg = Msg(name="Alice", content=input,role="black")
                                msg = board(msg)
                                print(msg.content)
                                chatbot.append((None, f"当前的棋局是{msg.content}"))
                                yield {
                                        user_chatbot1: chatbot,
                                        preview_chat_input1: '',
                                }
                                time.sleep(1)
                                if board.game_end:
                                    pass
                                else:
                                    msg = white(msg)
                                    chatbot.append((None, f"Bob出手是{msg.content}"))
                                    yield {
                                                    user_chatbot1: chatbot,
                                                    preview_chat_input1: '',
                                            }
                                    msg = board(msg)
                                    print(msg.content)
                                    chatbot.append((None, f"当前的棋局是{msg.content}"))
                                    yield {
                                            user_chatbot1: chatbot,
                                            preview_chat_input1: '',
                                    }
                                    time.sleep(1)
                                    # end the game if draw or win
                                    #if board.game_end:
                                    #    chatbot.append((None, f"{msg.content}"))
                                    #    yield {
                                    #        user_chatbot: chatbot,
                                    #        preview_chat_input: '',
                                    #    }
                                    #    break
                                    #else:
                                    i += 1
                            else:
                                chatbot.append((None , f"请回答正确的格式，如[5,7]"))
                                yield {
                                    user_chatbot1: chatbot,
                                    preview_chat_input1: '',
                                }
                    # make a move
            img = "assets/current_board.png"
            if board.game_end:
                #msg = board(msg)
                chatbot.append((None, f"比赛结束"))
                yield {
                                    user_chatbot1: chatbot,
                                    preview_chat_input1: '',
                        }
                time.sleep(1)
                j == 0
            elif i == MAX_STEPS:
                chatbot.append((None, f"比赛超时，结束"))
                yield {
                        user_chatbot1: chatbot,
                        preview_chat_input1: '',
                }
        return (user_chatbot1, preview_chat_input1,img)

y=0
def send_message(chatbot, chatsys, user_input, _state):
        # 然后使用 message 变量来发送消息
        # 将发送的消息添加到聊天历史
        global y
        host_agent = _state['host_agent']
        judge_agent = _state['judge_agent']
        judge_AI_agent = _state['judge_AI_agent']
        parti_agent = _state['parti_agent']
        user_agent = _state['user_agent']
        msg = Msg(name="user", content=user_input)
        chatbot.append((f'{msg.content}',None))
        yield {
            user_chatbot: chatbot,
            user_chatsys: chatsys,
            user_chat_input: '',
        }
        if y==0:
            if('开始' in user_input):
                        
                        msg = Msg(name="system", content="猜谜语游戏规则：请依据谜面猜出谜底。下面有请主持人出题。")
                        chatsys.append((f'{msg.content}', None))
                        yield {
                            user_chatbot: chatbot,
                            user_chatsys: chatsys,
                        }
                        host_msg = host_agent(msg)
                        chatsys.append((f"主持人：本轮的关键字是：{host_msg.content}", None))
                        yield {
                            user_chatbot: chatbot,
                            user_chatsys: chatsys,
                        }
                        global pre_host_key
                        pre_host_key = host_msg.content
                        y=y+1
            else:
                #print('请输入：开始')
                msg = Msg(name="system", content="请输入：开始。")
                chatsys.append((f'{msg.content}', None))
                yield {
                            user_chatbot: chatbot,
                            user_chatsys: chatsys,
                    }
        elif(y>0):         
            if '开始' in user_input or  '继续' in user_input:
                        msg = Msg(name="system", content="猜谜语游戏规则：请依据谜面猜出谜底。下面有请主持人出题。")
                        chatsys.append((f'{msg.content}', None))
                        yield {
                            user_chatbot: chatbot,
                            user_chatsys: chatsys,
                        }
                        host_msg = host_agent(msg)
                        chatsys.append((f"主持人：本轮的谜语是：{host_msg.content}，请输入你的答案", None))
                        yield {
                            user_chatbot: chatbot,
                            user_chatsys: chatsys,
                        }
                        pre_host_key = host_msg.content
                        y=y+1
            else:
                        
                            judge_content = f'主持人的谜语是{pre_host_key}，用户的谜底是{user_input}，请确认用户的得分，是否赢得了比赛'
                            judge_msg = judge_agent(Msg(name='judge', content=judge_content))
                            chatsys.append((None, f'评审官：{judge_msg.content}'))
                            yield {
                                user_chatbot: chatbot,
                                user_chatsys: chatsys,
                            }
                            y+=1
                            time.sleep(1)
                            if '7' not in judge_msg.content:
                                msg = Msg(name="system", content="下面请AI-Agent答题。")
                                chatsys.append((f'{msg.content}', None))
                                yield {
                                    user_chatbot: chatbot,
                                    user_chatsys: chatsys,
                                }
                                time.sleep(1)
                                parti_content = f'主持人的谜语是{pre_host_key}'
                                parti_msg = parti_agent(Msg(name='parti', content=parti_content))
                                chatbot.append((None, f'AI-Agent答题：{parti_msg.content} 我是AI界的光头强'))
                                yield {
                                        user_chatbot: chatbot,
                                        user_chatsys: chatsys,
                                    }
                                judge_AI_content = f'主持人的谜语是{pre_host_key}，AI-Agent的谜底是{parti_msg.content}，请确认AI-Agent的得分，是否赢得了比赛'
                                judge_AI_msg = judge_AI_agent(Msg(name='judge', content=judge_AI_content))
                                chatsys.append((None, f'评审官：{judge_AI_msg.content}'))
                                yield {
                                        user_chatbot: chatbot,
                                        user_chatsys: chatsys,
                                    }
                                if '7' not in judge_AI_msg.content:
                                    chatsys.append(('如果想要尝试更多的谜语，请回复「继续」', None))
                                    yield {
                                                        user_chatbot: chatbot,
                                                        user_chatsys: chatsys,
                                                    }
                                    y+=1
                                else:
                                    chatsys.append(('AI光头强获胜，请回复「开始」，重新开局', None))
                                    yield {
                                                        user_chatbot: chatbot,
                                                        user_chatsys: chatsys,
                                                    }
                                    y= -1
                            else:
                                chatsys.append(('恭喜你获胜，请回复「开始」，重新开局', None))
                                yield {
                                                        user_chatbot: chatbot,
                                                        user_chatsys: chatsys,
                                                    }
                                y = -1
                     
        else:
            judge_msg = judge_agent(Msg(name='judge', content='本轮游戏结束，请将选手得分score重新初始化为5'))
            chatsys.append(('游戏重置成功，您可以再次和AI对战，请回答「开始」', None))
            yield {
                                user_chatbot: chatbot,
                                user_chatsys: chatsys,
                            }
            y=0  
        return [[user_chatbot], [user_chatsys], [user_chat_input]]


def game_ui():
        return {tabs: gr.update(visible=False), game_tabs: gr.update(visible=True)}

def welcome_ui():
        return {tabs: gr.update(visible=True), game_tabs: gr.update(visible=False)}

image_src_1 = covert_image_to_base64('assets/wechat.png')

def updateimg():
    global j 
    if j==0:
        img = "assets/board.png"
    else:
        img = "assets/current_board.png"
    return img

# 创建 Gradio 界面
demo = gr.Blocks(css='assets/app.css')
with demo:
    warning_html_code = """
        <div class="hint" style="background-color: rgba(255, 255, 0, 0.15); padding: 10px; margin: 10px 0; border-radius: 5px; border: 1px solid #ffcc00;">
            <p>\N{fire} Powered by <a href="https://github.com/modelscope/agentscope">AgentScope</a> 正在参加黑客松，请点个小心心，联系我，发你源码，wx:16763326966</p>
        </div>
        """
    gr.HTML(warning_html_code)
    
    state = gr.State({'session_seed': uid})
    tabs = gr.Tabs(visible=False)
    with tabs:
        welcome_tab = gr.Tab('游戏介绍', id=0)
        with welcome_tab:
            user_chat_bot_cover = gr.HTML(format_welcome_html())
        with gr.Row():
            new_button = gr.Button(value='🚀开始挑战', variant='primary')
    
    game_tabs = gr.Tabs(visible=True)
    with game_tabs:
        main_tab = gr.Tab('猜谜语', id=0)
        with main_tab:
            with gr.Row():
                with gr.Column(min_width=270):
                    user_chatbot = gr.Chatbot(
                        elem_classes="app-chatbot",
                        avatar_images=[user_avatar, parti_avatar],
                        label="答题区",
                        show_label=True,
                        bubble_full_width=False,
                    )
                with gr.Column(min_width=270):
                    user_chatsys = gr.Chatbot(
                        value=[['您好，欢迎来到玩心眼之猜谜语大挑战，如果你准备好了，请回答「开始」', None]],
                        elem_classes="app-chatbot",
                        avatar_images=[host_avatar, judge_avatar],
                        label="系统栏",
                        show_label=True,
                        bubble_full_width=False,
                    )
            with gr.Row():
                with gr.Column(scale=12):
                    user_chat_input = gr.Textbox(
                        label='user_chat_input',
                        show_label=False,
                        placeholder='尽情挥洒你的才情吧')
                with gr.Column(min_width=70, scale=1):
                    send_button = gr.Button('📣发送', variant='primary')
                #with gr.Column(min_width=70, scale=1):
                #    start_button = gr.Button('📣开始', variant='primary')
            with gr.Row():
                return_welcome_button = gr.Button(value="↩️返回首页")
        sub_tab = gr.Tab('五子棋', id=1)
        with sub_tab:
                gr.Markdown('# <center> \N{fire} 和AI比心眼：玩五子棋、谁是卧底、猜谜语</center>')
                with gr.Row(elem_classes='container'):
                    with gr.Column(scale=4):
                        with gr.Column():
                            #img = gr.Image(interactive=False)  
                            user_chatbot1 = Chatbot(
                                value=[[None, '系统指挥官提示：您好，欢迎来到五子棋对战，如果你准备好了，请回答开始']],
                                elem_id='user_chatbot',
                                elem_classes=['markdown-body'],
                                avatar_images=[judge_avatar,user_avatar, host_avatar, parti_avatar],
                                height=600,
                                latex_delimiters=[],
                                show_label=False)
                        with gr.Row():
                            with gr.Column(scale=12):
                                preview_chat_input1 = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    placeholder='开始')
                            with gr.Column(min_width=70, scale=1):
                                preview_send_button1 = gr.Button('发送', variant='primary')

                    with gr.Column(scale=1):
                        img = gr.Image(value="assets/board.png",interactive=False)  
                        update_button = gr.Button(value='🔄更新棋盘局势')
                        #user_chat_bot_cover1 = gr.HTML(f'<div class="bot_cover">'
                        #                '<div class="bot_name">"五子棋"</div>'
                        #                '<div class="bot_desp">"和AI下五子棋"</div>'
                        #                '</div>')
                with gr.Row():
                    return_welcome_button1 = gr.Button(value="↩️返回首页")

    update_button.click(updateimg, inputs=[], outputs=img)

    preview_send_button1.click(
        gomoku_send_message,
        inputs=[user_chatbot1, preview_chat_input1, state],
        outputs=[user_chatbot1, preview_chat_input1,img])

    # change ui
    new_button.click(game_ui, outputs=[tabs, game_tabs])
    return_welcome_button.click(welcome_ui, outputs=[tabs, game_tabs])
    return_welcome_button1.click(welcome_ui, outputs=[tabs, game_tabs])
 
    user_chat_input.submit(
        send_message,
        inputs=[user_chatbot, user_chatsys, user_chat_input, state],
        outputs=[user_chatbot, user_chatsys, user_chat_input]
    )
    send_button.click(
         send_message,
        inputs=[user_chatbot, user_chatsys, user_chat_input, state],
        outputs=[user_chatbot, user_chatsys, user_chat_input]
    )
   
    demo.load(init_game, inputs=[state], outputs=[state])
    demo.load(gomoku_init_user, inputs=[state], outputs=[state])

#demo.queue()
demo.launch(share= False)
#demo.launch(share=True)
