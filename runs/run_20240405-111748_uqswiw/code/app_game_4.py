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
from functools import partial
from agentscope.agents.dict_dialog_agent import DictDialogAgent
import datetime
import random

model_configs = json.load(open('model_configs.json', 'r'))
os.environ["DASHSCOPE_API_KEY"] = "sk-ebf86b67058945fa827863a3742df0b0" 
model_configs[0]["api_key"] = "sk-ebf86b67058945fa827863a3742df0b0" 
agents = agentscope.init(
        model_configs=model_configs,
        agent_configs="agent_configs.json",
    )

uid = threading.current_thread().name
host_avatar = 'assets/host_image.png'
user_avatar = 'assets/parti_image.png'
judge_avatar = 'assets/judge_image.png'
judge_AI_avatar = 'assets/judge_image.png'
parti_avatar = 'assets/ai.png'

play_sys_prompt = f'''**背景 B (Background):** 
- 游戏是一款“谁是卧底”类型的游戏，其中每个玩家（包括AI玩家）都会收到一个词，这个词对于大多数玩家是相同的，但卧底的词略有不同。玩家需要通过描述自己的词（不直接透露这个词）来让其他人猜测，同时也要根据其他玩家的描述来找出卧底。
- 你参与这个游戏，你将收到一个词，并根据这个词来进行描述。你需要在描述时保持简洁（不超过10个字），并在听到其他玩家的描述后，尝试判断谁可能是卧底，并进行投票。

**角色 R (Role):**
- “谁是卧底”的玩家，需要具备简洁有效表达的能力，并能通过分析其他玩家的描述来推断谁可能是卧底。同时，AI应该能够根据游戏情况调整自己的策略，以避免被认定为卧底。

**目标 O (Objective):**
- 在游戏中有效地描述自己的词，同时尽量识别并投票给真正的卧底。
- 保持自己的安全，尽可能避免在游戏中被其他玩家投票出局。

**关键结果 KR (Key Result):**
1. 能在每轮游戏中提供一个既符合自己词义又不易让人直接猜出词的描述。
2. 能够根据其他玩家的描述，逻辑地分析并推断出最可能是卧底的玩家，并对其进行投票。
3. 需要在游戏过程中自我保护，避免因描述不当被误认为是卧底。

**步骤 S (Steps):**
1. 理解接收到的词及如何根据该词进行有效描述。
2. 推理系统，使其能够根据所有玩家的描述来推断谁可能是卧底。
3. 描述和投票，确保你在游戏中的行为既自然又具有竞争力。'''


prompt_juge = f'''
    你是一个出题机器人，你需要按照品类，随机地给出两个有很多共同之处又不太一样地词语，例如水果类里，苹果和梨子，他们都是水果但不太相同；动物类里，鲸鱼和鲤鱼，他们都是鱼但也不太相同。
    请你通过json的格式返回。加上词语1，词语2的形式返回。
    返回内容是一个字典{"{"}"词语1":str, "词语2":str{"}"}。
    示范：{"{"}'词语1': '足球', '词语2': '篮球'{"}"}
    '''


play1_agent = DictDialogAgent(
    name="play1",
    model_config_name="qwen",
    #model_config_name="qwen",
    use_memory=True,
    sys_prompt=play_sys_prompt,
)
play2_agent = DictDialogAgent(
    name="play2",
    model_config_name="qwen",
    #model_config_name="qwen",
    use_memory=True,
    sys_prompt=play_sys_prompt,
)
play3_agent = DictDialogAgent(
    name="play3",
    model_config_name="qwen",
    #model_config_name="qwen",
    use_memory=True,
    sys_prompt=play_sys_prompt,
)

juge_agent = DictDialogAgent(
    name="juge",
    model_config_name="qwen",
    use_memory=True,
    sys_prompt=prompt_juge,
)


HostMsg = partial(Msg, name="Moderator", role="assistant", echo=True)

def init_state(state):
        state["in_game"]=False
        state["step"]=0
        state["current_round"]=0
        state["alive_flag"]=[1, 1, 1, 1]
        state["player_words"]=["", "", "", ""]
        state["player1_history"]=[]
        state["player2_history"]=[]
        state["player3_history"]=[]
        state["player4_history"]=[]
        return state

#游戏控制，开局或者重启函数
def fn_start_or_restart(user_chatbot,state):
    # 初始化state
    #state = init_state()  #创建全局变量
    state["in_game"] = True
    categories = ["运动类", "电器类", "家居类", "蔬菜类","动物类","水果类"]
    content1 = random.choice(categories)
    hint = HostMsg(content=content1)
    #hint = Msg(name="system",content=content1,role="assistant")
    juge_resp = juge_agent(hint)
    json_str = json.dumps(juge_resp.content)  #获得字典，这里可以拆分为独立函数，只要能记住词就行
    json_dict = json.loads(json_str)
    word1 = json_dict["词语1"]
    word2 = json_dict["词语2"]
    index = random.choice([0, 1, 2, 3])  #随机分配每个人物的词语
    state["player_words"] = [word1, word1, word1, word1]
    state["player_words"][index] = word2
    state["step"] = 0  #开始游戏
    #print(state)
    print(state["player_words"][index])
    user_chatbot.append((f"你的词语是 `{state['player_words'][0]}`，请进行发言",None))
    return state, user_chatbot,f"你的词语是 `{state['player_words'][0]}`，请进行发言", "", "", "", "", "未选择", "", "", "", "Source/four.jpg"


def parse_func_wodi(response):
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


def fn_speek(state, prompt,user_chatbot,):
    system_text = ""
    player_text = [str(prompt), "", "", ""]  #玩家的发言
    play_agents = {
        1: play1_agent,
        2: play2_agent,
        3: play3_agent,
    }
    if state["in_game"] == False:
        system_text = "游戏未开始，请点击重新开始/开始按钮进行游戏"
        user_chatbot.append((f"游戏未开始，请点击重新开始/开始按钮进行游戏",None))
    elif state["step"] == 1:
        system_text = f"你的词语是 `{state['player_words'][0]}`，请先投票"
        user_chatbot.append((f"你的词语是 `{state['player_words'][0]}`，请先投票",None))
    else:
        user_chatbot.append((f"你的词语是 `{state['player_words'][0]}`",None))
        user_chatbot.append((None,f"你的发言是{player_text[0]}"))
        state["step"] = 1
        #按顺序发言
        for i in range(1,4):
            # 检查该玩家是否存活
            if state["alive_flag"][i] == 0:
                player_text[i] = "已死亡，无发言"
            # 如果该玩家存活，则进行发言
            else:
                #设定prompt
                prompt1 = f'''
                    你是一个描述机器人，你需要说出你收到的\<词语>的一个特征，并且字数越短越好，通常10个字以内，你给出任意一个特征都可以，比如大致的形状、功效、使用特征等，当你在进行表达的时候，你需要参考上一个玩家的发言，尽量与他有一些不同。
                    请尽量不要表述过多，以适当地隐藏自己的\<词语>。
                    此外，你也可以只说一个词，只要能够形容这个\<词语>即可。
                    更重要的是，绝对不能直接说出自己的词语。
                    你的身份是玩家{i}，当前的时间是{datetime.date.today()}, 你上一位玩家的发言内容是{player_text[i-1]}。
                    你收到的词语是:{state['player_words'][i]}
                    请你通过json的格式返回。
                    返回内容是一个字典{"{"}"特征":str{"}"}。
                    '''
                hint = HostMsg(content=prompt1)
                play_resp = play_agents[i](hint)
                print("5678")
                json_str = json.dumps(play_resp.content)  #获得字典，这里可以拆分为独立函数，只要能记住词就行
                json_dict = parse_func_wodi(json_str)
                player_text[i] = json_dict["特征"]  #获得特征  这里可以改为bot，就像玩家一样
                print(player_text[i])
                user_chatbot.append((f"{i+1}号玩家的发言是{player_text[i]}",None))
        state["player1_history"].append(player_text[0])
        state["player2_history"].append(player_text[1])
        state["player3_history"].append(player_text[2])
        state["player4_history"].append(player_text[3])
        system_text = f"你的词语是 `{state['player_words'][0]}`，请根据其他玩家的发言进行投票"
        user_chatbot.append((f"你的词语是 `{state['player_words'][0]}`，请根据其他玩家的发言进行投票",None))

    return (state, user_chatbot,system_text,
            player_text[1], player_text[2], player_text[3])

def fn_vote(state, vote_prompt,user_chatbot):
    
    system_text = ""
    player_vote = ["", "", "", ""]  #玩家的投票
    img = "Source/start.jpg"
    play_agents = {
        1: play1_agent,
        2: play2_agent,
        3: play3_agent,
    }
    user_chatbot.append((None,f"你的投票的玩家是 `{ vote_prompt}`"))

    # 统计来自玩家的票
    vote_count = [0, 0, 0, 0]
    for i in range(1, 4):
        if f"Player {str(i+1)}" == vote_prompt: # 
            vote_count[i] += 1
    if state["in_game"] == False:
        system_text = "游戏未开始，请点击重新开始/开始按钮进行游戏"
    elif state["step"] == 0:
        system_text = f"你的词语是 `{state['player_words'][0]}`，请先进行表述"
    else:
        state["step"] = 0

        state["current_round"] += 1  
        for i in range(1,4):
            # 检查该玩家是否存活
            if state["alive_flag"][i] == 0:
                player_vote[i] = "已死亡，无投票"
            else:
                other_info = ""
                alive_number = ""    
                for j in range(4):
                    if state["alive_flag"][j] == 1:
                        other_info += f"玩家{j}的发言内容是{state[f'player{j+1}_history'][-1]}；"
                        alive_number += f"{str(i)},"
                #设定prompt
                prompt2 = f'''
                现在进行投票，你需要根据你收到的\<词语>，以及其他玩家的发言，选择一个和你的差距最大的玩家。
                你收到的词语是:{state['player_words'][i]};
                其他玩家的发言内容是{other_info}。
                其中，投票结果应当为{alive_number}中的数字，数字对应相应的玩家。
                请你通过json的格式返回。
                返回内容是一个字典{"{"}"投票结果":int{"}"}。
                '''
                #投票，参与者判断谁是卧底
                hint = HostMsg(content=prompt2)
                play_resp = play_agents[i](hint)
                json_str = json.dumps(play_resp.content)  #获得字典，这里可以拆分为独立函数，只要能记住词就行
                json_dict = json.loads(json_str)
                vote_idx = int(json_dict["投票结果"])
                vote_outcome = f"Player {vote_idx+1}"
                user_chatbot.append((f'{i+1}号玩家的投票是 {vote_outcome}',None))
                print(vote_idx)
                if vote_idx <= 3:
                    vote_count[vote_idx] += 1
                    player_vote[i] = f"卧底是Player {vote_idx+1}" 
        max_vote = max(vote_count) #总结当前最大的票数
        if vote_count.count(max_vote) == 1:  #判断是否平局，大家都没有出局
            out_index = vote_count.index(max_vote) 
            state["alive_flag"][out_index] = 0   #该玩家出局
            remain_words_list = [state["player_words"][i] for i in range(0,4) if state["alive_flag"][i] == 1] #剩余的词语列表
            if len(list(set(remain_words_list))) == 2 and len(remain_words_list) <= 2: #如果只有两个词语，则卧底获胜
                state["in_game"] = False
                system_text = f"本局游戏结束，卧底获胜。几个玩家的词语分别是是{state['player_words']}。\n请点击重新开始"
                user_chatbot.append((system_text,None))
                img = "Source/loss.jpg"
            elif len(list(set(remain_words_list))) == 1: #如果只有一个词语，则非卧底获胜
                state["in_game"] = False
                system_text = f"本局游戏结束，非卧底获胜。几个玩家的词语分别是是{state['player_words']}。\n请点击重新开始" 
                user_chatbot.append((system_text,None))
                img = "Source/win.jpg"
            else:
                system_text = f"本局游戏出局的是Player {out_index+1}, 你的词语是 `{state['player_words'][0]}`，请进行发言"
                user_chatbot.append((system_text,None))
                if sum(state["alive_flag"]) == 4:
                    img = "Source/four.jpg"
                elif sum(state["alive_flag"]) == 3:
                    img = "Source/three.jpg"
        else:
            system_text = f"本局没有人出局，请继续游戏。\n你的词语是 `{state['player_words'][0]}`，请进行发言"
            user_chatbot.append((system_text,None))
            if sum(state["alive_flag"]) == 4:
                img = "Source/four.jpg"
            elif sum(state["alive_flag"]) == 3:
                img = "Source/three.jpg"

    return (state, user_chatbot,system_text,
            "", "", "", "",
            player_vote[1], player_vote[2], player_vote[3],
            img)

title_text = """
    # 谁是卧底AI版 - 和AI比心眼：玩五子棋、谁是卧底、猜谜语
    谁是卧底是一个聚会推理游戏，你和你的朋友们会各自收到一张卡片，其中有一个人的卡片和其他人都不一样。
    例如你的卡片是“苹果”，其他人的卡片都是“香蕉”。则你是卧底，你们需要对各自卡片上的内容进行描述，并根据大家的描述，找出那个卧底。
    请注意，卧底的目标是不被发现，而其他人的目标是找出卧底。
    如果第一轮没有找到卧底，则游戏继续，并且淘汰的人无法继续游戏。
    直到仅剩两个人，则卧底获胜，否则其他玩家获胜。
    本次游戏借鉴了百度星河社区的谁是卧底游戏，但我们对游戏进行了改进，调整了Prompt的设定，适应LLM的多样性；用agentscop框架，实现multi-agent applications的核心功能。
    """



def init_game(state):
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
        main_tab = gr.Tab('谁是卧底', id=0)
        with main_tab:
            gr.Markdown(value=title_text)
            #state = gr.State(value=init_state())
            # 开始游戏按钮
            with gr.Row():
              start_btn = gr.Button(value="重置/开始游戏")
            with gr.Row():
            # 一张图片
                with gr.Column(min_width=270):
                        wodi_user_chatbot = gr.Chatbot(
                            height=500,
                            value=[['您好，欢迎来到谁是卧底游戏，如果你准备好了，请点击顶部「开始」按钮', None]],
                            elem_classes="app-chatbot",
                            avatar_images=[user_avatar, parti_avatar],
                            label="游戏进程观察区",
                            show_label=True,
                            bubble_full_width=False,
                        )
                        img = gr.Image(value="Source/start.jpg", interactive=False)
                with gr.Column():
                    system_text = gr.Textbox(label="系统信息展示框", value="点击 重置/开始游戏 进行游戏")
                    with gr.Row():
                        with gr.Column():
                            # 四个文本输入框，分别代表四个玩家，其中第一个玩家为当前玩家，其他玩家均为Agent
                            player1_text = gr.Textbox(label="Player 1 发言框",
                                                    info="您的序号为Player 1，请根据您的单词进行描述，当你被投票出局之后，"
                                                        "你可以继续观战，但是你的发言将无效")
                            player2_text = gr.Textbox(label="Player 2 发言框", interactive=False)
                            player3_text = gr.Textbox(label="Player 3 发言框", interactive=False)
                            player4_text = gr.Textbox(label="Player 4 发言框", interactive=False)
                            speek_btn = gr.Button(value="确定发言(即使你被淘汰了，你可以继续观战，看看自己一方到底赢了没)")
                        with gr.Column():
                            # 四个文本输入框，分别代表四个玩家，其中第一个玩家为当前玩家，其他玩家均为Agent
                            player1_vote = gr.Dropdown(label="Player 1 指认框",
                                                    info="请选择你认为是间谍的人，注意不要指认自己，当你被投票出局之后，"
                                                            "你可以继续观战，但是你的投票将无效",
                                                    choices=["未选择" ,"Player 2", "Player 3", "Player 4"],
                                                    value="未选择",
                                                    interactive=True)
                            player2_vote = gr.Textbox(label="Player 2 指认框", interactive=False)
                            player3_vote = gr.Textbox(label="Player 3 指认框", interactive=False)
                            player4_vote = gr.Textbox(label="Player 4 指认框", interactive=False)
                            vote_btn = gr.Button(value="确认投票(即使你被淘汰了，你可以继续观战，看看自己一方到底赢了没)")
        main_tab2 = gr.Tab('猜谜语', id=1)
        with main_tab2:
            gr.Markdown('# <center> \N{fire} 和AI比心眼：玩五子棋、谁是卧底、猜谜语</center>')
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
        sub_tab = gr.Tab('五子棋', id=2)
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
                        gomoku_img = gr.Image(value="assets/board.png",interactive=False)  
                        update_button = gr.Button(value='🔄更新棋盘局势')
                        #user_chat_bot_cover1 = gr.HTML(f'<div class="bot_cover">'
                        #                '<div class="bot_name">"五子棋"</div>'
                        #                '<div class="bot_desp">"和AI下五子棋"</div>'
                        #                '</div>')
                with gr.Row():
                    return_welcome_button1 = gr.Button(value="↩️返回首页")

    update_button.click(updateimg, inputs=[], outputs=gomoku_img)

    preview_send_button1.click(
        gomoku_send_message,
        inputs=[user_chatbot1, preview_chat_input1, state],
        outputs=[user_chatbot1, preview_chat_input1,gomoku_img])

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
   
    start_btn.click(fn=fn_start_or_restart,
                    inputs=[wodi_user_chatbot,state],
                    outputs=[state,
                             wodi_user_chatbot,
                             system_text,
                             player1_text, player2_text, player3_text, player4_text,
                             player1_vote, player2_vote, player3_vote, player4_vote,
                             img])
    speek_btn.click(fn=fn_speek,
                    inputs=[state, player1_text,wodi_user_chatbot],
                    outputs=[state, wodi_user_chatbot, system_text, player2_text, player3_text, player4_text])
    vote_btn.click(fn=fn_vote,
                   inputs=[state, player1_vote,wodi_user_chatbot],
                   outputs=[state,
                            wodi_user_chatbot,
                            system_text,
                            player1_text, player2_text, player3_text, player4_text,
                            player2_vote, player3_vote, player4_vote,
                            img])
    
    demo.load(init_state,inputs=[state], outputs=[state])
    demo.load(init_game, inputs=[state], outputs=[state])
    demo.load(gomoku_init_user, inputs=[state], outputs=[state])
    


if __name__ == '__main__':
    #demo.launch()
#demo.queue()
    demo.launch(share= False)
#demo.launch(share=True)
