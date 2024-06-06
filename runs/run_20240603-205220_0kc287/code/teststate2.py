import gradio as gr
import random
import uuid

outcome = {
    "12312312" : 0,
    "5656565" : 0,
}

def generate_uid():
    uid = str(uuid.uuid4())
    outcome[uid] = 0
    print(f"Generated new UID: {uid}")
    for key, value in outcome.items():
        print(f'outcome["{key}"] = {value}')
    return uid
def increase(var, user_id, stats):
    var += 1
    outcome[user_id] = var
    print(f"Processing request for user {user_id} with session seed {stats['session_seed']}")
    for key, value in outcome.items():
        print(f'outcome["{key}"] = {value}')
    return var, var**2, stats['session_seed'], stats, outcome[user_id]

demo = gr.Blocks(css="""#btn {color: red} .abc {font-family: "Comic Sans MS", "Comic Sans", cursive !important}""")

with demo:
    default_json = {"a": "a"}
    uid_btn = gr.Button("Generate UID")  # 添加一个按钮用于生成新的 UID
    num = gr.State(value=0)
    squared = gr.Number(value=0)
    btn = gr.Button("Next Square", elem_id="btn", elem_classes=["abc", "def"])
    user_id = gr.Textbox()  # 添加一个输入框让用户输入他们的序号
    output = gr.Textbox(value="输出")
    stats = gr.State(value={'session_seed': None})
    table = gr.JSON()
    uid_btn.click(generate_uid, [], [user_id])  # 当用户点击 uid_btn 时，调用 generate_uid 函数，并将结果显示在 user_id 输入框中
    btn.click(increase, [num, user_id, stats], [num, squared, stats, stats,output])  # 将用户的序号传递给 increase 函数

if __name__ == "__main__":
    demo.launch()