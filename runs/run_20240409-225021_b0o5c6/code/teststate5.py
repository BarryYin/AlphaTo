import uuid
import gradio as gr

uids =[]
def greet(name):
    uid = uuid.uuid4()  # 生成一个新的UUID
    uids.append(uid)  # 将UUID添加到列表中
    print(uids)
    return f"Hello {name}, your unique ID is {uid}"

iface = gr.Interface(fn=greet, inputs="text", outputs="text")
iface.launch()