import os
import gradio as gr
os.environ["DASHSCOPE_API_KEY"] = "sk-3e99355c01a04a46ae3f3b8fcd67fe18"
print(os.environ["DASHSCOPE_API_KEY"])

def updateimg():
    img = "assets/current_board.png"
    return img

demo = gr.Blocks(css='assets/app.css')
with demo:
    img = gr.Image(value="assets/board.png", interactive=False)
    Button = gr.Button(value='开始游戏')
    gr.Markdown('## 谜语猜谜语')

    Button.click(updateimg, inputs=[], outputs=img)

demo.launch()

