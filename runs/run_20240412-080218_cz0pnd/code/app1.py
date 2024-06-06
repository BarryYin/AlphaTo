import os
import gradio as gr
from utils import check_uuid

def process_request(uid):
    #uid = check_uuid(uid)
    uid = 
    return uid

demo = gr.Blocks(css='assets/app.css')
with demo:
    
    uuid = gr.Textbox(label='modelscope_uuid', visible=False)
    with gr.Row():
         butt2 = gr.Button(value="显示uid")
         output = gr.Textbox(value="输出uid")
    butt2.click(process_request, inputs=[uuid], outputs=[output])
