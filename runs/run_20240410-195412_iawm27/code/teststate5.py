import uuid
import gradio as gr
import threading

lock = threading.Lock()
s1 = {
    "12312312" :"这是12312312",
    "5656565" :"这是5656565",
}
def greet(input):
   # uid = uuid.uuid4()  # 生成一个新的UUID

    # 创建一个新的线程来处理请求
    uid = str(uuid.uuid4())  # 生成一个新的UUID
    thread = threading.Thread(target=process_request, args=(input,), name=uid)
    thread.start()

    #return f"Hello {name}, your unique ID is {uid}"

def process_request(input):
    print(input)
    # 在这里处理请求...
    with lock:
        uid = threading.current_thread().name
        print(f"Processing request forthreading with UID {uid}")
        print(f"Processing request forthreading with UID {input}")
        print(f"Processing request forthreading with UID {input}")
        ti = f"{s1[input]}"
        return ti

demo = gr.Blocks(css='assets/app.css')
with demo:
    state = gr.State()
    uidinput = gr.Textbox(label="modelscope_uuid", visible=True)
    butt1 = gr.Button(value="新的线程")
    butt2 = gr.Button(value="显示线程名字")
    output = gr.Textbox(value="输出")
    butt1.click(greet, inputs=[uidinput], outputs=[])
    butt2.click(process_request, inputs=[uidinput], outputs=[output])
#
#demo = gr.Interface(fn=greet, inputs="text", outputs="text")

  
    demo.queue(default_concurrency_limit=None)

   
if __name__ == "__main__":
    demo.launch(max_threads=400)