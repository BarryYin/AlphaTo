import gradio as gr
import random
import uuid

def increase(var, stats):
        var += 1
        #stats['session_seed'] = str(uuid.uuid4())
        print(stats['session_seed'])
        return var, var**2, stats['session_seed'], stats
demo = gr.Blocks(css="""#btn {color: red} .abc {font-family: "Comic Sans MS", "Comic Sans", cursive !important}""")

with demo:
    default_json = {"a": "a"}
    num = gr.State(value=0)
    squared = gr.Number(value=0)
    btn = gr.Button("Next Square", elem_id="btn", elem_classes=["abc", "def"])
    stats = gr.State(value={'session_seed': str(uuid.uuid4())})
    table = gr.JSON()
    btn.click(increase, [num, stats], [num, squared, stats, stats])

if __name__ == "__main__":
    demo.launch()