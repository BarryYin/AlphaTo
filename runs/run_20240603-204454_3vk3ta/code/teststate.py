import gradio as gr
import random
 
demo = gr.Blocks(css="""#btn {color: red} .abc {font-family: "Comic Sans MS", "Comic Sans", cursive !important}""")
 
with demo:
    default_json = {"a": "a"}
 
    num = gr.State(value=0)
    squared = gr.Number(value=0)
    btn = gr.Button("Next Square", elem_id="btn", elem_classes=["abc", "def"])
    draw_seed = random.randint(0, 1000000000)
    stats = gr.State({})
    stats.session_seed = draw_seed
    
    #stats = gr.State(value=default_json)
    table = gr.JSON()
 
    def increase(var, stats_history):
        var += 1
        stats_history[str(var)] = var**2
        print(stats.session_seed)
        return var, var**2, stats_history, stats_history
 
    btn.click(increase, [num, stats], [num, squared, stats, table])
 
if __name__ == "__main__":
    demo.launch()