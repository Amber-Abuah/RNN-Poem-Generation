import gradio as gr
from PoemGeneration import PreloadedRNNModel

model = PreloadedRNNModel()

def generate_poem(temperature, output_length):
  return model.generate_text(temperature, output_length)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            temperature_slider = gr.Slider(0.01, 1, label="Temperature", value=0.5)
            output_length = gr.Number(value=700, label="Output Length")
            start_button = gr.Button(variant="primary")
            examples = gr.Examples([0.2, 0.4, 0.6, 0.8, 1], temperature_slider)
        with gr.Column():
            output_text = gr.Text(label="Generated text", interactive=False)

    start_button.click(fn=generate_poem, inputs=[temperature_slider, output_length], outputs=output_text)

demo.launch()