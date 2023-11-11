from dataclasses import dataclass

import gradio as gr
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer

from load import load_model

@dataclass
class Model:
    repo_id: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    streamer: TextStreamer
    
    @property
    def device(self):
        return self.model.device

@dataclass
class GenerationInfo:
    device: str

loaded_model: Model|None = None



def reload_model(repo_id: str, model_args: dict, tokenizer_args: dict):
    global loaded_model
    if loaded_model is None or loaded_model.repo_id != repo_id:
        model, tokenizer = load_model(repo_id, model_args, tokenizer_args)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        loaded_model = Model(repo_id, model, tokenizer, streamer)

def reload_default_model():
    repo_id = "TheBloke/calm2-7B-chat-GPTQ"
    model_args = {
        "use_cache": True,
        "device_map": "auto",
        "revision": "gptq-4bit-32g-actorder_True",
    }
    tokenizer_args = {
        "revision": "gptq-4bit-32g-actorder_True",
    }
    reload_model(repo_id, model_args, tokenizer_args)

def get_model() -> Model:
    if loaded_model is None:
        reload_default_model()
    return loaded_model


def main(prompt):
    model = get_model()
    input_ids = model.tokenizer.encode(prompt, return_tensors="pt")
    
    output = model.model.generate(
        input_ids=input_ids.to(model.device),
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.8,
        #streamer=model.streamer,
        output_scores=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=True,
    )
    
    output_ids = output.sequences
    result = model.tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    info = GenerationInfo(model.model.device)
    
    return result, info

def main_wrap(*args, **kwargs):
    try:
        result, info = main(*args, **kwargs)
        return [
            result,
            gr.update(value=info, visible=True),
            gr.update(value="", visible=False),
        ]
    except RuntimeError as e:
        return [
            "",
            gr.update(value="", visible=False),
            gr.update(value=str(e), visible=True),
        ]

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(lines=5, value="こんにちは。", placeholder="input prompt here", label="Prompt")
            run = gr.Button(value="Run")
        with gr.Column():
            result = gr.Textbox(lines=5, label="Output", interactive=False)
            info = gr.Textbox(label="Info", interactive=False)
            error = gr.Textbox(label="Errors", interactive=False, visible=False)
    run.click(main, inputs=[prompt], outputs=[result, info, error])

if __name__ == "__main__":
    demo.launch()
