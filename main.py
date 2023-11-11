from dataclasses import dataclass
import html
import re

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer
import gradio as gr

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
class GenerationResult:
    input: str
    input_ids: torch.Tensor
    input_tokens: list[str]
    output: str
    output_ids: torch.Tensor
    output_tokens: list[str]

    def __str__(self):
        return self.output

@dataclass
class GenerationInfo:
    device: torch.device
    seed: int

    def __str__(self):
        return f"""seed={self.seed}
device={self.device.type}:{self.device.index}
"""

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


def main(prompt, *args):
    (
        seed,
        max_new_tokens,
        min_new_tokens,
        early_stopping,
        do_sample,
        use_cache,
        temperature,
        top_k,
        top_p,
    ) = args
    
    
    model = get_model()
    input_ids = model.tokenizer.encode(prompt, return_tensors="pt")
    
    if seed < 0:
        seed = torch.seed()
    else:
        torch.manual_seed(seed)
    
    generate_args = {}
    
    if 0 < max_new_tokens:
        generate_args["max_new_tokens"] = int(max_new_tokens)
    if 0 < min_new_tokens:
        generate_args["min_new_tokens"] = int(min_new_tokens)
    if early_stopping in ("True", "False", "Never"):
        generate_args["early_stopping"] = {"True": True, "False": False, "Never": "Never"}[early_stopping]
    generate_args["do_sample"] = bool(do_sample)
    generate_args["use_cache"] = bool(use_cache)
    generate_args["temperature"] = float(temperature)
    if 0 < top_k:
        generate_args["top_k"] = int(top_k)
    if 0 < top_p < 1:
        generate_args["top_p"] = float(top_p)

    print(generate_args)
    output = model.model.generate(
        input_ids=input_ids.to(model.device),
        **generate_args,
        #min_new_tokens=1,
        #max_new_tokens=16,
        #do_sample=True,
        #temperature=0.8,
        #streamer=model.streamer,
        output_scores=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=True,
    )
    
    output_ids = output.sequences
    result = model.tokenizer.decode(output_ids[0,:], skip_special_tokens=False, clean_up_tokenization_spaces=True)
    
    out = GenerationResult(
        input=prompt,
        input_ids=input_ids,
        input_tokens=[model.tokenizer.batch_decode(ids) for ids in input_ids],
        output=result,
        output_ids=output_ids,
        output_tokens=[model.tokenizer.batch_decode(ids) for ids in output_ids],
    )
    info = GenerationInfo(
        device=model.model.device,
        seed=seed,
    )
    
    return out, info

def main_wrap(*args, **kwargs):
    try:
        result, info = main(*args, **kwargs)
        result_html = []
        for i, (id, token) in enumerate(zip(result.output_ids[0], result.output_tokens[0])):
            token = re.sub(r"\r?\n", '&lt;|newline|&gt;', html.escape(token))
            if token.startswith("&lt;|"):
                if token.endswith("|&gt;"):
                    token = f'<span class="special">{token}</span>'
            ele = f'<span class="token" data-token-pos="{i}" data-token-id="{id}" title="index {i}\nid {id}">{token}</span>'
            result_html.append(ele)

        header = '<label>Output<div class="output">'
        content = "".join(result_html).replace("&lt;|newline|&gt;", "&lt;|newline|&gt;<br/>")
        return [
            header + content + "</div></label>",
            result.output,
            gr.update(value=info, visible=True),
            gr.update(value="", visible=False),
        ]
    except RuntimeError as e:
        import sys, traceback
        traceback.print_exc(file=sys.stderr)
        return [
            "",
            "",
            gr.update(value="", visible=False),
            gr.update(value=str(e), visible=True),
        ]


def R(min, max, value, step=None):
    d = { "minimum": min, "maximum": max, "value": value }
    if step is not None:
        d["step"] = step
    return d


with gr.Blocks(analytics_enabled=False, css="./default.css", js="./default.js") as demo:
    with gr.Tab("Main"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=5, value="こんにちは。", placeholder="input prompt here", label="Prompt")
                seed = gr.Number(value=-1, label="seed")
                run = gr.Button(value="Run", variant="primary")
                
                with gr.Accordion(label="Params", open=False):
                    with gr.Accordion(label="Generation Config"):
                        with gr.Accordion(label="Length", open=True):
                            #max_length =     gr.Slider(**R(0, 1<<16, 0, 1), label="max_length", info="""The maximum length the generated tokens can have. Corresponds to the length of the input prompt + max_new_tokens. Its effect is overridden by max_new_tokens, if also set.""")
                            #min_length =     gr.Slider(**R(0, 1<<16, 0, 1), label="min_length", info="""The minimum length of the sequence to be generated. Corresponds to the length of the input prompt + min_new_tokens. Its effect is overridden by min_new_tokens, if also set.""")
                            max_new_tokens = gr.Slider(**R(0, 1<<13, 0, 1), label="max_new_tokens", info="""The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.""")
                            min_new_tokens = gr.Slider(**R(0, 1<<13, 0, 1), label="min_new_tokens", info="""The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.""")
                            early_stopping = gr.Radio(choices=["True", "False", "Never"], value="False", label="early_stopping", info="""Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values: True, where the generation stops as soon as there are num_beams complete candidates; False, where an heuristic is applied and the generation stops when is it very unlikely to find better candidates; "never", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).""")
                            #max_time = gr.Slider(**R(0, 180, 0, 0.5), label="max_time", info="""The maximum amount of time you allow the computation to run for in seconds. generation will still finish the current pass after allocated time has been passed.""")
                        
                        with gr.Accordion(label="Strategy", open=True):
                            do_sample = gr.Checkbox(value=True, label="do_sample", info="""Whether or not to use sampling ; use greedy decoding otherwise.""")
                            #num_beams = gr.Slider(**R(0, 32, 0, 1), label="num_beam", info="""Number of beams for beam search. 1 means no beam search.""")
                            #num_beam_groups = gr.Slider(**R(0, 32, 0, 1), label="num_beam_groups")
                            #penalty_alpha = gr.Slider(**R(-1, 2, -1, 0.01), label="penalty_alpha", info="""The values balance the model confidence and the degeneration penalty in contrastive search decoding.""")
                            use_cache = gr.Checkbox(value=True, label="use_cache")

                        with gr.Accordion(label="Logits", open=True):
                            temp = gr.Slider(**R(0, 2, 1.0, 0.01), label="temperature", info="The value used to modulate the next token probabilities.")
                            top_k = gr.Slider(**R(0, 200, 0, 1), label="top_k", info="""The number of highest probability vocabulary tokens to keep for top-k-filtering.""")
                            top_p = gr.Slider(**R(0, 1, 1, 0.01), label="top_p", info="""If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.""")
                            #typical_p = gr.Slider(**R(0, 1, 1, 0.01), label="typical_p", info="""Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to typical_p or higher are kept for generation. See this paper for more details.""")
                            #epsilon_cutoff = gr.Slider(**R(0, 1, 0, 0.01), label="epsilon_cutoff", info="""If set to float strictly between 0 and 1, only tokens with a conditional probability greater than epsilon_cutoff will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the size of the model. See Truncation Sampling as Language Model Desmoothing for more details.""")
                            #eta_cutoff =gr.Slider(**R(0, 1, 0, 0.01), label="eta_cutoff", info="""Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly between 0 and 1, a token is only considered if it is greater than either eta_cutoff or sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits))). The latter term is intuitively the expected next token probability, scaled by sqrt(eta_cutoff). In the paper, suggested values range from 3e-4 to 2e-3, depending on the size of the model. See Truncation Sampling as Language Model Desmoothing for more details.""")


            with gr.Tab("Analyze"):
                with gr.Column():
                    result = gr.HTML(label="Output", value='<label>Output<div class="output"></div></label>', elem_id="output", elem_classes="output")
                    gr.HTML(elem_id="output_token_info", elem_classes="output_token_info")
                    info = gr.Textbox(label="Info", elem_id="output_info", elem_classes="output_info", interactive=False)
                    error = gr.Textbox(label="Errors", interactive=False, visible=False)
            with gr.Tab("Original"):
                result_original = gr.Textbox(value="", label="Output", lines=50, max_lines=50, interactive=False)
    
    with gr.Tab("Attentions"):
        pass
    
    inputs = [
        prompt,
        seed,
        max_new_tokens,
        min_new_tokens,
        early_stopping,
        do_sample,
        use_cache,
        temp,
        top_k,
        top_p,
    ]
    
    run.click(main_wrap, inputs=inputs, outputs=[result, result_original, info, error])

    #ui_config = gr.CSVLogger()
    #ui_config.setup(inputs, "./")
    #run.click(lambda *args: ui_config.flag(args), inputs, None, preprocess=False)
    #demo.load(lambda *args: ui_config.flag(args), inputs, None, preprocess=False)
    #ui_config.components

if __name__ == "__main__":
    demo.launch()
