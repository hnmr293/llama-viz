from dataclasses import dataclass
import html
import re

import torch
import gradio as gr

from load import get_model
from ui import ui

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

    print("-"*80)
    print("[Generate]")
    print("Prompt =", prompt)
    print("Args =", generate_args)
    
    output = model.model.generate(
        input_ids=input_ids.to(model.device),
        **generate_args,
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



if __name__ == "__main__":
    demo = ui(main_wrap)
    demo.launch()
