# todo
# hovertemplate
# repo_id選択

from dataclasses import dataclass
import html
import re
import time

import numpy as np
import torch
import einops
import gradio as gr
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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

    attentions: list

    def __str__(self):
        return self.output

@dataclass
class GenerationInfo:
    device: torch.device
    seed: int
    time: int
    time_per_token: float

    def __str__(self):
        return f"""seed={self.seed:d}
device={self.device.type}:{self.device.index}
time={self.time/1000000:.1f}ms
     {self.time_per_token/1000000:.1f}ms/token ({1000000000/self.time_per_token:.1f}tokens/s)
"""



_last_generation_result: GenerationResult|None = None

def retrieve_attentions(attentions: tuple[tuple[torch.Tensor]]):
    # attensions:
    #   Tuple (one element for each generated token) of 
    #   tuples (one element for each layer of the decoder) of 
    #   torch.FloatTensor of shape (num_return_sequences*batch_size, num_heads, generated_length, sequence_length).
    
    # N = input_ids.size(-1)
    # n = output.sequences.size(-1)
    # attentions = (
    #     ( (1,32,N,N), ..., (1,32,N,N) ), # 32 elements for token[0]
    #     ( (1,32,1,N+1), ..., (1,32,1,N+1) ), # 32 elements for token[1]
    #     ...,
    #     ( (1,32,1,N+n-1), ..., (1,32,1,N+n-1) ), # 32 elements for token[n-1]
    # )

    assert attentions[0][0].size(0) == 1
    
    attentions = [
        einops.rearrange(
            torch.stack(token),       # (32',1,32,m,n) where m is the number of generating tokens, n is context size (input + generated tokens)
            "a b h m n -> m b a n h", # (m,1,32',n,32)
        ).squeeze(1)                  # (m,32',n,32)
        for token in attentions
    ]
    
    return attentions


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
    
    # 
    # Generate!
    # 
    t0 = time.perf_counter_ns()

    output = model.model.generate(
        input_ids=input_ids.to(model.device),
        **generate_args,
        #streamer=model.streamer,
        output_scores=False,
        output_attentions=True,
        output_hidden_states=False,
        return_dict_in_generate=True,
    )

    t1 = time.perf_counter_ns()

    print(f"Finished! [{(t1-t0)/1000000:.2f}ms]")
    
    output_ids = output.sequences
    result = model.tokenizer.decode(output_ids[0,:], skip_special_tokens=False, clean_up_tokenization_spaces=True)
    
    out = GenerationResult(
        input=prompt,
        input_ids=input_ids,
        input_tokens=[model.tokenizer.batch_decode(ids) for ids in input_ids],
        output=result,
        output_ids=output_ids,
        output_tokens=[model.tokenizer.batch_decode(ids) for ids in output_ids],
        attentions=retrieve_attentions(output.attentions),
    )
    info = GenerationInfo(
        device=model.model.device,
        seed=seed,
        time=t1-t0,
        time_per_token=(t1-t0)/(output_ids.numel() - input_ids.numel())
    )
    
    return out, info

def main_wrap(*args, **kwargs):
    global _last_generation_result
    
    try:
        result, info = main(*args, **kwargs)
        
        # create HTML for the token-separated text
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
        content = header + content + "</div></label>"
        
        _last_generation_result = result
        
        return [
            content,
            result.output,
            content,
            gr.update(value=info, visible=True),
            gr.update(value="", visible=False),
        ]
    except RuntimeError as e:
        import sys, traceback
        traceback.print_exc(file=sys.stderr)
        return [
            "",
            "",
            "",
            gr.update(value="", visible=False),
            gr.update(value=str(e), visible=True),
        ]

def attn(args: tuple):
    type, *selected = args
    result = _last_generation_result
    
    if result is None:
        return gr.update(value=None, visible=False)
    
    if type == "All":
        selected = list(range(result.input_ids.size(-1), result.output_ids.size(-1)))
    elif type == "Selected":
        if len(selected) == 0:
            return gr.update(value=None, visible=False)
        selected = [int(x) for x in selected]
    else:
        return gr.update(value=None, visible=False)

    # create heapmaps
    heatmaps = []
    
    max_layer_count = max([x.size(1) for x in result.attentions])
    max_context_size = max([x.size(2) for x in result.attentions])
    
    for token_index, attn_map in enumerate(result.attentions, start=result.input_ids.size(-1)):
        if token_index not in selected:
            continue
        
        # attn_map := (generated_tokens, layers(32), context_size, head)
        
        token = result.output_tokens[0][token_index]
        attn_map = attn_map[-1] # -> (layers(32), context_size, head)
        attn_map = torch.mean(attn_map, dim=2) # (layers(32), context_size)
        layer_count, context_size = attn_map.shape
        expanded_map = torch.zeros((max_layer_count, max_context_size), dtype=torch.float, device='cpu')
        expanded_map[:,:] = torch.nan
        expanded_map[:layer_count, :context_size] = attn_map
        map = go.Heatmap(
            z=expanded_map.to('cpu', dtype=torch.float),
            xgap=1, ygap=1,
            zmin=0, zmax=1,
            showscale=False,
        )
        heatmaps.append((map, layer_count, context_size, token_index, token))
    
    if len(heatmaps) == 0:
        return gr.update(value=None, visible=False)
    
    W = 800
    H = 400
    space = 0.2 / len(heatmaps)
    fig = make_subplots(
        rows=len(heatmaps)+1,
        cols=1,
        subplot_titles=[""] + [
            f'token{t[-2]} "{t[-1]}"' for t in heatmaps
        ],
        vertical_spacing=space,
        row_heights=[0.05] + [1] * len(heatmaps),
    )
    for index, (map, layer_count, ctx_size, _, _) in enumerate(heatmaps):
        row = index + 2
        fig.add_trace(map, row=row, col=1)
        # 長すぎると見づらいので最大3文字にしておく
        xticks_long = [x for x in result.output_tokens[0][:ctx_size]] + [""] * (max_context_size - ctx_size)
        xticks = [x[:3] for x in xticks_long]
        fig.update_xaxes(
            row=row, col=1,
            tickvals=list(range(max_context_size)),
            ticktext=xticks,
        )
        fig.update_yaxes(
            row=row, col=1,
            tickvals=list(range(max_layer_count)),
            ticktext=[str(x) for x in range(layer_count)] + [""] * (max_layer_count - layer_count),
            autorange="reversed",
        )
        fig.update_traces(
            row=row, col=1,
            hovertemplate='layer=%{y}<br>value=%{z}',
            #hovertemplate='token=%{customdata[0]} "%{customdata[1]}"<br>layer=%{y}<br>value=%{z}',
            #customdata=np.dstack((
            #    result.output_ids.to("cpu").reshape((1,1,-1)).repeat((1,max_layer_count,1)),
            #    np.array(result.output_tokens).reshape((1,1,-1)).repeat(max_layer_count, axis=1)
            #))

        )
    fig.update_layout(
        #**{ f"xaxis{i+1}": dict(side="top") for i in range(len(heatmaps)+1) },
        #width=W,
        height=H*len(heatmaps) + 0.05*H,
    )
    
    return gr.update(value=fig, visible=True)



if __name__ == "__main__":
    demo = ui(main_wrap, attn)
    demo.launch()
