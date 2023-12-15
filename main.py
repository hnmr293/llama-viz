from dataclasses import dataclass
import html
import re
import time
import math
import json

import numpy as np
import torch
import einops
import gradio as gr
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from load import reload_model_hf, get_loaded_model_or_default
from ui import ui
from utils import fix_seed

@dataclass
class GenerationResult:
    input: str
    input_ids: torch.Tensor
    input_tokens: list[str]
    output: str
    output_ids: torch.Tensor
    output_tokens: list[str]

    attentions: list
    all_hidden_states: tuple[tuple[torch.Tensor]]

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

AttnValueSelector = {
    "Mean": lambda vs: torch.mean(vs, dim=2),
    "Median": lambda vs: torch.median(vs, dim=2)[0],
    "Max": lambda vs: torch.max(vs, dim=2)[0],
    "Min": lambda vs: torch.min(vs, dim=2)[0],
    "2-Norm": lambda vs: torch.linalg.vector_norm(vs, dim=2),
}

AttnValueScaler = {
    "Linear": lambda vs: vs,
    "Log10": lambda vs: torch.log10(vs),
}

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


def main(
    model_id: str,
    cache_dir: str,
    local_only: bool,
    trust_remote_code: bool,
    prompt: str,
    *args
):
    if model_id == 'None':
        return None, None
    
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
    
    if model_id is None or len(model_id) == 0:
        raise RuntimeError("repo id is not specified")
    
    model_args = { "device_map": "auto" }
    tokenizer_args = { "device_map": "auto" }
    
    if cache_dir is not None and len(cache_dir) != 0:
        model_args["cache_dir"] = cache_dir
        tokenizer_args["cache_dir"] = cache_dir
    
    if local_only:
        model_args["local_files_only"] = True
        tokenizer_args["local_files_only"] = True

    if trust_remote_code:
        model_args["trust_remote_code"] = True
        tokenizer_args["trust_remote_code"] = True

    model_rev = model_id.rindex(":")
    model_id, model_rev = model_id[:model_rev], model_id[model_rev+1:]
    model = reload_model_hf(model_id, model_rev, model_args, tokenizer_args)
    input_ids = model.tokenizer.encode(prompt, return_tensors="pt")
    
    seed = int(seed)
    if seed < 0:
        torch.seed()
        seed = torch.randint(low=0, high=0xffff_ffff, size=(1,)).item()
    seed = seed & 0xffff_ffff
    if seed == 0xffff_ffff:
        seed = 0xffff_fffe
        # numpy needs a seed is between 0..2^32-1
    
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

    fix_seed(seed)
    print("Seed =", seed)
    
    output = model.model.generate(
        input_ids=input_ids.to(model.device),
        **generate_args,
        #streamer=model.streamer,
        output_scores=False,
        output_attentions=True,
        output_hidden_states=True,
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
        all_hidden_states=output.hidden_states,
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
        if result is None:
            return [
                "",
                "",
                "",
                "",
                gr.update(value="", visible=True),
                gr.update(value="", visible=False),
            ]

        model = get_loaded_model_or_default()
        
        # create HTML for the token-separated text
        result_html = []
        for i, (id, token) in enumerate(zip(result.output_ids[0], result.output_tokens[0])):
            if id in model.tokenizer.all_special_ids:
                token = f'<span class="special">{html.escape(token)}</span>'
            else:
                token = re.sub(r"\r?\n", '&lt;|newline|&gt;', html.escape(token))
                if token.startswith("&lt;|"):
                    if token.endswith("|&gt;"):
                        token = f'<span class="special">{token}</span>'
            klass = ["token"]
            if i < result.input_ids.size(-1):
                klass.append("input_token")
            else:
                klass.append("output_token")
            ele = f'<span class="{" ".join(klass)}" data-token-pos="{i}" data-token-id="{id}" title="index {i}\nid {id}">{token}</span>'
            result_html.append(ele)

        header = '<label>Output<div class="output">'
        content = "".join(result_html).replace("&lt;|newline|&gt;", "&lt;|newline|&gt;<br/>")
        content = header + content + "</div></label>"
        
        _last_generation_result = result
        
        return [
            content,
            result.output,
            content,
            content,
            gr.update(value=info, visible=True),
            gr.update(value="", visible=False),
        ]
    except Exception as e:
        import sys, traceback
        traceback.print_exc(file=sys.stderr)
        return [
            "",
            "",
            "",
            "",
            gr.update(value="", visible=False),
            gr.update(value=str(e), visible=True),
        ]

def attn(
    type: str,
    kind: str,
    scale_type: str,
    zmin: float, zmax: float,
    zmin_log: float, zmax_log: float,
    selected_json: str
):
    zmin = float(zmin)
    zmax = float(zmax)
    if scale_type == "Linear":
        if not math.isfinite(zmin) or not math.isfinite(zmax):
            print("invalid z_min/z_max value(s):")
            print("  z_min = ", zmin)
            print("  z_max = ", zmax)
            print("z_min = 0, z_max = 1 will be used instead.")
            zmin = 0.0
            zmax = 1.0
    else:
        if not math.isfinite(zmin_log) or not math.isfinite(zmax_log):
            print("invalid log(z)_min/log(z)_max value(s):")
            print("  log(z)_min = ", zmin_log)
            print("  log(z)_max = ", zmax_log)
            print("log(z)_min = -2, log(z)_max = 0 will be used instead.")
            zmin = -2
            zmax = 0
        else:
            zmin = zmin_log
            zmax = zmax_log

    if zmax < zmin:
        zmin, zmax = zmax, zmin
    
    result = _last_generation_result
    
    if result is None:
        return gr.update(value=None, visible=False)
    
    if selected_json is None or len(selected_json) == 0:
        return gr.update(value=None, visible=False)
    
    selected = json.loads(selected_json)
    if type == "All":
        selected = list(range(result.input_ids.size(-1), result.output_ids.size(-1)))
    elif type == "Selected":
        if len(selected) == 0:
            return gr.update(value=None, visible=False)
        selected = [int(x) for x in selected]
    else:
        return gr.update(value=None, visible=False)
    
    value_scaler = AttnValueScaler[scale_type]
    value_selector = AttnValueSelector[kind]

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
        attn_map = value_selector(attn_map) # (layers(32), context_size)
        attn_map = value_scaler(attn_map)
        #attn_map[attn_map == torch.nan] = zmin
        
        layer_count, context_size = attn_map.shape
        expanded_map = torch.zeros((max_layer_count, max_context_size), dtype=torch.float, device='cpu')
        expanded_map[:,:] = torch.nan
        expanded_map[:layer_count, :context_size] = attn_map
        
        map = go.Heatmap(
            z=expanded_map.to('cpu', dtype=torch.float),
            xgap=1, ygap=1,
            zmin=zmin, zmax=zmax,
            showscale=False,
        )
        heatmaps.append((map, layer_count, context_size, token_index, token))
    
    if len(heatmaps) == 0:
        return gr.update(value=None, visible=False)
    
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
            hovertemplate='token=%{customdata[0]} "%{customdata[1]}"<br>layer=%{y}<br>value=%{z}',
            customdata=np.dstack((
                result.output_ids.repeat((max_layer_count,1)).to("cpu"),
                np.array(result.output_tokens).repeat(max_layer_count, axis=0)
            ))
        )
    fig.update_layout(
        #**{ f"xaxis{i+1}": dict(side="top") for i in range(len(heatmaps)+1) },
        #width=W,
        height=H*len(heatmaps) + 0.05*H,
    )
    
    return gr.update(value=fig, visible=True)

def hidden_states_vectors(
    type: str,
    base_layer: int,
    only_angle: bool,
    selected_json: str,
):
    base_layer = int(base_layer)
    selected = json.loads(selected_json)
    
    result = _last_generation_result
    
    input_len = len(result.input_ids[0])
    output_ids = result.output_ids[0][input_len:]
    output_tokens = result.output_tokens[0][input_len:]
    all_hidden_states = result.all_hidden_states
    
    if type == 'All':
        selected = list(range(result.input_ids.size(-1), result.output_ids.size(-1)))
    elif type == 'Selected':
        if len(selected) == 0:
            return gr.update(value=None, visible=False)
        selected = [int(x) for x in selected]
    else:
        return gr.update(value=None, visible=False)
    
    selected = [x - input_len for x in selected]
    
    plots = []
    titles = []
    ns = []
    ts = []

    for token_index, (token_id, token, hidden_states) in enumerate(zip(output_ids, output_tokens, all_hidden_states)):
        if token_index not in selected:
            continue
        
        # [L,(1,N,4096)]
        hidden_states = torch.stack(hidden_states) # (L,1,N,4096)
        hidden_states = hidden_states.squeeze(dim=1).transpose(0,1) # (N,L,4096)
        assert hidden_states.dim() == 3, str(hidden_states.shape)
        assert hidden_states.size(1) == 33, str(hidden_states.shape)
        assert hidden_states.size(2) == 4096, str(hidden_states.shape)
        
        # 最後のベクトルだけが次のトークンの計算に使われる
        hidden_state = hidden_states[-1] # (L,4096)
        
        # 内積の計算誤差を小さくするために double にしておく
        hidden_state = hidden_state.to(dtype=torch.float64, device='cpu')
        norms = torch.linalg.vector_norm(hidden_state, ord=2, dim=1)
        ns.append(norms.numpy())
        
        hidden_state = torch.nn.functional.normalize(hidden_state)
        reg_norms = norms.div(norms[base_layer])
        thetas = torch.acos(torch.clamp(
            hidden_state @ hidden_state[base_layer],
            -1, 1
        ))

        title = f'{token} ({token_id})'
        titles.append(title)

        xs = []
        ys = []
        texts = []
        ts_ = []
        for layer, (r, t) in enumerate(zip(reg_norms, thetas)):
            if only_angle:
                r = 1.0
            x = r * torch.cos(t)
            y = r * torch.sin(t)
            xs.append(x.item())
            ys.append(y.item())
            texts.append(str(layer))
            ts_.append(t.item())
        ts.append(np.array(ts_))
        
        plot = go.Scatter(
            name=title,
            x=xs,
            y=ys,
            mode='lines+markers+text',
            text=texts,
            textposition='bottom right',
            textfont=dict(
                size=8,
            ),
        )
        plots.append(plot)
    
    fig = make_subplots(
        cols=4, rows=math.ceil(len(plots)/4),
        #subplot_titles=
        #vertical_spacing=0,
        #horizontal_spacing=0,
        #row_heights=[1,1,1,...]
    )

    for i, (plot, title, norm, theta) in enumerate(zip(plots, titles, ns, ts)):
        row = i // 4 + 1
        col = i % 4 + 1
        fig.add_trace(plot, row=row, col=col)
        
        min_x, min_y, max_x, max_y = min(plot.x), min(plot.y), max(plot.x), max(plot.y)
        mmin, mmax = min(min_x, min_y), max(max_x, max_y)
        D = mmax - mmin
        mmin, mmax = mmin-D*0.05, mmax+D*0.05

        fig.add_trace(go.Scatter(
            x=[mmin-1, mmax+1],
            y=[mmin-1, mmax+1],
            mode='lines',
            line=dict(
                color='#444444',
                width=0.5,
                dash='dot',
            ),
        ), row=row, col=col)

        fig.update_traces(
            row=row, col=col,
            hovertemplate='layer=%{text}<br>norm=%{customdata[0]:.3f}<br>theta=%{customdata[1]:.3f} [%{customdata[2]:.1f}\xb0]',
            customdata=np.stack((norm, theta, theta * 180 / np.pi), axis=-1),
        )

        fig.update_xaxes(range=[mmin, mmax], row=row, col=col, constrain='domain', title_text=title)
        fig.update_yaxes(range=[mmin, mmax], row=row, col=col, constrain='domain')
    
    fig.update_layout(
        #width=1600,
        height=(len(plots) // 4 + 1) * 400,
        margin_l=0, margin_t=0, margin_r=0, margin_b=0,
        autosize=True,
        showlegend=False,
        #yaxis=dict(
        #    scaleanchor='x',
        #    scaleratio=1,
        #),
    )

    return gr.update(value=fig, visible=True)


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    
    demo = ui(main_wrap, attn, hidden_states_vectors)
    
    t1 = time.perf_counter()
    print(f"launch: {(t1-t0)*1000:.1f}ms")
    
    demo.launch()
