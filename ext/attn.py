import math
import json

import numpy as np
import torch
import gradio as gr
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from ext.generation_result import GenerationResult


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


def attn(
    result: GenerationResult|None,
    type: str,
    kind: str,
    scale_type: str,
    zmin: float, zmax: float,
    zmin_log: float, zmax_log: float,
    selected_json: str
):
    if result is None:
        return gr.update(value=None, visible=False)
    
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
