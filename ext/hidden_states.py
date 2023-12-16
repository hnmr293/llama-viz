import math
import json

import numpy as np
import torch
import gradio as gr
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from ext.generation_result import GenerationResult


def each_token(output_ids, output_tokens, all_hidden_states, selected, base_layer):
    for token_index, (token_id, token, hidden_states) in enumerate(zip(output_ids, output_tokens, all_hidden_states)):
        # hidden_states := [L,(1,N,4096)]

        if token_index not in selected:
            continue
        
        hidden_states = torch.stack(hidden_states) # (L,1,N,4096)
        hidden_states = hidden_states.squeeze(dim=1).transpose(0,1) # (N,L,4096)
        assert hidden_states.dim() == 3, str(hidden_states.shape)
        #assert hidden_states.size(1) == 33, str(hidden_states.shape)
        #assert hidden_states.size(2) == 4096, str(hidden_states.shape)
        
        # 最後のベクトルだけが次のトークンの計算に使われる
        hidden_state = hidden_states[-1] # (L,4096)
        
        title = f'{token} ({token_id})'
        yield title, hidden_state, hidden_state[[base_layer],:]


def each_layer(output_ids, output_token, all_hidden_states, selected, base_token_index):
    all_hidden_states = [
        torch.stack([y[:,-1,:] for y in x])
        for x in all_hidden_states
        # x := [L,(1,N,d)]
        # y := (1,N,d)
    ]
    # [n,(L,1,1,d)]

    all_hidden_states = torch.stack(all_hidden_states) # (n,L,1,1,d)
    all_hidden_states = all_hidden_states.squeeze(dim=(2,3)) # (n,L,d)
    all_hidden_states = all_hidden_states.transpose(0, 1) # (L,n,d)

    for i, hidden_states in enumerate(all_hidden_states):
        # (n,d)
        title = f'layer {i}'
        yield title, hidden_states[selected, :], hidden_states[[base_token_index], :]


def hidden_states(
    result: GenerationResult|None,
    mode: str,
    type: str,
    base_layer: int,
    norm_type: str,
    angle_type: str,
    selected_json: str,
):
    if result is None:
        return gr.update(value=None, visible=False)

    base_layer = int(base_layer)
    selected = json.loads(selected_json)
    
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

    # all_hidden_states := [n,L,(1,N,d)]
    #   where
    #     n := generated tokens
    #     L := embedding + layer outputs
    #     1 := batch size
    #     N := sequence length (typically 1, except for the 0-th element)
    #     d := hidden size
    #  []: tuple
    #  (): tensor
    
    if mode == 'Each tokens':
        each = each_token
    elif mode == 'Each layers':
        each = each_layer
    else:
        raise ValueError(f'unknown mode: {mode}')
    
    for title, hidden_state, reference in each(output_ids, output_tokens, all_hidden_states, selected, base_layer):
        
        # 内積の計算誤差を小さくするために double にしておく
        hidden_state = hidden_state.to(dtype=torch.float64, device='cpu')
        reference = reference.to(dtype=torch.float64, device='cpu')
        
        # ノルム計算
        norms = torch.linalg.vector_norm(hidden_state, ord=2, dim=1)
        ns.append(norms.numpy())
        if norm_type == 'Absolute':
            reg_norms = norms
        elif norm_type == 'Relative to reference':
            ref_norm = torch.linalg.vector_norm(reference, ord=2, dim=1)
            reg_norms = norms.div(ref_norm)
        elif norm_type == 'Relative to previous':
            norms_ = [1.0]
            for x in range(norms.size(0)-1):
                a = norms[x]
                b = norms[x+1]
                norms_.append((b/a).item())
            reg_norms = torch.tensor(norms_)
        elif norm_type == 'Ignore':
            reg_norms = torch.ones_like(norms)
        else:
            raise ValueError(f'unknown norm_type: {norm_type}')
        
        # 角度計算
        hidden_state = torch.nn.functional.normalize(hidden_state)
        reference = torch.nn.functional.normalize(reference)
        if angle_type == 'Relative to reference':
            thetas = torch.acos(torch.clamp(hidden_state @ reference[0,:], -1, 1))
        elif angle_type == 'Relative to previous':
            thetas_ = [0.0]
            for x in range(hidden_state.size(0)-1):
                a = hidden_state[x]
                b = hidden_state[x+1]
                t = torch.acos(torch.clamp(torch.dot(a, b), -1, 1))
                thetas_.append(t.item())
            thetas = torch.tensor(thetas_)
        else:
            raise ValueError(f'unknown angle_type: {angle_type}')

        titles.append(title)

        xs = []
        ys = []
        texts = []
        ts_ = []
        for layer, (r, t) in enumerate(zip(reg_norms, thetas)):
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

        template = 'norm=%{customdata[0]:.3f}<br>theta=%{customdata[1]:.3f} [%{customdata[2]:.1f}\xb0]'
        if mode == 'Each tokens':
            template = 'layer=%{text}<br>' + template
            deg = theta * 180 / np.pi
            customdata = np.stack((norm, theta, deg), axis=-1)
        elif mode == 'Each layers':
            template = 'token=%{text} %{customdata[3]} (%{customdata[4]})<br>' + template
            deg = theta * 180 / np.pi
            tokens = np.array(output_tokens)
            ids = output_ids.cpu()
            customdata = np.stack((norm, theta, deg, tokens[selected], ids[selected]), axis=-1)

        fig.update_traces(
            row=row, col=col,
            hovertemplate=template,
            customdata=customdata,
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
