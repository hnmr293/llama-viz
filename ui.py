from typing import Callable
import os
import gradio as gr
from huggingface_hub import scan_cache_dir, _CACHED_NO_EXIST

DefaultModelDir = f"{os.path.dirname(__file__)}/models"

def get_models(dir):
    if not os.path.exists(dir):
        raise ValueError(f"{dir} is not exist")
    
    models = []
    for cache_info in scan_cache_dir(dir).repos:
        if cache_info.repo_type != 'model':
            continue
        repo_id = cache_info.repo_id
        for rev in cache_info.revisions:
            refs = next(iter(rev.refs))
            models.append(f'{repo_id}:{refs}')

    return 'None', ['None'] + sorted(models)

def R(min, max, value, step=None):
    d = { "minimum": min, "maximum": max, "value": value }
    if step is not None:
        d["step"] = step
    return d

def main_tab():
    with gr.Row():
        with gr.Column():
            with gr.Group():
                #hf_or_local = gr.Radio(choices=["HF", "Local"], value="HF", label="Load model from")
                with gr.Row():
                    default_model, default_models = get_models(DefaultModelDir)
                    model_id = gr.Dropdown(choices=default_models, value=default_model, label="Model repo ID", interactive=True, allow_custom_value=True)
                    with gr.Row():
                        refresh = gr.Button(value="\U0001f504", elem_classes=["refresh", "iconbutton"])
                        cache_dir = gr.Textbox(value=DefaultModelDir, placeholder="put path to cache dir here", label="Cache dir")
                        def refresh_models(dir):
                            default_model, default_models = get_models(dir)
                            return gr.update(choices=default_models, value=default_model, interactive=True)
                        refresh.click(fn=refresh_models, inputs=[cache_dir], outputs=[model_id])
                local_only = gr.Checkbox(value=False, label="Local only")
                trust_remote_code = gr.Checkbox(value=False, label="trust_remote_code")
            with gr.Group():
                prompt = gr.Textbox(lines=5, value="こんにちは。", placeholder="input prompt here", label="Prompt")
                seed = gr.Number(value=-1, minimum=-1, maximum=0xffff_fffe, precision=0, label="Seed (-1 for random)")
            run = gr.Button(value="Run (Ctrl+Enter)", variant="primary", elem_id="run")
            
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
                info = gr.Textbox(label="Info", lines=5, elem_id="output_info", elem_classes="output_info", interactive=False)
                error = gr.Textbox(label="Errors", lines=5, interactive=False, visible=False)
        with gr.Tab("Original"):
            result_original = gr.Textbox(value="", label="Output", lines=50, max_lines=50, interactive=False)
        
        inputs = [
            model_id,
            cache_dir,
            local_only,
            trust_remote_code,
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

        return [
            run,
            result,
            result_original,
            info,
            error,
            inputs,
        ]

def attn_tab(attn: Callable):
    with gr.Group():
        attn_show = gr.Radio(choices=["All", "Selected", "None"], value="Selected", label="Visibility")
        with gr.Row():
            zsel = gr.Radio(choices=["Mean", "Median", "Max", "Min", "2-Norm"], value="Mean", label="Display value (how `num_heads` will be aggregated)")
            zscale = gr.Radio(choices=["Linear", "Log10"], value="Linear", label="Scale")
            with gr.Group(visible=True) as attn_scale_linear:
                zmin = gr.Slider(minimum=-1, maximum=2, value=0, step=0.01, label="z_min")
                zmax = gr.Slider(minimum=-1, maximum=2, value=1, step=0.01, label="z_max")
            with gr.Group(visible=False) as attn_scale_log:
                zmin_log = gr.Slider(minimum=-10, maximum=0, value=-2, step=0.01, label="log(z)_min")
                zmax_log = gr.Slider(minimum=-10, maximum=0, value=0, step=0.01, label="log(z)_max")
            def toggle_scale(v):
                return gr.update(visible=v=="Linear"), gr.update(visible=v=="Log10")
            zscale.change(toggle_scale, inputs=[zscale], outputs=[attn_scale_linear, attn_scale_log])
    attn_select = gr.HTML(value='<label>Output<div class="output"></div></label>', elem_classes="output attn")
    attn_select_clear = gr.Button(value="Clear Selection")
    attn_graph_create = gr.Button(value="Show (Ctrl+Shift+Enter)", variant="primary", elem_id="show-attn")
    attn_graph = gr.Plot(visible=False)

    attn_dummy = gr.Textbox(visible=False)

    attn_select_clear.click(None, [], [], js='_ => Array.from(document.querySelectorAll(".output.attn .token")).forEach(z => z.classList.remove("selected")) || []')
    attn_graph_create.click(
        attn,
        inputs=[attn_show, zsel, zscale, zmin, zmax, zmin_log, zmax_log, attn_dummy],
        outputs=[attn_graph],
        js='(...xs) => [...xs.slice(0,-2), JSON.stringify(Array.from(document.querySelectorAll(".output.attn .token.selected")).map(z => +z.dataset.tokenPos))]'
    )

    return attn_select

def hidden_states_tab(show_states: Callable):
    #mode = gr.Radio(choices=['Tokens', 'Layers'], value='Tokens', label='Graphs')
    #base_token = gr.
    with gr.Group():
        mode = gr.Radio(choices=['Each tokens', 'Each layers'], value='Each tokens', label='Mode', info='''
[Each token] Show graphs for each tokens.
[Each layers] Show graphs for each layers.
'''.strip())
        show = gr.Radio(choices=['All', 'Selected', 'None'], value='Selected', label='Visibility')
        with gr.Row():
            base = gr.Number(value=0, label="Reference", info='''
The index of the reference layer (for "Each tokens" mode) or the reference token (for "Each layer" mode).
[Each tokens] 0: before first LlamaDecoderLayer; >0: after LlamaDecoderLayer(s).
[Each layers] The index of output tokens.
'''.strip())
            norm = gr.Radio(choices=['Absolute', 'Ignore', 'Relative to reference', 'Relative to previous'], value='Relative to reference',
                            label='Norm', info='Ignore: always 1\nRelative to: ||current||/||reference||')
            angle = gr.Radio(choices=['Relative to reference', 'Relative to previous'], value='Relative to reference',
                             label='Angle', info='Relative to: Arg(current - reference)')
    select = gr.HTML(value='<label>Output<div class="output"></div></label>', elem_classes="output hidden_states")
    select_clear = gr.Button(value="Clear Selection")
    graph_create = gr.Button(value="Show", variant="primary")
    graph = gr.Plot(visible=False)

    dummy = gr.Textbox(visible=False)

    select_clear.click(None, [], [], js='_ => Array.from(document.querySelectorAll(".output.hidden_states .token")).forEach(z => z.classList.remove("selected")) || []')
    graph_create.click(
        show_states,
        inputs=[mode, show, base, norm, angle, dummy],
        outputs=[graph],
        js='(...xs) => [...xs.slice(0,-2), JSON.stringify(Array.from(document.querySelectorAll(".output.hidden_states .token.selected")).map(z => +z.dataset.tokenPos))]'
    )

    return select

def ui(main: Callable, attn: Callable, states: Callable):
    
    with gr.Blocks(analytics_enabled=False, css="./default.css", js="./default.js") as demo:
        
        with gr.Tab("Main"):
            run, result, result_original, info, error, inputs_main = main_tab()
        
        with gr.Tab("Attentions"):
            attn_select = attn_tab(attn)
        
        with gr.Tab("Hidden States"):
            hidden_states_select = hidden_states_tab(states)
        
        run.click(main, inputs=inputs_main, outputs=[result, result_original, attn_select, hidden_states_select, info, error])
    
    return demo
