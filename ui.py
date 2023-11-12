from typing import Callable
import os
import gradio as gr

def R(min, max, value, step=None):
    d = { "minimum": min, "maximum": max, "value": value }
    if step is not None:
        d["step"] = step
    return d

def ui(main: Callable, attn: Callable):
    
    with gr.Blocks(analytics_enabled=False, css="./default.css", js="./default.js") as demo:
        
        with gr.Tab("Main"):
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        #hf_or_local = gr.Radio(choices=["HF", "Local"], value="HF", label="Load model from")
                        with gr.Row():
                            model_id = gr.Textbox(value="meta-llama/Llama-2-7b", placeholder="put repo id here", label="Model repo ID")
                            model_rev = gr.Textbox(value="", placeholder="if exists, put model revision here", label="Model revision")
                        cache_dir = gr.Textbox(value=f"{os.path.dirname(__file__)}/models", placeholder="put path to cache dir here", label="Cache dir")
                        local_only = gr.Checkbox(value=False, label="Local only")
                        #model_path = gr.Textbox(value="", placeholder="put path to the model here", label="Model path", visible=False)
                        #def hf_or_local_callback(v: str):
                        #    if v == "HF":
                        #        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
                        #    else:
                        #        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
                        #hf_or_local.change(hf_or_local_callback, inputs=[], outputs=[model_id, cache_dir, model_path])
                    with gr.Group():
                        prompt = gr.Textbox(lines=5, value="こんにちは。", placeholder="input prompt here", label="Prompt")
                        seed = gr.Number(value=-1, label="seed")
                    run = gr.Button(value="Run", variant="primary", elem_id="run")
                    
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
        
        with gr.Tab("Attentions"):
            with gr.Group():
                attn_show = gr.Radio(choices=["All", "Selected", "None"], value="Selected", label="Visibility")
                with gr.Row():
                    zmin = gr.Slider(minimum=-1, maximum=2, value=0, step=0.01, label="z_min")
                    zmax = gr.Slider(minimum=-1, maximum=2, value=1, step=0.01, label="z_max")
            attn_select = gr.HTML(value='<label>Output<div class="output"></div></label>', elem_classes="output attn")
            attn_select_clear = gr.Button(value="Clear Selection")
            attn_graph_create = gr.Button(value="Show", variant="primary")
            attn_graph = gr.Plot(visible=False)

            attn_dummy = gr.Textbox(visible=False)

            attn_select_clear.click(None, [], [], js='_ => Array.from(document.querySelectorAll(".output.attn .token")).forEach(z => z.classList.remove("selected")) || []')
            attn_graph_create.click(attn, inputs=[attn_show, zmin, zmax, attn_dummy], outputs=[attn_graph], js='(x0,x1,x2,x3) => [x0, x1, x2, JSON.stringify(Array.from(document.querySelectorAll(".output.attn .token.selected")).map(z => +z.dataset.tokenPos))]')
        
        inputs = [
            #hf_or_local,
            model_id,
            model_rev,
            cache_dir,
            local_only,
            #model_path,
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
        
        run.click(main, inputs=inputs, outputs=[result, result_original, attn_select, info, error])
    
    return demo
