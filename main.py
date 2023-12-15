import html
import re
import time

import gradio as gr

from ext.load import get_loaded_model_or_default
from ext.generation_result import GenerationResult
from ext.generate import generate
from ext.attn import attn
from ext.hidden_states import hidden_states
from ui import ui

_last_generation_result: GenerationResult|None = None


def _attn(*args, **kwargs):
    return attn(_last_generation_result, *args, **kwargs)


def _hidden_states(*args, **kwargs):
    return hidden_states(_last_generation_result, *args, **kwargs)


def main(*args, **kwargs):
    global _last_generation_result
    
    try:
        result, info = generate(*args, **kwargs)
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


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    
    demo = ui(main, _attn, _hidden_states)
    
    t1 = time.perf_counter()
    print(f"launch: {(t1-t0)*1000:.1f}ms")
    
    demo.launch()
