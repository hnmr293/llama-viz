import time
import torch
import einops

from ext.load import reload_model_hf
from ext.utils import fix_seed
from ext.generation_result import GenerationResult, GenerationInfo


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


def generate(
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

    model_rev = model_id.rfind(":")
    if model_rev < 0:
        model_rev = None
    else:
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
