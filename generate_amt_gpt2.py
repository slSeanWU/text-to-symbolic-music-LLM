import os
import sys
from multiprocessing import Pool

import torch
import tqdm
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
from anticipation.convert import events_to_midi

from model.modeling_gpt2_llamacond import GPT2WithLlamaConditioning
from utils.generation import (
    read_text_prompts,
    output_text_and_synthesize,
)

CACHE_DIR = "/workspace/.cache"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B"
GPT2_BOS_ID = 55026
TOP_P = 0.98
BATCH_SIZE = 8
SEQLEN = 1024
N_EVAL_SAMPLES = 1000

CKPT_DIR = sys.argv[1]

if len(sys.argv) > 2:
    OUTDIR_SUFFIX = sys.argv[2]
else:
    OUTDIR_SUFFIX = None

def sample_nucleus(scores: torch.FloatTensor, top_p: float, min_tokens_to_keep: int = 1):
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[..., -min_tokens_to_keep :] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    scores_processed = scores.masked_fill(indices_to_remove, -float("Inf"))

    probs = nn.functional.softmax(scores_processed, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  

    return next_tokens

@torch.inference_mode()
def nucleus_amt_generate(
    model: GPT2WithLlamaConditioning,
    llama_input_ids,
    max_new_tokens=1023,
    llama_attention_mask=None,
    is_batched=False
):
    if not is_batched:
        gen_tokens = [[GPT2_BOS_ID]]
    else:
        batch_size = llama_input_ids.size(0)
        gen_tokens = [[GPT2_BOS_ID] for _ in range(batch_size)]

    # 1st step to get KV cache
    outputs = model.forward(
        llama_input_ids=llama_input_ids,
        llama_attention_mask=llama_attention_mask,
        input_ids=torch.tensor(gen_tokens).long().cuda(),
        use_cache=True,
    )
    if not is_batched:
        gen_tokens.append(
            sample_nucleus(outputs.logits[:, -1], TOP_P)[0].item()
        )
    else:
        all_next_tokens = sample_nucleus(outputs.logits[:, -1], TOP_P)
        for i in range(batch_size):
            gen_tokens[i].append(all_next_tokens[i].item())

    for i in range(max_new_tokens - 1):
        if not i % 100:
            print(f"[DEBUG] Generated {i} tokens")
        # print(gen_tokens[-1])
        outputs = model.forward(
            input_ids=torch.tensor([[samp_tokens[-1]] for samp_tokens in gen_tokens]).long().cuda(),
            position_ids=torch.tensor([[i + 1] for _ in range(batch_size)]).long().cuda(),
            past_key_values=outputs.past_key_values,
            use_cache=True,
        )
        if not is_batched:
            gen_tokens.append(
                sample_nucleus(outputs.logits[:, -1], TOP_P)[0].item()
            )
        else:
            all_next_tokens = sample_nucleus(outputs.logits[:, -1], TOP_P)
            for i in range(batch_size):
                gen_tokens[i].append(all_next_tokens[i].item())

    if not is_batched:
        return gen_tokens[1:]
    else:
        return [gen_tokens[i][1:] for i in range(batch_size)]


if __name__ == "__main__":
    # Load model
    # model = AutoModelForCausalLM.from_pretrained(
    #     "openai-community/gpt2",
    #     cache_dir=CACHE_DIR,
    # ).cuda()
    model = GPT2WithLlamaConditioning.from_pretrained(
        CKPT_DIR,
        llama_model_name=LLAMA_MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype="bfloat16",
        use_weighted_llama_states="weighted" in CKPT_DIR,
    ).cuda()
    print("[INFO] Model loaded")

    if model.use_weighted_llama_states:
        print("Using weighted llama states")
        print(model.llama_state_weights)
        print(model.llama_state_weights.requires_grad)
    
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_MODEL_NAME,
        cache_dir=CACHE_DIR,
        pad_token="<|eot_id|>",
    )

    if OUTDIR_SUFFIX is None:
        output_root = os.path.join(CKPT_DIR, "generations")
    else:
        output_root = os.path.join(CKPT_DIR, f"generations_{OUTDIR_SUFFIX}")

    # Input prompt
    all_prompts = read_text_prompts(max_samples=N_EVAL_SAMPLES)

    # Tokenize the input prompt
    for i in tqdm.tqdm(range(0, len(all_prompts), BATCH_SIZE)):
        keys, prompts = [x for x, y in all_prompts[i:i+BATCH_SIZE]], [y for x, y in all_prompts[i:i+BATCH_SIZE]]
        print(f"[INFO] Generating for {keys} ...")
 
        print(f"[INFO] Prompt: {prompts}")

        # Tokenize the input prompt
        llama_input_bundle = tokenizer(
            [prompt for prompt in prompts],
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )

        llama_input_ids = llama_input_bundle["input_ids"].cuda()
        llama_attention_mask = llama_input_bundle["attention_mask"].cuda()

        print("[INFO] Start generation ...")
        # Generate text
        outputs = nucleus_amt_generate(
            model,
            llama_input_ids,
            max_new_tokens=SEQLEN - 1,
            is_batched=True,
            llama_attention_mask=llama_attention_mask,
        )

        print("[INFO] Generated tokens:")
        for out in outputs:
            print(out[:30])

        args = []
        p = Pool(min(BATCH_SIZE, 8))

        for key, prompt, output in zip(keys, prompts, outputs):
            args.append((output, prompt, output_root, key))

        res = p.starmap(output_text_and_synthesize, args)
        p.close()

        print("[INFO] Successfully generated", sum(res), "samples out of", len(res))

        # Decode and print the output
        # print(output)

        # midi_path = write_events_and_midi(output, prompt, key, output_root)
        # synthesize_midi(midi_path)

    # llama_input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    # input_ids = torch.tensor([[GPT2_BOS_ID]]).long().cuda()

    # print("[INFO] Start generation ...")
    # # Generate text
    # output = nucleus_amt_generate(model, llama_input_ids, max_new_tokens=1023)

    # # Decode and print the output
    # print(output)

    # mid = events_to_midi(output)
    # mid.save("/workspace/shared/outputs/test.mid")
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(generated_text)