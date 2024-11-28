import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
from anticipation.convert import events_to_midi

from model.modeling_gpt2_llamacond import GPT2WithLlamaConditioning

CACHE_DIR = "/workspace/.cache"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B"
GPT2_BOS_ID = 55026
TOP_P = 0.98

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
):
    gen_tokens = [GPT2_BOS_ID]

    # 1st step to get KV cache
    outputs = model.forward(
        llama_input_ids=llama_input_ids,
        llama_attention_mask=llama_attention_mask,
        input_ids=torch.tensor([gen_tokens]).long().cuda(),
        use_cache=True,
    )
    gen_tokens.append(
        sample_nucleus(outputs.logits[:, -1], TOP_P)[0].item()
    )

    for i in range(max_new_tokens - 1):
        if not i % 100:
            print(f"[DEBUG] Generated {i} tokens")
        # print(gen_tokens[-1])
        outputs = model.forward(
            input_ids=torch.tensor([[gen_tokens[-1]]]).long().cuda(),
            position_ids=torch.tensor([[i + 1]]).long().cuda(),
            past_key_values=outputs.past_key_values,
            use_cache=True,
        )
        gen_tokens.append(
            sample_nucleus(outputs.logits[:, -1], TOP_P)[0].item()
        )

    return gen_tokens[1:]


if __name__ == "__main__":
    # Load model
    # model = AutoModelForCausalLM.from_pretrained(
    #     "openai-community/gpt2",
    #     cache_dir=CACHE_DIR,
    # ).cuda()
    model = GPT2WithLlamaConditioning.from_pretrained(
        "/workspace/shared/ckpt/amt_llama_text_cond/plain_24-11-28/checkpoint-5000",
        llama_model_name=LLAMA_MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype="bfloat16",
    ).cuda()
    print("[INFO] Model loaded")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_MODEL_NAME,
        cache_dir=CACHE_DIR,
    )

    # Input prompt
    prompt = "A melodic and relaxing pop song with a touch of electronic elements, featuring a piano lead accompanied by electric bass, clean electric guitar, Hammond organ, and drums. Set in the key of C# major with a 4/4 time signature, this short song moves at an Adagio tempo, evoking a sense of love and creating a cinematic atmosphere."

    # Tokenize the input prompt
    llama_input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    input_ids = torch.tensor([[GPT2_BOS_ID]]).long().cuda()

    print("[INFO] Start generation ...")
    # Generate text
    output = nucleus_amt_generate(model, llama_input_ids, max_new_tokens=1023)

    # Decode and print the output
    print(output)

    mid = events_to_midi(output)
    mid.save("/workspace/shared/outputs/test.mid")
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(generated_text)