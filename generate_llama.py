import os
import sys
import tqdm

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
from anticipation.convert import events_to_midi

from model.modeling_llama_expvocab import LlamaWithExpandedVocabForCausalLM
from utils.generation import (
    read_text_prompts,
    synthesize_midi,
    write_events_and_midi,
)


CACHE_DIR = "/workspace/.cache"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B"
LLAMA_VOCAB_SIZE = 128256
GPT2_BOS_ID = 55026
TOP_P = 0.98
GPT2_VOCAB_SIZE = 55030
CKPT_DIR = sys.argv[1]

if len(sys.argv) > 2:
    OUTDIR_SUFFIX = sys.argv[2]
else:
    OUTDIR_SUFFIX = None

if __name__ == "__main__":
    # Load model
    # model = AutoModelForCausalLM.from_pretrained(
    #     "openai-community/gpt2",
    #     cache_dir=CACHE_DIR,
    # ).cuda()
    model = LlamaWithExpandedVocabForCausalLM.from_pretrained(
        CKPT_DIR,
        expanded_vocab_size=GPT2_VOCAB_SIZE,
        torch_dtype="bfloat16",
    ).cuda()
    print("[INFO] Model loaded")

    print("trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_MODEL_NAME,
        cache_dir=CACHE_DIR,
    )
    

    # Input prompt
    # prompt = (
    #     "You are a world-class composer. Please compose some music according to the following description: "
    #     "A melodic and relaxing pop song with a touch of electronic elements, "
    #     "featuring a piano lead accompanied by electric bass, clean electric guitar, "
    #     "Hammond organ, and drums. Set in the key of C# major with a 4/4 time signature, "
    #     "this short song moves at an Adagio tempo, evoking a sense of love and creating a "
    #     "cinematic atmosphere."
    # )
    # Input prompt
    prompts = read_text_prompts()

    if OUTDIR_SUFFIX is None:
        output_root = os.path.join(CKPT_DIR, "generations")
    else:
        output_root = os.path.join(CKPT_DIR, f"generations_{OUTDIR_SUFFIX}")

    for i in tqdm.tqdm(range(len(prompts))):
        key, prompt = prompts[i]
        print(f"[INFO] Generating for {key} ...")

        prefix = "You are a world-class composer. Please compose some music according to the following description: " 
        print(f"[INFO] Prompt: {prefix + prompt}")

        # Tokenize the input prompt
        llama_input_ids = tokenizer.encode(prefix + prompt, return_tensors="pt").cuda()
        input_ids = torch.tensor([[GPT2_BOS_ID + LLAMA_VOCAB_SIZE]] ).long().cuda()
        input_ids = torch.cat([llama_input_ids, input_ids], dim=1)
        print(input_ids)
        

        prompt_len = input_ids.size(1)
        # exit()


        print("[INFO] Start generation ...")
        # Generate text
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                do_sample=True,
                max_new_tokens=1023,
                top_p=TOP_P,
                temperature=1.1,
                num_return_sequences=1,
            )

        output = output[:, prompt_len:]
        output -= LLAMA_VOCAB_SIZE
        output = output[0].cpu().tolist()

        midi_path = write_events_and_midi(output, prompt, key, output_root)
        synthesize_midi(midi_path)



    # for ev in output:
    #     print(ev)
    # # Decode and print the output
    # # print(output)


    # mid = events_to_midi(output)
    # mid.save("/workspace/shared/outputs/test_llama.mid")
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(generated_text)