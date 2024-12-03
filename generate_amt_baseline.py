import os
import sys
import tqdm
from multiprocessing import Pool

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
from anticipation.convert import events_to_midi

from model.modeling_llama_expvocab import LlamaWithExpandedVocabForCausalLM
from utils.generation import (
    read_text_prompts,
    output_text_and_synthesize,
)

CACHE_DIR = "/workspace/.cache"
GPT2_MODEL_NAME = "stanford-crfm/music-large-800k"
GPT2_BOS_ID = 55026
TOP_P = 0.98
GPT2_VOCAB_SIZE = 55030
CKPT_DIR = sys.argv[1]
BATCH_SIZE = 16
SEQLEN = 1024

N_EVAL_SAMPLES = 100

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
    model = AutoModelForCausalLM.from_pretrained(
        GPT2_MODEL_NAME,
        torch_dtype="bfloat16",
        cache_dir=CACHE_DIR,
    ).cuda()
    print("[INFO] Model loaded")

    print("trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.eval()
    

    # Input prompt example
    # prompt = (
    #     "You are a world-class composer. Please compose some music according to the following description: "
    #     "A melodic and relaxing pop song with a touch of electronic elements, "
    #     "featuring a piano lead accompanied by electric bass, clean electric guitar, "
    #     "Hammond organ, and drums. Set in the key of C# major with a 4/4 time signature, "
    #     "this short song moves at an Adagio tempo, evoking a sense of love and creating a "
    #     "cinematic atmosphere."
    # )

    # Input prompts
    all_prompts = read_text_prompts(max_samples=N_EVAL_SAMPLES)

    if OUTDIR_SUFFIX is None:
        output_root = os.path.join(CKPT_DIR, "generations")
    else:
        output_root = os.path.join(CKPT_DIR, f"generations_{OUTDIR_SUFFIX}")


    for i in tqdm.tqdm(range(0, len(all_prompts), BATCH_SIZE)):
        keys, prompts = [x for x, y in all_prompts[i:i+BATCH_SIZE]], [y for x, y in all_prompts[i:i+BATCH_SIZE]]
        print(f"[INFO] Generating for {keys} ...")
 
        print(f"[INFO] Prompt: {prompts}")

        # Tokenize the input prompt
        input_ids = torch.tensor([[GPT2_BOS_ID]] * len(prompts)).long().cuda()
        print(input_ids)
        
        prompt_len = input_ids.size(1)
        # exit()

        print("[INFO] Start generation ...")
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                do_sample=True,
                max_new_tokens=SEQLEN - 1,
                top_p=TOP_P,
                num_return_sequences=1,
            )

        print(outputs)

        outputs = outputs[:, prompt_len:]
        print(outputs[:, :30])
        outputs = outputs.cpu().tolist()

        args = []
        p = Pool(BATCH_SIZE)

        for key, prompt, output in zip(keys, prompts, outputs):
            args.append((output, prompt, output_root, key))

        p.starmap(output_text_and_synthesize, args)
        p.close()

        # midi_path = write_events_and_midi(output, prompt, key, output_root)
        # synthesize_midi(midi_path, save_mp3_only=True)



    # for ev in output:
    #     print(ev)
    # # Decode and print the output
    # # print(output)


    # mid = events_to_midi(output)
    # mid.save("/workspace/shared/outputs/test_llama.mid")
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(generated_text)