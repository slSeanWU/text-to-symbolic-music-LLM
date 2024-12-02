import os
from multiprocessing import Pool

from dataloading.data_gpt2_llamacond import MidiCapsTextMusicForAMTDataset
from utils.generation import output_text_and_synthesize

out_root = "/workspace/shared/data/lmd_full_testset_first_3k"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B"
SPLIT_FILE = "/workspace/scratch-slseanwu/text-to-symbolic-music-LLM/splits_updated_2.json"
SEQLEN = 1024
CACHE_DIR = "/workspace/.cache"
GPT2_PAD_ID = 50256

N_EVAL_SAMPLE = 3000




if __name__ == "__main__":
    dset = MidiCapsTextMusicForAMTDataset(
        LLAMA_MODEL_NAME,
        SPLIT_FILE,
        "/workspace/shared/data/lmd_full_tokenized",
        split="test",
        music_max_length=SEQLEN,
    )
    print("[INFO] Dataset loaded")

    tokenizer = dset.text_tokenizer

    p = Pool(16)
    args = []
    for i in range(N_EVAL_SAMPLE):
        example = dset[i]
        example_id = os.path.basename(dset.samples[i]).replace(".mid", "")
        print(example_id)

        text = tokenizer.decode(example["llama_input_ids"], skip_special_tokens=True)
        print(text)

        music = example["input_ids"].tolist()
        music = [m for m in music if m != GPT2_PAD_ID][1:]
        print(len(music))

        if (
            os.path.exists(os.path.join(out_root, example_id)) and
            len(os.listdir(os.path.join(out_root, example_id))) == 4
        ):
            print(f"[INFO] {example_id} Already exists")
            continue

        args.append((music, text, out_root, example_id))

        print("\n\n\n")

    res = p.starmap(output_text_and_synthesize, args)
    print("[INFO] Successfully converted", sum(res), "samples out of", len(res))