import json
import os
import logging
import glob

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


CACHE_DIR = "/workspace/.cache"
LLAMA_PAD_ID = 128009 # "content": "<|eot_id|>"
GPT2_PAD_ID = 55025   # `SEPARATOR` in anticipation


class DataCollatorWithLlamaLeftPadding:
    def __init__(self):
        self.padding_token_id = LLAMA_PAD_ID

    def __call__(self, features):
        # Get the max sequence length in the batch
        max_length = max(len(f["llama_input_ids"]) for f in features)

        # Prepare left-padded inputs and masks
        input_ids = []
        attention_mask = []

        for feature in features:
            seq = list(feature["llama_input_ids"])
            seq_length = len(seq)

            if seq_length < max_length:
                padding_length = max_length - seq_length

                # Left-pad the sequence
                padded_seq = [self.padding_token_id] * padding_length + seq
                input_ids.append(padded_seq)

                # Create attention mask (1 for tokens, 0 for padding)
                mask = [0] * padding_length + [1] * seq_length
                attention_mask.append(mask)
            else:
                input_ids.append(seq)
                attention_mask.append([1] * seq_length)

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # Return batch dictionary
        batch = {
            "llama_input_ids": input_ids,
            "llama_attention_mask": attention_mask,
        }

        for key in features[0]:
            if key not in batch:
                batch[key] = torch.tensor([list(f[key]) for f in features], dtype=torch.long)

        return batch


class RandomTestTextMusicDataset(Dataset):
    """Just to test out model training & implementation"""
    def __init__(self, llama_model_name, music_max_length=1024, text_max_length=256):
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            llama_model_name,
            cache_dir=CACHE_DIR,
            truncation_side="right",
            pad_token="<|eot_id|>",
        )
        
        # 10 random sentences with different lengths
        self.text = [
            "The movie was so bad that I had to laugh and laugh and laugh.",
            "The first few minutes were the only good ones.",
            "The rest was just a bunch of actors who were trying to be funny and ended up making themselves look like",
            "The acting was so bad that I was laughing so hard that I had to take a break.",
            "I'm not even sure if the film was shot or not, because it was so bad that I couldn't tell.",
            "I couldn't even tell if it was a movie or a play.",
            "The acting was so bad that I was laughing so hard that I had to take a break.",
            "The movie was so bad that I had to laugh and laugh and laugh.",
            "The first few minutes were the only good ones.",
            "The rest was just a bunch of actors who were trying to be funny and ended up making themselves look like",
        ]

        self.text_max_length = text_max_length
        self.music_max_length = music_max_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        # Randomly sample a sequence length
        music_seqlen = torch.randint(1, self.music_max_length * 2, (1,)).item()

        # Get text and tokenize it
        text = self.text[idx]
        text = self.text_tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=self.text_max_length,
            truncation=True,
        )

        # Generate random music
        music = torch.randint(0, 388, (music_seqlen,))
        if len(music) > self.music_max_length:
            music = music[:self.music_max_length]
        else:
            music = torch.cat([
                music,
                torch.full((self.music_max_length - len(music),), fill_value=GPT2_PAD_ID, dtype=music.dtype)
            ])

        return {
            "input_ids": music,
            "labels": music,
            "llama_input_ids": text["input_ids"][0],
            "llama_attention_mask": text["attention_mask"][0],
        }


class MidiCapsTextMusicForAMTDataset(Dataset):
    """The real dataset used for finetuning AMT symbolic music model with paired text"""
    def __init__(
            self,
            llama_model_name,
            split_file,
            tokenized_music_dir,
            split="train",
            music_max_length=1024,
            text_max_length=256
        ):
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            llama_model_name,
            cache_dir=CACHE_DIR,
            truncation_side="right",
            pad_token="<|eot_id|>",
        )
        
        self.split = split
        self.tokenized_music_dir = tokenized_music_dir
        self.samples = json.load(open(split_file))[split]
        self.texts = self.read_texts()
        self.musics = self.read_musics()

        self.text_max_length = text_max_length
        self.music_max_length = music_max_length

        print(f"[INFO] loaded {split} split with {len(self.samples)} samples")


    def read_texts(self):
        texts = dict()

        ds = load_dataset("amaai-lab/MidiCaps", cache_dir=CACHE_DIR)["train"]
        for example in ds:
            texts[example["location"]] = example["caption"]

        print(f"[INFO] read {len(texts)} texts from MidiCaps dataset")

        return texts

    def read_musics(self):
        tokenized_music_files = glob.glob(os.path.join(self.tokenized_music_dir, "tokenized-*"))

        music_tokens = dict()

        for f in tokenized_music_files:
            lines = open(f).readlines()

            for l in lines:
                music_id, token_text = l.strip().split(" | ")
                tokens = [int(t) for t in token_text.split()]

                music_tokens[music_id] = tokens

        return music_tokens

    def pad_or_truncate_music_tokens(self, music_tokens):
        if len(music_tokens) > self.music_max_length:
            music_tokens = music_tokens[:self.music_max_length]
        elif len(music_tokens) < self.music_max_length:
            music_tokens = music_tokens + [GPT2_PAD_ID] * (self.music_max_length - len(music_tokens))

        return music_tokens

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]

        # Get text and tokenize it
        text = self.texts[sample_id]
        text = self.text_tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=self.text_max_length,
            truncation=True,
        )

        # Get music and tokenize it
        music = self.musics[sample_id]
        music = self.pad_or_truncate_music_tokens(music)
        music = torch.tensor(music, dtype=torch.long)

        # replace GPT2_PAD_ID with -100
        music_labels = music.clone()
        music_labels[music_labels == GPT2_PAD_ID] = -100

        return {
            "input_ids": music,
            "labels": music_labels,
            "llama_input_ids": text["input_ids"][0],
            "llama_attention_mask": text["attention_mask"][0],
        }


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, GPT2LMHeadModel, LlamaForCausalLM
    import numpy as np


    # dataset = RandomTestTextMusicDataset("meta-llama/Llama-3.2-1B")
    # dataset = MidiCapsTextMusicForAMTDataset(
    #     "meta-llama/Llama-3.2-1B",
    #     "/workspace/scratch-slseanwu/text-to-symbolic-music-LLM/splits_updated.json",
    #     "/workspace/shared/data/lmd_full_tokenized",
    #     split="valid",
    #     music_max_length=1024,
    # )

    model = AutoModelForCausalLM.from_pretrained(
        "stanford-crfm/music-large-800k",
        torch_dtype="bfloat16",
    ).cuda()

    def parse_amt_tokens(token_file):
        lines = open(token_file).readlines()

        all_tokens = []

        for l in lines:
            token_text = l.strip().split(" | ")[-1]
            tokens = [int(t) for t in token_text.split()]
            # print(len(tokens), tokens[:10])

            all_tokens.append(tokens)

        return all_tokens
    
    all_tokens = parse_amt_tokens("/workspace/shared/data/lmd_full_tokenized/tokenized-events-e.txt")

    all_losses = []
    for i in range(200):
        # example = dataset[i]
        # music_tokens = example["input_ids"]
        # music_labels = example["labels"]
        music_tokens = torch.tensor(all_tokens[i][:1024], dtype=torch.long)
        music_labels = music_tokens.clone()

        print(len(music_tokens), music_tokens[:10])

        loss = model(
            input_ids=music_tokens.unsqueeze(0).cuda(),
            labels=music_labels.unsqueeze(0).cuda(),
        ).loss.item()

        print(loss)
        all_losses.append(loss)

    print(np.mean(all_losses))

    # print("[train]")
    # for i in range(len(dataset)):
    #     if not (i % 1000):
    #         print("now at", i)
        
    #     example = dataset[i]
    #     music_tokens = example["input_ids"]

    #     assert music_tokens.max() < 55030 and music_tokens.min() >= 0