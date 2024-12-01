import json
import os
import logging
import glob
import random
import multiprocessing

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


CACHE_DIR = "/workspace/.cache"
LLAMA_VOCAB_SIZE = 128256
LLAMA_PAD_ID = 128009 # "content": "<|eot_id|>"
GPT2_PAD_ID = 55025   # `SEPARATOR` in anticipation

TEXT_CACHE = "/workspace/.cache/midicaps_text_cache.json"

class DataCollatorWithLlamaLeftPadding:
    def __init__(self, use_text_model=False):
        self.padding_token_id = LLAMA_PAD_ID
        self.use_text_model = use_text_model

        if self.use_text_model:
            self.expanded_padding_token_id = GPT2_PAD_ID + LLAMA_VOCAB_SIZE

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

            if self.use_text_model:
                input_ids[-1].extend(list(feature["input_ids"]))
                attention_mask[-1].extend([1] * len(feature["input_ids"]))
                del feature["llama_input_ids"]

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)


        if not self.use_text_model:
            batch = {
                "llama_input_ids": input_ids,
                "llama_attention_mask": attention_mask,
            }
        else:
            labels = input_ids.clone()
            labels[labels == self.padding_token_id] = -100
            labels[labels == self.expanded_padding_token_id] = -100
            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        for key in features[0]:
            if key not in batch:
                batch[key] = torch.tensor([list(f[key]) for f in features], dtype=torch.long)

        return batch


class DataCollatorWithGPTRightPadding:
    def __init__(self):
        self.padding_token_id = GPT2_PAD_ID + LLAMA_VOCAB_SIZE

    def __call__(self, features):
        # Get the max sequence length in the batch
        max_length = max(len(f["input_ids"]) for f in features)

        # Prepare left-padded inputs and masks
        input_ids = []

        for feature in features:
            seq = list(feature["input_ids"])
            seq_length = len(seq)

            if seq_length < max_length:
                padding_length = max_length - seq_length

                # Right-pad the sequence
                padded_seq = seq + [self.padding_token_id] * padding_length
                input_ids.append(padded_seq)
            else:
                input_ids.append(seq)

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        
        # Replace paddings with -100
        labels[labels == self.padding_token_id] = -100

        # Return batch dictionary
        batch = {
            "input_ids": input_ids,
            "labels": labels,
        }

        for key in features[0]:
            if key not in batch:
                batch[key] = torch.tensor([list(f[key]) for f in features], dtype=torch.long)

        return batch


class RandomTestTextMusicDataset(Dataset):
    """Just to test out model training & implementation"""
    def __init__(
        self,
        llama_model_name,
        music_max_length=1024,
        text_max_length=256,
        use_text_model=False,
    ):
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
        self.use_text_model = use_text_model

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
        if len(music) < self.music_max_length:
            music = torch.cat([
                music,
                torch.full((self.music_max_length - len(music),), fill_value=GPT2_PAD_ID, dtype=music.dtype)
            ])

        if not self.use_text_model:
            return {
                "input_ids": music,
                "labels": music,
                "llama_input_ids": text["input_ids"][0],
                "llama_attention_mask": text["attention_mask"][0],
            }

        else:
            music += LLAMA_VOCAB_SIZE

            return {
                "input_ids": music,
                "llama_input_ids": text["input_ids"][0],
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
            text_max_length=256,
            use_text_model=False,
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
        self.musics_counter = {k: 0 for k, v in self.musics.items()}

        self.text_max_length = text_max_length
        self.music_max_length = music_max_length
        self.use_text_model = use_text_model

        print(f"[INFO] loaded {split} split with {len(self.samples)} samples")


    def read_texts(self):
        if not os.path.exists(TEXT_CACHE):
            texts = dict()

            ds = load_dataset("amaai-lab/MidiCaps", cache_dir=CACHE_DIR)["train"]
            for example in ds:
                texts[example["location"]] = example["caption"]

            with open(TEXT_CACHE, "w") as f:
                jsonstring = json.dumps(texts, indent=4)
                f.write(jsonstring + "\n")
        else:
            texts = json.load(open(TEXT_CACHE))

        print(f"[INFO] read {len(texts)} texts from MidiCaps dataset")

        return texts
    
    def _read_music_thread(self, music_file):
        music_tokens = dict()
        lines = open(music_file).readlines()

        for l in lines:
            music_id, token_text = l.strip().split(" | ")
            tokens = [int(t) for t in token_text.split()]

            if music_id in music_tokens:
                music_tokens[music_id].append(tokens)
            else:
                music_tokens[music_id] = [tokens]

        return music_tokens

    def read_musics(self):
        if self.split == "train":
            tokenized_music_files = glob.glob(os.path.join(self.tokenized_music_dir, "tokenized-*"))
        elif self.split == "valid":
            tokenized_music_files = glob.glob(os.path.join(self.tokenized_music_dir, "tokenized-events-e.txt"))
        elif self.split == "test":
            tokenized_music_files = glob.glob(os.path.join(self.tokenized_music_dir, "tokenized-events-f.txt"))
        else:
            raise ValueError(f"Invalid split: {self.split}")
    

        p = multiprocessing.Pool(len(tokenized_music_files))
        args = []
        for f in tokenized_music_files:
            args.append((f,))
        music_tokens = p.starmap(self._read_music_thread, args)

        # concatenate returned dictionaries
        all_music_tokens = dict()
        for d in music_tokens:
            all_music_tokens.update(d)

        return all_music_tokens

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

        # For CFG conditioning
        if self.split == "train":
            if random.random() < 0.1:
                if self.use_text_model:
                    text = "You are a world-class composer. Please compose some music."
                else:
                    text = ""
            else:
                if self.use_text_model:
                    text = "You are a world-class composer. Please compose some music according to the following description: " + text
                else:
                    text = text

        text = self.text_tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=self.text_max_length,
            truncation=True,
        )

        # Get music and tokenize it
        music = self.musics[sample_id]
        n_seqs_for_sample = len(music)

        if self.split == "train":
            music = music[self.musics_counter[sample_id]]
            self.musics_counter[sample_id] = (self.musics_counter[sample_id] + 1) % n_seqs_for_sample
        else:
            music = music[n_seqs_for_sample // 2]

        music = self.pad_or_truncate_music_tokens(music)
        music = torch.tensor(music, dtype=torch.long)

        if not self.use_text_model:
            # replace GPT2_PAD_ID with -100
            music_labels = music.clone()
            music_labels[music_labels == GPT2_PAD_ID] = -100

            return {
                "input_ids": music,
                "labels": music_labels,
                "llama_input_ids": text["input_ids"][0],
                "llama_attention_mask": text["attention_mask"][0],
            }
        else:
            return {
                "llama_input_ids": text["input_ids"][0],
                "input_ids": music + LLAMA_VOCAB_SIZE,
            }


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, GPT2LMHeadModel, LlamaForCausalLM
    import numpy as np


    # dataset = RandomTestTextMusicDataset("meta-llama/Llama-3.2-1B")
    dataset = MidiCapsTextMusicForAMTDataset(
        "meta-llama/Llama-3.2-1B",
        "/workspace/scratch-slseanwu/text-to-symbolic-music-LLM/splits_updated_2.json",
        "/workspace/shared/data/lmd_full_tokenized",
        split="valid",
        music_max_length=1024,
        use_text_model=True,
    )

    for i in range(100):
        dataset[i]

    exit()

    # model = AutoModelForCausalLM.from_pretrained(
    #     "stanford-crfm/music-large-800k",
    #     torch_dtype="bfloat16",
    # ).cuda()

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