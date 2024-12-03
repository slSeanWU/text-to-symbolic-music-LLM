import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys

import bitsandbytes as bnb
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, GPT2LMHeadModel
from torch.utils.data import SequentialSampler, Subset

from model.modeling_llama_expvocab import (
    LlamaWithExpandedVocabForCausalLM,
    GPT2_VOCAB_SIZE,
)
from dataloading.data_gpt2_llamacond import (
    RandomTestTextMusicDataset,
    MidiCapsTextMusicForAMTDataset,
    DataCollatorWithLlamaLeftPadding,
)

LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-3B"
CACHE_DIR = "/workspace/.cache"
SPLIT_FILE = "/workspace/scratch-slseanwu/text-to-symbolic-music-LLM/splits_updated_2.json"
CKPT_DIR = sys.argv[1]
SEQLEN = 1024
LR = 2e-4
GPT2_MODEL_NAME = "stanford-crfm/music-large-800k"
USE_GPT2_EMBEDDINGS = True


class SequentialTrainer(Trainer):
    """(Shih-Lun) for fair comparison at same training steps, no shuffling"""
    def _get_train_sampler(self):
        return SequentialSampler(self.train_dataset)


if __name__ == "__main__":
    # Load model
    model = LlamaWithExpandedVocabForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype="bfloat16",
        expanded_vocab_size=GPT2_VOCAB_SIZE,
        attn_implementation="flash_attention_2",
    ).cuda()

    if USE_GPT2_EMBEDDINGS:
        model.init_expanded_embed_from_source(GPT2_MODEL_NAME)
        print("loaded GPT2 embeddings from", GPT2_MODEL_NAME, f"dim = {model.model.expanded_embed_tokens.weight.size(1)}")

    print("total trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # exit()

    # For debugging
    # ds_train = RandomTestTextMusicDataset(
    #     LLAMA_MODEL_NAME,
    #     music_max_length=SEQLEN,
    #     use_text_model=True,
    # )
    # ds_valid = RandomTestTextMusicDataset(
    #     LLAMA_MODEL_NAME,
    #     music_max_length=SEQLEN,
    #     use_text_model=True,
    # )

    ds_train = MidiCapsTextMusicForAMTDataset(
        LLAMA_MODEL_NAME,
        SPLIT_FILE,
        "/workspace/shared/data/lmd_full_tokenized",
        split="train",
        music_max_length=SEQLEN,
        use_text_model=True,
    )
    # ds_valid =  MidiCapsTextMusicForAMTDataset(
    #         LLAMA_MODEL_NAME,
    #         SPLIT_FILE,
    #         "/workspace/shared/data/lmd_full_tokenized",
    #         split="valid",
    #         music_max_length=SEQLEN,
    #     )

    # NOTE(Shih-Lun) -- 4K samples takes around 5 minutes
    ds_valid = Subset(
        MidiCapsTextMusicForAMTDataset(
            LLAMA_MODEL_NAME,
            SPLIT_FILE,
            "/workspace/shared/data/lmd_full_tokenized",
            split="valid",
            music_max_length=SEQLEN,
            use_text_model=True,
        ), range(4000)
    )

    # optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR) 
    optimizer = AdamW(model.parameters(), lr=LR)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=CKPT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        max_steps=15000,
        save_steps=500,
        logging_dir="./logs",
        eval_steps=250,
        logging_steps=10,
        bf16=True,  # Enable mixed precision
        report_to="none",
        dataloader_num_workers=8,
        do_eval=True,
        eval_strategy="steps",
        gradient_accumulation_steps=16,
        save_safetensors=False,
        eval_on_start=True,
        remove_unused_columns=False,
        eval_accumulation_steps=32,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        data_collator=DataCollatorWithLlamaLeftPadding(use_text_model=True),
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        optimizers=(optimizer, None),
    )

    # Train
    trainer.train()