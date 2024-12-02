import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

import bitsandbytes as bnb
from datasets import load_dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, GPT2LMHeadModel
from torch.utils.data import SequentialSampler, Subset

from model.modeling_gpt2_llamacond import GPT2WithLlamaConditioning
from dataloading.data_gpt2_llamacond import (
    RandomTestTextMusicDataset,
    MidiCapsTextMusicForAMTDataset,
    DataCollatorWithLlamaLeftPadding,
)

GPT2_MODEL_NAME = "stanford-crfm/music-large-800k"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B"
CACHE_DIR = "/workspace/.cache"
SPLIT_FILE = "/workspace/scratch-slseanwu/text-to-symbolic-music-LLM/splits_updated_2.json"
CKPT_DIR = sys.argv[1]
SEQLEN = 1024
LR = 1e-5

class SequentialTrainer(Trainer):
    """(Shih-Lun) for fair comparison at same training steps, no shuffling"""
    def _get_train_sampler(self):
        return SequentialSampler(self.train_dataset)

if __name__ == "__main__":
    # Load model
    model = GPT2WithLlamaConditioning.from_pretrained(
        GPT2_MODEL_NAME,
        llama_model_name=LLAMA_MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype="bfloat16",
        use_weighted_llama_states=True,
    ).cuda()

    if model.use_weighted_llama_states:
        print("Using weighted llama states")
        print(model.llama_state_weights)

    print("total trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # exit()

    # For debugging
    # ds_train = RandomTestTextMusicDataset(
    #     LLAMA_MODEL_NAME,
    #     music_max_length=SEQLEN,
    # )
    # ds_valid = RandomTestTextMusicDataset(
    #     LLAMA_MODEL_NAME,
    #     music_max_length=SEQLEN,
    # )

    ds_train = MidiCapsTextMusicForAMTDataset(
        LLAMA_MODEL_NAME,
        SPLIT_FILE,
        "/workspace/shared/data/lmd_full_tokenized",
        split="train",
        music_max_length=SEQLEN,
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
        ), range(4000)
    )

    # optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR) 
    optimizer = AdamW(model.parameters(), lr=LR)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=CKPT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        max_steps=60000,
        save_steps=2000,
        logging_dir="./logs",
        eval_steps=2000,
        logging_steps=10,
        bf16=True,  # Enable mixed precision
        report_to="none",
        dataloader_num_workers=4,
        do_eval=True,
        eval_strategy="steps",
        gradient_accumulation_steps=4,
        save_safetensors=False,
        eval_on_start=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        data_collator=DataCollatorWithLlamaLeftPadding(),
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        optimizers=(optimizer, None),
    )

    # Train
    trainer.train()