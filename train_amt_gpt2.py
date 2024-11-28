import bitsandbytes as bnb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, GPT2LMHeadModel
from transformers.optimization import AdamW
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
SPLIT_FILE = "/workspace/scratch-slseanwu/text-to-symbolic-music-LLM/splits_updated.json"
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
        new_max_seqlen=1024,
    ).cuda()
    # model.extend_pos_emb() # extend to `new_max_seqlen` tokens

    print("total trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # exit()

    # TODO: switch to real dataset
    ds_train = MidiCapsTextMusicForAMTDataset(
        LLAMA_MODEL_NAME,
        SPLIT_FILE,
        "/workspace/shared/data/lmd_full_tokenized",
        split="train",
        music_max_length=1024,
    )
    # ds_valid = MidiCapsTextMusicForAMTDataset(
    #     LLAMA_MODEL_NAME,
    #     SPLIT_FILE,
    #     "/workspace/shared/data",
    #     split="valid",
    #     music_max_length=1024,
    # )
    ds_valid = Subset(
        MidiCapsTextMusicForAMTDataset(
            LLAMA_MODEL_NAME,
            SPLIT_FILE,
            "/workspace/shared/data/lmd_full_tokenized",
            split="valid",
            music_max_length=1024,
        ), range(200)
    )

    # optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR) 
    optimizer = AdamW(model.parameters(), lr=LR)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./quick_trials_24-11-26/quick_bsize_test_1K",
        learning_rate=LR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_steps=200,
        save_steps=100,
        logging_dir="./logs",
        eval_steps=100,
        logging_steps=10,
        bf16=True,  # Enable mixed precision
        report_to="none",
        dataloader_num_workers=8,
        do_eval=True,
        eval_strategy="steps",
        gradient_accumulation_steps=16,
        # eval_on_start=True,
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