import bitsandbytes as bnb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from model.modeling_gpt2_llamacond import GPT2WithLlamaConditioning
from dataloading.data_gpt2_llamacond import RandomTestTextMusicDataset

GPT2_MODEL_NAME = "stanford-crfm/music-large-800k"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B"
CACHE_DIR = "/workspace/.cache"
LR = 1e-5


if __name__ == "__main__":
    # Load model
    model = GPT2WithLlamaConditioning.from_pretrained(
        GPT2_MODEL_NAME,
        llama_model_name=LLAMA_MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2"
    ).cuda()

    dataset = RandomTestTextMusicDataset(LLAMA_MODEL_NAME, music_max_length=256, text_max_length=64)

    print(type(model))
    # exit()

    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=LR) 

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./quick_trials_24-11-26/random_dataset",
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
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        optimizers=(optimizer, None)
    )

    # Train
    trainer.train()