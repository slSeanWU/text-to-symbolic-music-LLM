import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

CACHE_DIR = "/workspace/.cache"
LLAMA_PAD_ID = 128009 # "content": "<|eot_id|>"
GPT2_PAD_ID = 55025   # `SEPARATOR` in anticipation


class RandomTestTextMusicDataset(Dataset):
    """Just to test out model training & implementation"""
    def __init__(self, llama_model_name, music_max_length=1024, text_max_length=256):
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            llama_model_name,
            cache_dir=CACHE_DIR,
            padding_side="left",
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
            padding="max_length",
            max_length=self.text_max_length,
            truncation=True,
            pad_to_max_length=True
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
        }


if __name__ == "__main__":
    dataset = RandomTestTextMusicDataset("meta-llama/Llama-3.2-1B")
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])
    print(dataset[3])
    print(dataset[4])
    print(dataset[5])
    print(dataset[6])
    print(dataset[7])
    print(dataset[8])
    print(dataset[9])