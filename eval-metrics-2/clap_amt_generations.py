import os
import glob
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import laion_clap
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from laion_clap.clap_module.factory import load_state_dict
import wget

import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import warnings
warnings.filterwarnings("ignore")


class newCLAPModule(laion_clap.CLAP_Module):
    def load_ckpt(self, ckpt = None, model_id = -1, verbose = True):
            """Load the pretrained checkpoint of CLAP model

            Parameters
            ----------
            ckpt: str
                if ckpt is specified, the model will load this ckpt, otherwise the model will download the ckpt from zenodo. \n 
                For fusion model, it will download the 630k+audioset fusion model (id=3). For non-fusion model, it will download the 630k+audioset model (id=1).
            model_id:
                if model_id is specified, you can download our best ckpt, as:
                    id = 0 --> 630k non-fusion ckpt \n
                    id = 1 --> 630k+audioset non-fusion ckpt \n
                    id = 2 --> 630k fusion ckpt \n
                    id = 3 --> 630k+audioset fusion ckpt \n
                Note that if your model is specied as non-fusion model but you download a fusion model ckpt, you will face an error.
            """
            download_link = 'https://huggingface.co/lukewys/laion_clap/resolve/main/'
            download_names = [
                '630k-best.pt',
                '630k-audioset-best.pt',
                '630k-fusion-best.pt',
                '630k-audioset-fusion-best.pt',
                'music_audioset_epoch_15_esc_90.14.pt'
            ]
            if ckpt is not None:
                print(f'Load the specified checkpoint {ckpt} from users.')
            else:
                print(f'Load our best checkpoint in the paper.')
                if model_id == -1:
                    model_id = 3 if self.enable_fusion else 1
                package_dir = os.path.dirname(os.path.realpath(__file__))
                weight_file_name = download_names[model_id]
                ckpt = os.path.join(package_dir, weight_file_name)
                if os.path.exists(ckpt):
                    print(f'The checkpoint is already downloaded')
                else:
                    print('Downloading laion_clap weight files...')
                    ckpt = wget.download(download_link + weight_file_name, os.path.dirname(ckpt))
                    print('Download completed!')
            print('Load Checkpoint...')
            ckpt = load_state_dict(ckpt, skip_params=True)
            self.model.load_state_dict(ckpt)
            if verbose:
                param_names = [n for n, p in self.model.named_parameters()]
                for n in param_names:
                    print(n, "\t", "Loaded" if n in ckpt else "Unloaded")
    

model = newCLAPModule(enable_fusion=False, amodel="HTSAT-base")

model.load_ckpt(model_id=4)

GEN_DIR = "../shared/outputs/amt_large_baseline/generations"


def write_scores(scores):
    pass

if __name__ == "__main__":
    subdirs = os.listdir(GEN_DIR)
    scores = dict()
    all_scores = []

    print(subdirs)

    for sd in subdirs:
        if sd == "clap_scores_amt_generations.json":
            continue

        subdir_path_mp3 = os.path.join("../shared/outputs/amt_large_baseline/generations", sd)
        print(subdir_path_mp3)
        subdir_path = os.path.join(GEN_DIR, sd)

        audio_files = glob.glob("../shared/outputs/amt_large_baseline/generations/" + sd + ".mp3")
        
        if not audio_files:
            print(f"Skipping {sd} - no .mp3 files found.")
            continue

        prompt_path = os.path.join(subdir_path, "prompt.txt")
        try:
            with open(prompt_path, "r") as file:
                text_prompt = file.read().replace("\n", " ")
        except FileNotFoundError:
            print(f"Skipping {sd} - prompt.txt not found.")
            continue

        print(f"Processed prompt for {sd}: {text_prompt}")

        with torch.no_grad():
            audio_embs = model.get_audio_embedding_from_filelist(
                audio_files, use_tensor=True
            )

            text_embs = model.get_text_embedding(text_prompt, use_tensor=True)
            text_embs = text_embs.repeat(audio_embs.size(0), 1)

            print(audio_embs.size(), text_embs.size())
        # Compute cosine similarities
        cos_sims = cosine_similarity(
            audio_embs.cpu().numpy(), text_embs.cpu().numpy()
        ).diagonal()

        scores[sd] = {
            "n_samples": audio_embs.size(0),
            "mean": float(cos_sims.mean()),
            "stdev": float(cos_sims.std()),
        }

        print(cos_sims)

        all_scores.extend(list(cos_sims))

    scores["ALL"] = {
        "n_samples": len(all_scores),
        "mean": float(np.mean(all_scores)),
        "stdev": float(np.std(all_scores)),
    }

    scores_str = json.dumps(scores, indent=4)
    with open(os.path.join(GEN_DIR, "clap_scores_amt_generations.json"), "w") as f:
        f.write(scores_str + "\n")

    print(scores_str)
