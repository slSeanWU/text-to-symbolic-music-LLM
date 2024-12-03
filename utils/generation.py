import os
import json
import textwrap

import midi2audio
import librosa
import librosa.effects
import soundfile as sf
from anticipation.convert import events_to_midi

SOUNDFONT = "/workspace/shared/soundfonts/FluidR3_GM.sf2"
TEXT_CACHE = "/workspace/.cache/midicaps_text_cache.json"
SPLIT_FILE = "/workspace/scratch-slseanwu/text-to-symbolic-music-LLM/splits_updated_2.json"
fs = midi2audio.FluidSynth(SOUNDFONT)

def synthesize_midi(midi_path, save_mp3_only=True):
    wav_path = midi_path.replace(".mid", ".wav")

    fs.midi_to_audio(midi_path, wav_path)

    # trim silence
    wav, sr = librosa.load(wav_path)
    _orig_len = len(wav)
    wav, _ = librosa.effects.trim(wav, top_db=30)
    _new_len = len(wav)
    print(f"Trimmed {(_orig_len - _new_len) / sr:.2f} seconds of silence")
    # os.rename(wav_path, wav_path.replace(".wav", "_orig.wav"))
    sf.write(wav_path, wav, sr)

    if save_mp3_only:
        # use ffmpeg to convert wav to mp3
        os.system(f"ffmpeg -i {wav_path} -codec:a libmp3lame -qscale:a 2 {midi_path.replace('.mid', '.mp3')} -y")
        os.remove(wav_path)


def write_events_and_midi(events, text, file_id, output_root):
    output_dir = os.path.join(output_root, file_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    midi_path = os.path.join(output_dir, f"{file_id}.mid")
    events_path = os.path.join(output_dir, f"{file_id}_events.txt")
    
    with open(events_path, "w") as f:
        for event in events:
            f.write(f"{event}\n")

    with open(os.path.join(output_dir, f"prompt.txt"), "w") as f:
        f.write(textwrap.fill(text))

    mid = events_to_midi(events)
    mid.save(midi_path)

    return midi_path


def read_text_prompts(text_prompt_cache=TEXT_CACHE, split="test", max_samples=100):
    all_text_prompts = json.load(open(text_prompt_cache, "r"))

    sample_keys = json.load(open(SPLIT_FILE, "r"))[split][:max_samples]

    text_prompts = []
    for key in sample_keys:
        text_prompts.append((os.path.basename(key).replace(".mid", ""), all_text_prompts[key]))

    return text_prompts

def output_text_and_synthesize(events, text, output_root, file_id):
    try:
        midi_path = write_events_and_midi(events, text, file_id, output_root)
        synthesize_midi(midi_path, save_mp3_only=True)
    except Exception as e:
        # print(e)
        return False
    
    return True

if __name__ == "__main__":
    text_prompts = read_text_prompts()

    for key, text_prompt in text_prompts:
        print(key, text_prompt)