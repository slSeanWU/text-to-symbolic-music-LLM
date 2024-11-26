import os
from pydub import AudioSegment

def is_wav_10_seconds(filepath):
    """Check if the .wav file is 10 seconds long."""
    try:
        audio = AudioSegment.from_file(filepath)
        return len(audio) == 10000  # Length in milliseconds
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def clean_directory(root_dir):
    for subdir, _, files in os.walk(root_dir):
        txt_files = [f for f in files if f.endswith('.txt') and f != "caption.txt"]
        wav_files = [f for f in files if f.endswith('.wav')]

        # Delete .txt files except "caption.txt"
        for txt_file in txt_files:
            filepath = os.path.join(subdir, txt_file)
            try:
                os.remove(filepath)
                print(f"Deleted {filepath}")
            except Exception as e:
                print(f"Error deleting {filepath}: {e}")

        # Handle .wav files
        ten_second_wav = None
        for wav_file in wav_files:
            filepath = os.path.join(subdir, wav_file)
            if is_wav_10_seconds(filepath):
                ten_second_wav = wav_file
                break  # Found a 10-second .wav file, no need to check further

        # Delete all .wav files except the 10-second one
        for wav_file in wav_files:
            if wav_file != ten_second_wav:
                filepath = os.path.join(subdir, wav_file)
                try:
                    os.remove(filepath)
                    print(f"Deleted {filepath}")
                except Exception as e:
                    print(f"Error deleting {filepath}: {e}")

if __name__ == "__main__":
    root_directory = 'small_audiollm_set_241120'
    if os.path.isdir(root_directory):
        clean_directory(root_directory)
    else:
        print("Invalid directory. Please provide a valid path.")
