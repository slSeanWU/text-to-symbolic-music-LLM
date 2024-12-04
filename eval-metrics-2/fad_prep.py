import os
import shutil

def move_mp3_files(source_directory, destination_directory):
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    for root, dirs, files in os.walk(source_directory):
        for file in files:
            if file.endswith('.mp3'):
                source_file = os.path.join(root, file)
                
                destination_file = os.path.join(destination_directory, file)
                
                counter = 1
                while os.path.exists(destination_file):
                    destination_file = os.path.join(destination_directory, f"{os.path.splitext(file)[0]}_{counter}.mp3")
                    counter += 1

                shutil.move(source_file, destination_file)
                print(f"Moved: {source_file} -> {destination_file}")

if __name__ == "__main__":
    source_dir = "../shared/outputs/amt_large_baseline/generations"  
    destination_dir = "../shared/outputs/amt_large_baseline/generations_mp3"  

    move_mp3_files(source_dir, destination_dir)
