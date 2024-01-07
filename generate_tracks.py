import os
import subprocess

path = "/Users/alois/Desktop/projects/Car"
tracks_dir = './data/tracks/'
main_script = './src/main.py'

for track_name in os.listdir(tracks_dir):
    # Get only the file name
    filename = os.path.basename(track_name)
    if "_sur" not in track_name and "DS_Store" not in track_name and filename.endswith(".png"):
        command = f"cd {path} && python {main_script}"
        input_data = f"'0\n{filename[:-4]}\n'"
        subprocess.Popen(['osascript', '-e', f'tell application "Terminal" to do script "{command} <<< {input_data}"'], text=True)
