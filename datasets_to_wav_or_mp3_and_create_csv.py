from datasets import load_dataset, load_from_disk
import soundfile as sf, os, re, neologdn, librosa
from tqdm import tqdm
import shutil

dataset = load_dataset("mozilla-foundation/common_voice_17_0", "ja")["train"].filter(lambda sample: bool(sample["sentence"])) #skips samples with no transcript

name = "cv_17"
ouput_dir = "./datasets/"
out_file = 'metadata.csv' # create metadata file with file names and transcripts
os.makedirs(ouput_dir + name, exist_ok=True)
folder_path = ouput_dir + name # Create a folder to store the audio and transcription files

## Removes silence from audio sample - top_db=30 - Then moves trimmed samples to trimmed subfolder
## Also removes samples with only silence - threshold=0.025 - Then moves silent samples to removed 
## subfolder for review and creates two csv files; removed and not_removed.

top_db=30

def is_silent(mp3_file, threshold=0.025):
    y, sr = librosa.load(mp3_file, sr=None)
    rms = librosa.feature.rms(y=y)[0]
    return all(value < threshold for value in rms)

def remove_silence(input_file, output_file, top_db=top_db):
        y, sr = sf.read(input_file)
        intervals = librosa.effects.split(y, top_db=top_db)
        y_trimmed = []
        for start, end in intervals:
            y_trimmed.extend(y[start:end])          
        if not os.path.exists(output_file):    
            sf.write(output_file, y_trimmed, sr)
            with open(csv_file2, "a") as f:
                file_name = os.path.basename(output_file)
                f.write(file_name + "\n")

def process_directory(input_dir, output_dir, top_db=top_db):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(removed_dir):
        os.makedirs(removed_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp3"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            removed_file = os.path.join(removed_dir, filename)
            remove_silence(input_file, output_file, top_db)
            if is_silent(output_file):
                with open(csv_file, "a") as f:
                    f.write(output_file + "\n")
                shutil.move(output_file, removed_file)


input_dir = folder_path
output_dir = folder_path + "/trimmed/"
removed_dir = folder_path + "/removed/"
csv_file = folder_path + "/removed.csv"   
csv_file2 = folder_path + "/not_removed.csv"  
process_directory(input_dir, output_dir)

## Downloads audio samples and transcrpits from hugging face datasets creating audio files with matching csv 
## contaning transcripts and file_name of your choice. Also does a bit of transcript pre-proccesing (https://github.com/ikegami-yukino/neologdn) 
## Japanese specific. 

min_char = 4
max = 20.0
min = 1.0

char = '[　０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890♬♪♩♫]'
special_characters = '[“%‘”~゛＃＄％＆（）＊＋：；〈＝〉＠＾＿｛｜｝～"█』『.;:<>_()*&^$#@`, ]' #「」

for i, sample in tqdm(enumerate(dataset)): # Process each sample in the filtered dataset
    if sample["sentence"] != "":
        audio_sample = name + f'_{i}.mp3' # or wav
        audio_path = os.path.join(folder_path, audio_sample)
        transcription_path = os.path.join(folder_path, out_file)  # Path to save transcription file 
        if not os.path.exists(audio_path):
            patterns = [(r"…",'。'), (r"!!",'!'), (special_characters,""), (r"\s+", "")] # (r"(.)\1{2}")
            for pattern, replace in patterns:
                sample["sentence"] = re.sub(pattern, replace, sample["sentence"])
                sample["sentence"] = (neologdn.normalize(sample["sentence"], repeat=1)) # for Japanese only, repeat number reduces repeat characters
                if sample["sentence"][-1] not in ["!", "?", "。"]:
                    sample["sentence"] += "。"  # Add a period if it doesn't end with ! or ?
                    sample["sentence_length"] = len(sample["sentence"]) # Get sentence lengths  
                    sample["audio_length"] = len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]  # Get audio length, remove if not needed
                    if max > sample["audio_length"] > min and not re.search(char, sample["sentence"]) and sample["sentence_length"] > min_char and bool(sample["sentence"]): #check again before write
                        sf.write(audio_path, sample['audio']['array'], sample['audio']['sampling_rate']) # Get files      
                        process_directory(folder_path, (folder_path + "/trimmed/")) # for use with audio sample silence removal script
                        if os.path.isfile(audio_path):
                            os.remove(audio_path)
                            with open(transcription_path, 'a', encoding='utf-8') as transcription_file:
                                transcription_file.write(audio_sample+",") # Save transcription file name  
                                transcription_file.write(sample['sentence']) # Save transcription 
                                transcription_file.write('\n')
