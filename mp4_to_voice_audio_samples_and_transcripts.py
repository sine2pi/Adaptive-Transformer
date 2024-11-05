import pandas as pd
import os, io, re, sys, time, datetime, wave, contextlib, librosa
from glob import glob
import numpy as np
from moviepy.editor import *
import soundfile as sf
from pydub import AudioSegment

def create_directories():
    slice_path = './ready_for_slice'
    if not os.path.exists(slice_path):
        try:
            os.mkdir(slice_path)
        except OSError:
            print('Creation of directory %s failed' %slice_path)
    sliced_audio = './sliced_audio'
    if not os.path.exists(sliced_audio):
        try:
            os.mkdir(sliced_audio)
        except OSError:
            print('Creation of directory %s failed' %sliced_audio)

    merged_csv_files = './merged_csv'
    if not os.path.exists(merged_csv_files):
        try:
            os.mkdir(merged_csv_files)
        except OSError:
            print('Creation of directory %s failed' %merged_csv_files)

    final_csv_files = './final_csv'
    if not os.path.exists(final_csv_files):
        try:
            os.mkdir(final_csv_files)
        except OSError:
            print('Creation of directory %s failed' %final_csv_files)
            
    audio = './audio'
    if not os.path.exists(audio):
        try:
            os.mkdir(audio)
        except OSError:
            print('Creation of directory %s failed' %audio)
            
    srt_files = './srt_files'
    if not os.path.exists(srt_files):
        try:
            os.mkdir(srt_files)
        except OSError:
            print('Creation of directory %s failed' %srt_files)

def merge_csv(path):
    print('Merging csv-files with transcriptions')
    csv_combined = pd.DataFrame()
    for entry in glob (path+'*.csv'):
        df = pd.read_csv(entry)
        csv_combined = csv_combined.append(df)

    csv_combined.to_csv('./merged_csv/Full_Transcript.csv', header=True, index=False, encoding='utf-8')
    print('All csv-files merged')

def change_encoding(srt):
    with io.open(srt, 'r', encoding='utf-8') as f:
        text = f.read()
        # process Unicode text
    with io.open(srt, 'w', encoding='utf-8') as f:
        f.write(text)

def convert_srt_to_csv(file):
    with open(file, 'r', encoding='utf-8') as h:
        sub = h.readlines()   #returns list of all lines

    re_pattern = r'[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3} --> [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}'
    regex = re.compile(re_pattern)
    # Get start times
    times = list(filter(regex.search, sub))
    end_times = [time.split('--> ')[1] for time in times] #returns a list
    start_times = [time.split(' ')[0] for time in times]  #returns a list

    # Get lines
    lines = [[]]
    for sentence in sub:
        if re.match(re_pattern, sentence):
            lines[-1].pop()
            lines.append([])
        else:
            lines[-1].append(sentence)

    lines = lines[1:]   #all text in lists

    column_names = ['id','start_times', 'end_times', 'sentence']
    df_text = pd.DataFrame(columns=column_names)

    df_text['start_times'] = start_times
    df_text['end_times'] = end_times
    df_text['sentence'] = [" ".join(i).replace('\n', '') for i in lines]
    df_text['end_times'] = df_text['end_times'].replace(r'\n', '', regex=True)

    df_text['id'] = np.arange(len(df_text))
    id_extension = os.path.basename(file).replace('.srt', '_')
    id_extension = id_extension.replace(' ', '_')
    id_extension = id_extension.replace('-', '_')
    id_extension = id_extension.replace('.', '_')
    id_extension = id_extension.replace('__', '_')
    id_extension = id_extension.replace('___', '_')
    df_text['id'] = id_extension +  df_text['id'].map(str)
    file_extension = id_extension[:-1]

    def convert_to_ms(time):
        h_ms = int(time[:2])*3600000
        m_ms = int(time[3:5])*60000
        s_ms = int(time[6:8])*1000
        ms = int(time[9:12])
        ms_total = h_ms + m_ms + s_ms + ms
        return(ms_total)

    def conv_int(start):
        new_start = int(start)
        return(new_start)

    df_text['start_times'] = df_text['start_times'].apply(convert_to_ms)
    df_text['end_times'] = df_text['end_times'].apply(convert_to_ms)
    df_text['start_times'] = df_text['start_times'].apply(conv_int)
    df_text.to_csv('./ready_for_slice/' + file_extension + '.csv', index=False, header=True, encoding='utf-8-sig')

def wmv_to_wav(entry):
    video = VideoFileClip(entry)
    audio = video.audio
    filename = os.path.basename(entry)
    filename = filename.replace(' ', '_')
    filename = filename.replace('-', '_')
    filename = filename.replace('.', '_')
    filename = filename.replace('__', '_')
    filename = filename.replace('___', '_')
    filename = filename[:-4] + '.wav'
    audio.write_audiofile('./audio/' +filename)

def mp4_to_wav(entry):
    video = VideoFileClip(entry)
    #extract audio from video
    audio = video.audio
    filename = os.path.basename(entry)
    filename = filename.replace(' ', '_')
    filename = filename.replace('-', '_')
    filename = filename.replace('.', '_')
    filename = filename.replace('__', '_')
    filename = filename.replace('___', '_')
    filename = filename[:-4] + '.wav'
    #filename = filename[:-4]+'.wav'
    #filename = filename[:10] + '_' + filename[-9:]
    audio.write_audiofile('./audio/' +filename)

def pre_process_audio(audio_path):
    path_audio_processed = './ready_for_slice/'
    if not os.path.exists(path_audio_processed):
        try:
            os.mkdir(path_audio_processed)
        except OSError:
            print('Creation of directory %s failed' %path_audio_processed)
        else:
            print('Successfully created the directory %s' %path_audio_processed)
    start_sub = time.time()
    n = 0
    print('Downsampling wav files...')
    for file in os.listdir(audio_path):
        if(file.endswith('.wav')):
            try:
                nameSolo_1 = file.rsplit('.', 1)[0]
                y, s = librosa.load(audio_path + file, sr=16000) # Downsample 44.1kHz to 8kHz
                sf.write(path_audio_processed + nameSolo_1 + '.wav', y, s)
                n = n+1
                print('File ', n , ' completed:', nameSolo_1)
            except EOFError as error:
                next

    s = 0
    print('Changing bit pro sample...')
    for file in os.listdir(path_audio_processed):
        if(file.endswith('.wav')):
            try:
                nameSolo_2 = file.rsplit('.', 1)[0]
                #nameSolo_2 = nameSolo_2.replace('')
                data, samplerate = sf.read(path_audio_processed + file)
                sf.write(path_audio_processed + nameSolo_2 + '.wav', data, samplerate, subtype='PCM_16')
                s = s + 1
                print('File ' , s , ' completed')
            except EOFError as error:
                next

    end_sub = time.time()
    print('The script took ', end_sub-start_sub, ' seconds to run')
    
def create_DS_csv (path):
    print('Extracting filepath and -size for every .wav file in ./sliced_audio')
    data = pd.DataFrame(columns=['file_name', 'duration'])
    df = pd.DataFrame(columns=['file_name', 'duration'])

    for entry in glob(path +'*.wav'):
        filename = os.path.basename(entry)
        with contextlib.closing(wave.open(entry, 'rb')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        df['file_name'] = [filename]
        df['duration'] = [duration]
        data = data.append(df)
    data.to_csv('./merged_csv/Filepath_Filesize.csv', header=True, index=False, encoding='utf-8')

def split_files(item, wav_item):
    song = AudioSegment.from_wav(wav_item)
    df = pd.read_csv(item)

    def audio_split(df):
        split = song[df['start_times']:df['end_times']]
        split.export('./sliced_audio/' + df['id'] + '.wav', format ='wav')

    df.apply(audio_split, axis=1)

def merge_transcripts_and_wav_files(transcript_path, DS_csv):
    df_final = pd.DataFrame()
    df_transcripts = pd.read_csv(transcript_path)
    df_files = pd.read_csv(DS_csv)
    def remove_path(path):
        path = path.split('/')[-1]
        return path
    df_files['id'] = df_files['file_name'].apply(remove_path)
    #filter out duration of less than 10 seconds
    def convert(duration):
        time = float(duration)
        return time
    df_files['duration'] = df_files['duration'].apply(convert)
    df_files = df_files[df_files['duration']<10.00]
    #drop unnecessary columns
    df_transcripts.drop(['start_times','end_times'], axis=1, inplace=True)
    df_files.drop(['duration'], axis=1, inplace=True)
    df_files['id'] = df_files['id'].replace('.wav', '', regex=True)
    #merge on column id
    df_final = pd.merge(df_transcripts, df_files, on='id')
    df_final.drop(['id'], axis=1, inplace=True)
    #rearrange columns
    df_final = df_final[['file_name', 'sentence']]
    df_final.to_csv('./final_csv/metadata.csv', header=True, index=False, encoding='utf-8')
    
    create_directories()
    
print("Put your video or audio files into the audio folder and srt files into the srt_files folder when you're ready...")

start_time = time.time()
srt_path = './srt_files'
audio_path = './audio/'
srt_counter = len(glob('./srt_files/' + '*.srt'))

#Extracting information from srt-files to csv
print('Extracting information from srt_file(s) to csv_files')
for file in glob('./srt_files/*.srt'):
    convert_srt_to_csv(file)
print('%s-file(s) converted and saved as csv-files to ./csv' %srt_counter)
print('---------------------------------------------------------------------')

#extract audio (wav) from mp4
for entry in glob('./audio/*.mp4'):
    mp4_to_wav(entry)
print('MP4 to WAV convert complete')
print('---------------------------------------------------------------------')


#Pre-process audio for folder in which wav files are stored
pre_process_audio(audio_path)
print('Pre-processing of audio files is complete.')
print('---------------------------------------------------------------------')

print('Slicing audio according to start- and end_times of transcript_csvs...')
for item in glob('./ready_for_slice/*.csv'):
    wav_item = item.replace('.csv','.wav')
    if os.path.exists(wav_item):
        split_files(item, wav_item)
    else:
        next
wav_counter = len(glob('./sliced_audio/' + '*.wav'))
print('Slicing complete. {} files in dir "sliced_audio"'.format(wav_counter))
print('---------------------------------------------------------------------')

create_DS_csv('./sliced_audio/')

#now join all seperate csv files
merge_csv('./ready_for_slice/')
print('Merged csv with all transcriptions created.')
print('---------------------------------------------------------------------')
transcript_path = './merged_csv/Full_Transcript.csv'
DS_path = './merged_csv/Filepath_Filesize.csv'
merge_transcripts_and_wav_files(transcript_path, DS_path)
