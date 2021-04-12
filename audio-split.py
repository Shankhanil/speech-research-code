from pydub import AudioSegment
import math, os
import pandas as pd

dataset = []

class SplitWavAudioMubin():
    def __init__(self, folder, dump, filename, age, gender):
        self.folder = folder
        self.dump = dump
        self.filename = filename
        self.filepath = folder + '\\' + filename
        
        self.audio = AudioSegment.from_wav(self.filepath)

        self.age = age
        self.gender = gender
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_sec, to_sec, split_filename):
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = self.audio[t1:t2]
        if split_audio.duration_seconds >= 3:
            print('too short')
            split_audio.export(self.dump + '\\' + split_filename, format="wav")
            dataset.append([
                split_filename,
                self.gender,
                self.age
            ])
            # print(self.age)
        
    def multiple_split(self, sec_per_split):
        total_secs = math.ceil(self.get_duration())
        for i in range(0, total_secs, sec_per_split):
            split_fn = (self.filename).split('.')[0] + '_' + str(i) + "_"+str(self.gender)+ str(self.age) + ".wav"
            # print("time reamining", total_secs - i)
            if (total_secs - i+sec_per_split >= 3):
                self.single_split(i, i+sec_per_split, split_fn)
                print(str(i) + ' Done')
            # if i == total_secs - sec_per_split:
        print('All splited successfully')
        # pd.DataFrame(self.dataset).to_csv('res.csv')
        # return d
folder = '.\\audio'
dump = '.\\split-2'
# file = "04d9a1b5-5fd6-431a-97b0-70c516147a22.wav"
for (dirpath, dirnames, file) in os.walk(folder):
    print(file)
    break
data = pd.read_csv('collected-age-gender-details.csv')
print(data.shape)
# i = 0
N = data.shape[0]
for i in range(N):
    _data = data.loc[i]
    age, gender = _data['age'], _data['gender']
    f = "{}.wav".format(_data['id'])
    print(f, age, gender)
    # _f = f.split('.')[0]
    # _data = data[data['id'] == _f]
    # print(_data.shap)
    # print(type(_data.iloc[0]['gender']))
    # gender, age = _data.iloc[0]['gender'], _data.iloc[0]['age']
    split_wav = SplitWavAudioMubin(folder, dump, f, age = age, gender = gender)
    split_wav.multiple_split(sec_per_split=5)
    
pd.DataFrame(dataset).to_csv('audio-split-annotations-update-2.csv')