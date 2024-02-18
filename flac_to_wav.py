import soundfile
import numpy
import pandas as pd

#read dataframe
human_voice=pd.read_csv("Project_Dataset/human_voice.csv")
Name_human_voice = human_voice["Name"].tolist()
print(Name_human_voice)
print(type(Name_human_voice))
print(len(Name_human_voice))

for i in Name_human_voice:
	name=i.split('.')[0]
	flac_file = r'{}'.format(i) #flac audio path
	# print(flac_file)
	audio, sr = soundfile.read(flac_file)
	soundfile.write('Project_Dataset/Data/human_voice/'+name+'.wav', audio, sr, 'PCM_16')


deepfake_voice=pd.read_csv("Project_Dataset/deepfake_voice.csv")
Name_deepfake_voice = deepfake_voice["Name"].tolist()[:18452]
print(Name_deepfake_voice)
print(type(Name_deepfake_voice))
print(len(Name_deepfake_voice))


for k in Name_deepfake_voice:
	name1=k.split('.')[0]
	flac_file1 = r'{}'.format(k) #flac audio path
	# print(flac_file1)
	audio1, sr1 = soundfile.read(flac_file1)
	soundfile.write('Project_Dataset/Data/deepfake_voice/'+name1+'.wav', audio1, sr1, 'PCM_16')

# wav_file = r'drama-02-005.flac'
# audio, sr = soundfile.read(wav_file)
# soundfile.write('drama-02-005.wav', audio, sr, 'PCM_16
