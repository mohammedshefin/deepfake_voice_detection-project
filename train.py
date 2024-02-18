#importing necessary libraries
import pandas as pd
import numpy as np
import os
import sys
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from IPython.display import Audio
import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
  


voice_path='Project_Dataset/Data/'

voice_list = os.listdir('Project_Dataset/Data')
print(voice_list)

_file_label=[]
_file_path=[]
for dir in voice_list:
    get_dir = os.listdir(voice_path + dir)
    for voice in get_dir:
     
        _file_path.append(voice_path + dir + '/' + voice)
        _file_label.append(dir)


#creating dataframe
label_df = pd.DataFrame(_file_label, columns=['Label'])
path_df = pd.DataFrame(_file_path, columns=['Path'])

#concatenating columns in the dataframe
merged_df = pd.concat([label_df, path_df], axis=1)
print(merged_df)



#feature extraction from an audio 
def _extract__Features_(_data):
    # Zero crossing rate
    _result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=_data).T, axis=0)
    _result=np.hstack((_result, zcr)) 

    # Chroma (chromagram)
    stft = np.abs(librosa.stft(_data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    _result = np.hstack((_result, chroma_stft)) 

   #mfcc
    mfcc = np.mean(librosa.feature.mfcc(y=_data, sr=sample_rate).T, axis=0)
    _result = np.hstack((_result, mfcc)) 

    # RMS Value
    rms = np.mean(librosa.feature.rms(y=_data).T, axis=0)
    _result = np.hstack((_result, rms))

   #melspectogram
    mel = np.mean(librosa.feature.melspectrogram(y=_data, sr=sample_rate).T, axis=0)
    _result = np.hstack((_result, mel)) # stacking horizontally
    
    return _result

def _get__Features_(path):
    
    _data, sample_rate = librosa.load(path, duration=2.5, offset=0.6, sr=None)
    print(path)
    
   
    res1 = _extract__Features_(_data)
    _result = np.array(res1)
    
     #augmr
    _noise___data = _noise_(_data)
    res2 = _extract__Features_(_noise___data)
    _result = np.vstack((_result, res2)) # stacking vertically
    
    # _data with stretching and pitching
    new__data = stretch(_data)
    _data_stretch_pitch = pitch(new__data, sample_rate)
    res3 = _extract__Features_(_data_stretch_pitch)
    _result = np.vstack((_result, res3)) # stacking vertically
    
    return _result

def _noise_(_data):
    _noise__amp = 0.035*np.random.uniform()*np.amax(_data)
    _data = _data + _noise__amp*np.random.normal(size=_data.shape[0])
    return _data

def stretch(_data, rate=0.8):
    return librosa.effects.time_stretch(_data, rate)

def pitch(_data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(_data, sampling_rate, pitch_factor)

get_show=merged_df.Path
print(get_show)
path = np.array(merged_df.Path)[1]
_data, sample_rate = librosa.load(path)

#data division
X, Y = [], []
for path, label in zip(merged_df.Path, merged_df.Label):
    try:
        feature = _get__Features_(path)
        for fe in feature:
            X.append(fe)
            Y.append(label)
    except:
        pass

# import pickle

#save as pickle file
pickle.dump(X,open('X.pkl','wb'))
pickle.dump(Y,open('Y.pkl','wb'))

#load pickle file
X=pickle.load(open('X.pkl','rb'))
Y=pickle.load(open('Y.pkl','rb'))


_Features = pd.DataFrame(X)
_Features['labels'] = Y

_Features.to_csv('Test/_Features.csv', index=False)
print(_Features.head())
print(_Features['labels'].value_counts())
_Features['labels']=_Features['labels'].replace("human_voice",0)
_Features['labels']=_Features['labels'].replace("deepfake_voice",1)
print("*********************************")
print(_Features['labels'].value_counts())

X = _Features.iloc[: ,:-1].values
Y = _Features['labels'].values

print(X)
print(Y)

# #perform label encoding
# encoder = OneHotEncoder()
# Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

#Train-Test splitting
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True,test_size=0.2)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#perform standardization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
pickle.dump(scaler,open('Project_Saved_Models/scaler.pkl','wb'))


#applying dimension expansion
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


from model import model_cnn_lstm

model=model_cnn_lstm(x_train)




#saving the model(checkpoint)
checkpoint=ModelCheckpoint("Project_Saved_Models/cnn_lstm_model.h5",monitor="accuracy",save_best_only=True,verbose=1)#when training deep learning model,checkpoint is "WEIGHT OF THE MODEL"
#Training
history=model.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_test, y_test), callbacks=[checkpoint])



#plot accuracy and loss 
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('Project_Extra/acc_plot.png')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('Project_Extra/loss_plot.png')
plt.show()
pi