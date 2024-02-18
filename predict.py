#importing ncessary libraries
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
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
#from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import pickle

#loading standardscaler
scaler=pickle.load(open('Project_Saved_Models/scaler.pkl','rb'))

#extract features
def _extract__Features_(_data,sample_rate):
    # Zeor crossing rate
    _result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=_data).T, axis=0)
    _result=np.hstack((_result, zcr)) 

    # Chroma
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
    
    _data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    print(path)
    
   
    res1 = _extract__Features_(_data,sample_rate)
    _result = np.array(res1)

    
    return _result


#load the trained model 
loaded_model=load_model('Project_Saved_Models/cnn_lstm_model.h5')


if __name__=="__main__":
    from tkinter.filedialog import askopenfilename
    # dict1={0:'human_voice', 1:'deepfake_voice'}
    path=askopenfilename()
    #feature extaction
    feat=_get__Features_(path)
    feat=np.array([feat])
    #perform standardization
    feat=scaler.transform(feat)
    #expand dimension
    feat = np.expand_dims(feat, axis=2)
   
   #prediction using the model
    pred=loaded_model.predict(feat)[0]
    pred=pred[0]
    print("PRED : ",pred)

    if pred<=0.5:
        print("Human Voice")
    if pred>0.5:
        print("Deepfake Voice")
