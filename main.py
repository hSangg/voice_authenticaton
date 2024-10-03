import numpy as np
import librosa
from hmmlearn import hmm
import os

def extract_mfcc(audio_path, n_mfcc=13):
    audio, sr = librosa.load(audio_path, sr=16000)  
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    mfcc_delta = librosa.feature.delta(mfcc_features)
    mfcc_delta2 = librosa.feature.delta(mfcc_features, order=2)
    
    combined_mfcc = np.vstack([mfcc_features, mfcc_delta, mfcc_delta2])
    
    return combined_mfcc.T  

def train_hmm(mfcc_features, n_components=5):
    model = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=100)
    model.fit(mfcc_features)  
    return model

def authenticate_user(test_audio_path, user_hmm):
    mfcc_features = extract_mfcc(test_audio_path)
    score = user_hmm.score(mfcc_features)
    return score

train_data_directory = 'train_data/'  
test_data_directory = 'test_data/' 

user_hmms = {}
for user in os.listdir(train_data_directory):
    user_audio_files = [f for f in os.listdir(os.path.join(train_data_directory, user)) if f.endswith('.wav')]
    user_mfcc = np.concatenate([extract_mfcc(os.path.join(train_data_directory, user, f)) for f in user_audio_files])
    
    user_hmm = train_hmm(user_mfcc)
    user_hmms[user] = user_hmm  

test_user = 'user1'  
test_audio_path = os.path.join(test_data_directory, test_user, 'test.wav')

best_score = None
best_user = None

for user, user_hmm in user_hmms.items():
    score = authenticate_user(test_audio_path, user_hmm)
    
    if best_score is None or score > best_score:
        best_score = score
        best_user = user

print(f"Best match: {best_user} with score: {best_score}")
