import librosa
import librosa.feature as lf
from numpy.core.arrayprint import format_float_scientific
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

featureList = []
foreda = []

def buildFeatEx(file): 
    """build features: MFCC, RMS, Spectral centroid, spectral bandwidth

    Args:
        file ([string]): [path of file from which audio data will be read for feature extraction]
    """
    pca = PCA(n_components = 1)
    y, sr = librosa.core.load(file)
    _featureList = []
    _foreda = []
    
    # features
    mfcc = lf.mfcc(y, sr)
    rms = lf.rms(y, sr)
    scent = lf.spectral_centroid(y,sr)
    sband = lf.spectral_bandwidth(y, sr)

    # PCA fit for MFCC - dimensionality reduction
    _mfcc = np.transpose(pca.fit_transform( mfcc ))

    # building featureList
    _featureList.extend(_mfcc.tolist()[0])
    _featureList.extend([np.mean(rms)])
    _featureList.extend([np.mean(scent)])
    _featureList.extend([np.mean(sband)])

    foreda.append(_mfcc.tolist()[0])
    foreda.append(rms)
    foreda.append(scent)
    foreda.append(sband)

    featureList.append(_featureList)
    # foreda.append(_mfcc, rms, scent, sband)
    print("{} done".format(file))

data = pd.read_csv('audio-split-annotations-update-2.csv')
# allFiles = data['id']
source = "split-2"
# print(allFiles)
# for i in range(data.shape[0]):
for i in range(data.shape[0]):
    f = data.loc[i, '0']
    gender = data.loc[i, '1']
    age = data.loc[i, '2']
    buildFeatEx("./{}/{}".format(source, f))


featex = pd.DataFrame([featureList])
featex = featex.transpose()
featex.to_csv('collected-annotated-2.csv')

foredadf = pd.DataFrame([foreda])
# foredadf = foredadf.transpose()
foredadf.to_csv('eda-2.csv')


# print(np.mean(a=mMFCC, axis=1))
# print(mMFCC, file=open("maleMFCC.csv", "w"))
# print(fMFCC, file=open("femaleMFCC.csv", "w"))

# print(*mRMS, file=open("maleRMS.csv", "w"))
# print(*fRMS, file=open("femaleRMS.csv", "w"))
# print(*mscent, file=open("maleScent.csv", "w"))
# print(*fscent, file=open("femaleScent.csv", "w"))
# print(*msband, file=open("maleSband.csv", "w"))
# print(*fsband, file=open("femaleSband.csv", "w"))
