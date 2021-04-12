import librosa
import librosa.feature as lf
import os
import numpy as np
from numpy.core.fromnumeric import mean
from tensorflow import keras, reshape
from datasetAnnotation import getAllFiles
from sklearn.decomposition import PCA

def concat(prefix, string):
    return prefix+string

def extractFeatures(allFiles):
    featureList = ['mfcc', 'rms', 'scent', 'sband']
    featureFns = [lf.mfcc, lf.rms, lf.spectral_centroid, lf.spectral_bandwidth]    
    Nfunc = len(featureList)

    dump = open('dump.txt', 'w')
    fileDict = {}
    for f in allFiles:
        print(f, file = dump)
        print('-----------------', file = dump)
        y, sr = librosa.load(f)

        feature = {}

        _feature = list(
                map(
                    lf.mfcc,
                    [y],
                    [sr]
                )
            )
        print(np.array(_feature)[0].shape, file = dump)
        feature['mfcc'] = np.array(_feature)[0]

        for i in range(1, Nfunc):
            _feature = list(
                map(
                    featureFns[i],
                    [y],
                    [sr]
                )
            )
            print(np.asarray(_feature)[0].shape, file = dump)
            feature[featureList[i]] = np.asarray(_feature)[0]
        
        fileDict[f] = feature
        print('', file = dump)
        # break
    return fileDict

def meanifyFeatures(fileFeature, readFile, featureList, dumpFile):
    # meanFeatures = {}
    # _file = open(dumpFile, 'w')
    # for k in fileFeature:
    feature = fileFeature[readFile]
    for f in featureList:
        meanFeature = list(
            map(
                mean,
                feature[f]
            )
        )
        print(*meanFeature, end = ' ', file = dumpFile)
    print('', file = dumpFile)


def convolute1D(F):
    # hard coded
    convolutionSize = 65

    croppedF = F[:, :convolutionSize]
    croppedF.shape  = (1, ) + croppedF.shape
    print("before==>", croppedF.shape)
    y = keras.layers.Conv1D(
        20, 5, 
        activation='relu',
        input_shape = croppedF.shape[2:]
    )(croppedF)
    print("after==>", y.shape)
    # print(reshape(y, [-1]))
    y = reshape(y, [-1])
    return y.numpy


def convulifyFeatures(fileFeature, readFile, dumpFile, featureList = 'mfcc'):
    # _file = open(dumpFile, 'w')
    feature = fileFeature[readFile]
    # convFeature = convolute1D(feature[featureList])
    pca = PCA(n_components = 1)
    convFeature = np.transpose(pca.fit_transform( feature[featureList] ))
    print(*convFeature, end = ' ', file = dumpFile)


def makeFeatures(fileFeature):
    # files = fileFeature[]
    _dumpFile = open('conv-feat-dump.txt', 'w')
    for f in fileFeature:
        convulifyFeatures(fileFeature, f, _dumpFile )
        meanifyFeatures(fileFeature, f, ['rms', 'scent', 'sband'], _dumpFile)

if __name__ == "__main__":
    allFiles = getAllFiles('./split', filetype = 'wav')
    f = extractFeatures(allFiles)
    # meanifyFeatures(f, 'output.txt')
    makeFeatures(f)
