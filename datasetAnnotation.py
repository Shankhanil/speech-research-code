import pandas as pd
import os
import librosa
import librosa.feature as lf
# import os
import numpy as np
from numpy.core.fromnumeric import mean


def getAllFiles(path, filetype = 'wav'):
    """[Traverses a dataset directory and retuns all (*wav) files]

    Args:
        path ([type]): [description]
        filetype (str, optional): [Filetyle to search and list]. Defaults to 'wav'.

    Returns:
        [list]: [list of path of all (*wav) files from the path directory]
    """    
    allFiles = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".{}".format(filetype)):
                allFiles.append(os.path.join(root, file))
    return allFiles

def generateAnnotations(allFiles):
    """Generate gender annotation from files

    Args:
        allFiles ([list]): [list of all audio files in the dataset folder]

    Returns:
        [None]: [dumps the dataset into csv file]
    """    
    audioID = []
    gender = []
    age = []

    for f in allFiles:
        _f = f.split('\\')
        # print(_f)
        audioID.append(_f[1] + _f[2].split('.')[0])
        gender.append( _f[1].split('-')[1][0])
        age.append(0)
    dataset = pd.DataFrame(
        {
            'audioID': audioID,
            'gender' : gender,
            'age'    : age
        } 
    )
    return dataset.to_csv('dataset.csv')
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
    return fileDict

def meanifyFeatures(fileFeature, dumpFile):
    _file = open(dumpFile, 'w')
    for k in fileFeature:
        feature = fileFeature[k]
        for f in feature:
            meanFeature = list(
                map(
                    mean,
                    feature[f]
                )
            )
            print(meanFeature, end = ' ', file = _file)
        print('', file = _file)

if __name__ == "__main__":
    allFiles = getAllFiles('./split')
    generateAnnotations(allFiles)
    f = extractFeatures(allFiles)
    meanifyFeatures(f, 'output.txt')
    

