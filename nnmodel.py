from datetime import datetime
import numpy as np
from numpy.core.fromnumeric import mean
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras, random
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.layers.core import Dropout
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA


# global output

# output = open('../logs/nnoutput.txt')

def NNModel(X_train, X_test, y_train, y_test ):  
    """Neural network model to classify gender

    Args: Train and test batched
        X_train (Dataframe): 
        X_test (Dataframe: 
        y_train (Dataframe): 
        y_test (Dataframe: 

    Returns:
        [float]: [returns test loss]
    """    
    model = keras.Sequential(
        [
            # meanify layers
            # keras.layers.Dense(25, activation='sigmoid' ),
            # keras.layers.Dense(10, activation='sigmoid' ),
            keras.layers.Dropout(0.01),
            keras.layers.Dense(25, activation='relu' ),
            keras.layers.Dropout(0.01),
            keras.layers.Dense(10, activation='relu' ),
            keras.layers.Dropout(0.01),
            keras.layers.Dense(5, activation='relu' ),
            keras.layers.Dense( 1, activation='sigmoid' )

            # conv layers
            # keras.layers.Dropout(0.2),
            # keras.layers.Dense(50, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(l1=1e-2, l2=1e-2) ),
            # keras.layers.Dropout(0.2),
            # keras.layers.Dense(25, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(l1=1e-2, l2=1e-2) ),
            # # keras.layers.Dropout(0.2),
            # keras.layers.Dense( 1, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(l1=1e-2, l2=1e-2)  )
        ]
    )

    model.compile(
        optimizer='rmsprop', 
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )


    # tinker with the epoch
    epoch = 500
    history = model.fit(X_train, y_train, validation_split=0.1, shuffle=True, epochs=epoch)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2) # two axes on figure
    
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set(xlabel='epoch', ylabel='accuracy')
    ax1.legend(['test accuracy', 'validation accuracy'])
    
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set(xlabel='epoch', ylabel='loss')
    ax2.legend(['test loss', 'validation loss'])

    print(model.summary())    
    plt.show()
    
    # model save paths for meanify and conv
    savedModelPath = {
        'meanify': 'saved-models-meanify',
        'conv': 'saved-models-conv'
    }
    # keras.models.save_model(
    #     model=model, 
    #     save_format='tf', 

    #     # write the file path, 'conv' vs 'meanify'
    #     # filepath=savedModelPath['conv']
    #     filepath=savedModelPath['meanify']
    # )
    return model.evaluate(X_test, y_test )

def logToFile(output, file):
    """Log the output to a log file

    Args:
        output  ([string]): [output string to log]
        file    ([file]):   [log file]
    """    
    print( output, file=file )


if __name__ == "__main__":
    scaler = MinMaxScaler()
    pca = PCA()
    # lda = LinearDiscriminantAnalysis()
    logfile = open('nnoutput.txt', 'a')
    # logfile = open('logs\\nnoutput-conv.txt', 'a')

    # path = 'datasets\\gender-annotated-dataset.csv'
    # path = 'datasets\\feature-dataset-conv.csv'
    path = 'embedded-collected.csv'
    data = pd.read_csv(path)
    
    # mean22f = np.mean(data.iloc[:, 22][data.iloc[:, -1] == 'female'])
    # mean22m = np.mean(data.iloc[:, 22][data.iloc[:, -1] == 'male'  ])
    X = data.iloc[:, :-1]
    _X = pd.DataFrame(scaler.fit_transform(X))
    print(_X.isna().any())
    Y = data.iloc[:, -1]
    Y.replace('male', 0, inplace=True)
    # Y.replace('m', 0, inplace=True)
    Y.replace('female', 1, inplace=True)
    # Y.replace('f', 1, inplace=True)
    # exit()
    # _tX = pca.fit_transform(_X)
    # _tX = lda.fit_transform(_X, Y)
    # print(_tX.shape)

    X_train, X_test, y_train, y_test = train_test_split(_X, Y, test_size=0.15)

    logToFile(datetime.now(), logfile)
    logToFile('-------------------------', logfile)

    [summary, results] = NNModel(X_train, X_test, y_train, y_test)
    # logToFile(
    #     summary,
    #     logfile
    # )
    logToFile(
        results,
        logfile
    )
    # logToFile(
    #     NNModel(X_train, X_test, y_train, y_test),
    #     logfile
    # )