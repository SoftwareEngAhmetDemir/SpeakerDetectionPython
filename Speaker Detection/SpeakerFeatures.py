import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc


def calculate_delta(array):
    """Calculate and returns the delta of given feature vector matrix"""

    rows, cols = array.shape
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i - j:
                first = rows - 1
            else:
                second = i + j
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
    return deltas


def extract_features(audio, rate):
    """extract 20 dim mfcc features from an audio, performs CMS and combines
    delta to make it 40 dim feature vector"""

    mac_feat = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, appendEnergy=True)
    mac_feat = preprocessing.scale(mac_feat)
    delta = calculate_delta(mac_feat)
    print(delta)
    combined = np.hstack((mac_feat, delta))
    return combined
