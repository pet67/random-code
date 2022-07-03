import sys

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import scipy
import scipy.io
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

from torch.autograd import Variable
import torch, torch.nn as nn
import torch.nn.functional as F
import matplotlib.image as mpimg

import scipy
import sklearn
import sklearn.metrics
import math

import numpy as np
import scipy
import scipy.signal
import random
import scipy
import math
import pandas as pd

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import pandas
import numpy as np

from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler

import math

import sklearn
import sklearn.metrics

import matplotlib
import matplotlib.pyplot as plt

import copy

import scipy
import scipy.signal



from IPython.display import display, clear_output

def downsample(X, Y, frequency, x_times):
    assert x_times in range(13)
    X = scipy.signal.decimate(X, x_times, axis=0)
    Y = scipy.signal.decimate(Y, x_times, axis=0)
    return X, Y, frequency / x_times


def high_pass_filtering(X, frequency, cutoff_frequency):
    b, a = scipy.signal.butter(3, cutoff_frequency * 1.0 / (frequency * 1.0 / 2), 'high')
    for i in range(X.shape[1]):
        X[:, i] = scipy.signal.filtfilt(b, a, X[:, i])
    return X


def norch_filtering(X, Y, frequency, remove_frequency):
    QUALITY = 30
    b, a = scipy.signal.iirnotch(remove_frequency * 1.0 / (frequency * 1.0 / 2), QUALITY)
    for i in range(X.shape[1]):
        X[:, i] = scipy.signal.filtfilt(b, a, X[:, i])
    return X, Y, frequency



def filename2target(filname):
    file_index = int(filname.split("_")[0])
    return FILEINDEX2TARGET[file_index]


FREQUENCY = 250

file = "//home/amplifier/common/pet67/ossadtchi-ml-test-bench/paper/1.mat"

data = scipy.io.loadmat(file)



def expand_target(Y):
    TARGETS_COUNT = 5
    Y_new = np.zeros((Y.shape[0], TARGETS_COUNT))
    for index, target in enumerate(Y):
        target = int(target)
        assert target in range(1, TARGETS_COUNT + 1), target
        Y_new[index, target - 1] = 1
    return Y_new

RIGHT_PAD = 1000

X = data["xraw"].transpose()[:-RIGHT_PAD]
Y = expand_target(np.round(data['stim']).astype("int").reshape((-1, 1))[:-RIGHT_PAD])

LAG_BACKWARD = 250
LAG_FOREWARD = 0

def make_lag_3D(X_3D, lag_backward, lag_forward, decimate=1):
    # TODO: probably this code can be rewritten shorter
    assert decimate > 0
    assert lag_backward >=0
    assert lag_forward >=0

    output_samples = X_3D.shape[0] - lag_backward - lag_forward
    channels = X_3D.shape[1]
    feature_per_channel = X_3D.shape[2]
    # TODO: CHECK IF ITS VALID
    features_backward = int(lag_backward / decimate)
    features_forward = int(lag_forward / decimate)
    output_features = feature_per_channel * (1 + features_forward + features_backward)

    X_output_3D = np.zeros((output_samples, channels, output_features))
    feature_index = 0
    for time_shift in range(decimate, lag_backward + decimate, decimate):
        feature_slice = slice(feature_index, feature_index + feature_per_channel)
        X_output_3D[:, :, feature_slice] = X_3D[lag_backward - time_shift:-lag_forward - time_shift]
        feature_index += feature_per_channel

    feature_slice = slice(feature_index, feature_index + feature_per_channel)
    X_output_3D[:, :, feature_slice] = X_3D[lag_backward:-lag_forward if lag_forward > 0 else None]  # cetntral point
    feature_index += feature_per_channel

    for time_shift in range(decimate, lag_forward + decimate, decimate):
        feature_slice = slice(feature_index, feature_index + feature_per_channel)
        X_output_3D[:, :, feature_slice] = X_3D[lag_backward + time_shift:-lag_forward + time_shift if lag_forward - time_shift > 0 else None]
        feature_index += feature_per_channel
    assert feature_index == output_features
    return X_output_3D


#X = high_pass_filtering(X, FREQUENCY, 5)
X = sklearn.preprocessing.scale(X)
X = make_lag_3D(X[:, :, None], LAG_BACKWARD, LAG_FOREWARD, 1)
Y = Y[LAG_BACKWARD:-LAG_FOREWARD if LAG_FOREWARD > 0 else None]


assert X.shape[0] == Y.shape[0], f"{X.shape[0]} != {Y.shape[0]}"

#X, Y, FREQUENCY = high_pass_filtering(X, Y, FREQUENCY, 10)
#X, Y, FREQUENCY = norch_filtering(X, Y, FREQUENCY, 50)

ntr = int(X.shape[0] * 0.5)

#CHANNELS = [0]


X_train_original = np.copy(X[:ntr])#[:, CHANNELS]
X_train = X[:ntr]#[:, CHANNELS]
Y_train = Y[:ntr]

X_test = X[ntr:]#[:, CHANNELS]
Y_test = Y[ntr:]

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class envelope_detector(nn.Module):
    def __init__(self, in_channels, channels_per_channel):
        super(self.__class__,self).__init__()
        self.FILTERING_SIZE = 50
        self.ENVELOPE_SIZE = 25
        self.CHANNELS_PER_CHANNEL = channels_per_channel
        self.OUTPUT_CHANNELS = self.CHANNELS_PER_CHANNEL * in_channels
        self.conv_filtering = nn.Conv1d(in_channels, self.OUTPUT_CHANNELS, bias=False, kernel_size=self.FILTERING_SIZE, groups=in_channels)
        self.conv_envelope = nn.Conv1d(self.OUTPUT_CHANNELS, self.OUTPUT_CHANNELS, kernel_size=self.ENVELOPE_SIZE, groups=self.OUTPUT_CHANNELS)
        self.conv_envelope.requires_grad = False
        self.pre_envelope_batchnorm = torch.nn.BatchNorm1d(self.OUTPUT_CHANNELS, affine=False)
        self.conv_envelope.weight.data = (torch.ones(self.OUTPUT_CHANNELS * self.ENVELOPE_SIZE) / self.FILTERING_SIZE).reshape((self.OUTPUT_CHANNELS, 1, self.ENVELOPE_SIZE))     
        self.relu = torch.nn.ReLU()
        self.intermidiate = None
        
        
    def forward(self, x):
        x = self.conv_filtering(x)
        #self.intermidiate = x.cpu().data.numpy()
        #x = self.pre_envelope_batchnorm(x)
        x = self.relu(x)
        #x = torch.abs(x)
        #x = self.conv_envelope(x)
        return x


class simple_net(nn.Module):
    def __init__(self, in_channels, output_channels, lag_backward, lag_forward):
        super(self.__class__,self).__init__()
        self.ICA_CHANNELS = 32
        self.CHANNELS_PER_CHANNEL = 1

        self.total_input_channels = self.ICA_CHANNELS# + in_channels
        self.lag_backward = lag_backward
        self.lag_forward = lag_forward

        self.final_out_features = 672  

        self.ica = nn.Conv1d(in_channels, self.ICA_CHANNELS, 1)

        self.detector = envelope_detector(self.total_input_channels, self.CHANNELS_PER_CHANNEL)
        self.features_batchnorm = torch.nn.BatchNorm1d(self.final_out_features, affine=False)
        self.unmixed_batchnorm = torch.nn.BatchNorm1d(self.total_input_channels, affine=False)

        self.wights_second = nn.Linear(self.final_out_features, output_channels)
        self.fuck_channels = None
        self.pre_out = None
        #self.softmax = torch.nn.Softmax()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):
        all_inputs = self.ica(inputs)
        self.fuck_channels = all_inputs.cpu().data.numpy()
        all_inputs = self.unmixed_batchnorm(all_inputs)

        detected_envelopes = self.detector(all_inputs)

        features  = detected_envelopes[:, :, ::10].contiguous()

        features = features.view(features.size(0), -1)
        features = self.features_batchnorm(features)
        self.pre_out = features.cpu().data.numpy()
        output = self.wights_second(features)
        return output


def data_generator(X, Y, batch_size, shuffle=True, infinite=True):
    assert len(X)==len(Y) or len(Y)==0
    total_lag = lag_backward + lag_forward
    all_batches = math.ceil((X.shape[0] )/batch_size)
    samples_in_last_batch = (X.shape[0]) % batch_size
    batch = 0
    random_core = np.arange(0, X.shape[0])
    while True:
        if shuffle:
            np.random.shuffle(random_core)
        for batch in range(all_batches):       
            batch_start = batch * batch_size
            batch_end = (batch + 1)*batch_size
            if batch_end >= len(random_core):
                batch_end = None
            batch_samples = random_core[batch_start : batch_end]

            batch_x = X[[batch_samples]]
            #batch_x = np.swapaxes(batch_x, 1, 2)

            if len(Y) > 0:
                batch_y = Y[[batch_samples]] 
                yield (batch_x, batch_y)
            else:
                yield batch_x
        
        if not infinite:
            break

batch_size = 100
lag_backward = LAG_BACKWARD
lag_forward = LAG_FOREWARD

model = simple_net(X_train.shape[1], Y_test.shape[1], lag_backward, lag_forward).cuda()


print("Trainable params: ",sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Total params: ",sum(p.numel() for p in model.parameters() if p.requires_grad))
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


loss_history = []
max_test_corr = 0

Y_test_sliced = Y_test[lag_backward:]

pbar = tqdm_notebook()

for x_batch, y_batch in data_generator(X_train, Y_train, batch_size, shuffle=True, infinite=True):
    #### Train
    y_batch = y_batch.argmax(axis=1)
    model.train()
    
    assert x_batch.shape[0] == y_batch.shape[0]
    x_batch = Variable(torch.FloatTensor(x_batch)).cuda()
    y_batch = Variable(torch.LongTensor(y_batch)).cuda()
    optimizer.zero_grad()

    y_predicted = model(x_batch)
    assert y_predicted.shape[0] == y_batch.shape[0]

    loss = loss_function(y_predicted, y_batch)

    loss.backward()
    optimizer.step()
    loss_history.append(np.mean(y_predicted.cpu().detach().numpy().argmax(axis=1) == y_batch.cpu().detach().numpy()))    
    pbar.update(1)
    eval_lag = min(100, len(loss_history))
    pbar.set_postfix(loss = np.mean(loss_history[-eval_lag:]), val_loss=max_test_corr)
    
    if len(loss_history) % 10000 == 0:
        break
    
    if len(loss_history) % 100 == 0:
        Y_predicted = []
        for x_batch, _ in data_generator(X_test, Y_test, batch_size, shuffle=False, infinite=False):
            #### Train
            model.eval()

            x_batch = Variable(torch.FloatTensor(x_batch)).cuda()
            y_predicted = model(x_batch).cpu().data.numpy()
            assert x_batch.shape[0]==y_predicted.shape[0]
            Y_predicted.append(y_predicted)

        Y_predicted = np.concatenate(Y_predicted, axis = 0)
        max_test_corr = np.mean(Y_predicted.argmax(axis=1) == Y_test.argmax(axis=1))
        
        print("Correlation   val", max_test_corr)

    if len(loss_history)>30000:
        break
