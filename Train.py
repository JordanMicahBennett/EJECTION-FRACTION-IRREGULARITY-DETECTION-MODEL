"""Training script, this is converted from a ipython notebook
"""

import os
import csv
import sys
import numpy as np
import mxnet as mx 
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# In[2]:

import mxnet as mx


def conv_factory(data, num_filter, kernel, stride, pad, act_type = 'relu', conv_type = 0):
    if conv_type == 0:
        conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
        bn = mx.symbol.BatchNorm(data=conv)
        act = mx.symbol.Activation(data = bn, act_type=act_type)
        return act
    elif conv_type == 1:
        conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
        bn = mx.symbol.BatchNorm(data=conv)
        return bn

def residual_factory(data, num_filter, dim_match):
    if dim_match == True: # if dimension match
        identity_data = data
        conv1 = conv_factory(data=data, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), act_type='relu', conv_type=0)

        conv2 = conv_factory(data=conv1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), conv_type=1)
        new_data = identity_data + conv2
        act = mx.symbol.Activation(data=new_data, act_type='relu')
        return act
    else:
        conv1 = conv_factory(data=data, num_filter=num_filter, kernel=(3,3), stride=(2,2), pad=(1,1), act_type='relu', conv_type=0)
        conv2 = conv_factory(data=conv1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), conv_type=1)

        # adopt project method in the paper when dimension increased
        project_data = conv_factory(data=data, num_filter=num_filter, kernel=(1,1), stride=(2,2), pad=(0,0), conv_type=0)
        new_data = project_data + conv2
        act = mx.symbol.Activation(data=new_data, act_type='relu')
        return act

def residual_net(data, n):
    #fisrt 2n layers
    for i in range(n):
        data = residual_factory(data=data, num_filter=16, dim_match=True)

    #second 2n layers
    for i in range(n):
        if i==0:
            data = residual_factory(data=data, num_filter=32, dim_match=False)
        else:
            data = residual_factory(data=data, num_filter=32, dim_match=True)

    #third 2n layers
    for i in range(n):
        if i==0:
            data = residual_factory(data=data, num_filter=64, dim_match=False)
        else:
            data = residual_factory(data=data, num_filter=64, dim_match=True)

    return data

def get_symbol(num_classes=600):
    conv = conv_factory(data=mx.symbol.Variable(name='data'), num_filter=16, kernel=(3,3), stride=(1,1), pad=(1,1), act_type='relu', conv_type=0)
    n = 2 # set n = 3 means get a model with 3*6+2=20 layers, set n = 9 means 9*6+2=56 layers
    resnet = residual_net(conv, n) #
    pool = mx.symbol.Pooling(data=resnet, kernel=(7,7), pool_type='avg')
    flatten = mx.symbol.Flatten(data=pool, name='flatten')
    fc = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes,  name='fc1')
    return mx.symbol.LogisticRegressionOutput(data=fc, name='softmax')



def CRPS(label, pred):
    """ Custom evaluation metric on CRPS.
    """
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1] - 1):
            if pred[i, j] > pred[i, j + 1]:
                pred[i, j + 1] = pred[i, j]
    return np.sum(np.square(label - pred)) / label.size


# In[3]:

def encode_label(label_data):
    """Run encoding to encode the label into the CDF target.
    """
    systole = label_data[:, 1]
    diastole = label_data[:, 2]
    systole_encode = np.array([
            (x < np.arange(600)) for x in systole
        ], dtype=np.uint8)
    diastole_encode = np.array([
            (x < np.arange(600)) for x in diastole
        ], dtype=np.uint8)
    return systole_encode, diastole_encode

def encode_csv(label_csv, systole_csv, diastole_csv):
    systole_encode, diastole_encode = encode_label(np.loadtxt(label_csv, delimiter=","))
    np.savetxt(systole_csv, systole_encode, delimiter=",", fmt="%g")
    np.savetxt(diastole_csv, diastole_encode, delimiter=",", fmt="%g")

# Write encoded label into the target csv
# We use CSV so that not all data need to sit into memory
# You can also use inmemory numpy array if your machine is large enough
encode_csv("./train-label.csv", "./train-systole.csv", "./train-diastole.csv")


# # Training the systole net

# In[4]:
print ( 'systole cycle' )
network = get_symbol()
batch_size = 5
devs = [mx.gpu(0)]
data_train = mx.io.CSVIter(data_csv="./train-64x64-data.csv", data_shape=(30, 64, 64),
                           label_csv="./train-systole.csv", label_shape=(600,),
                           batch_size=batch_size)

data_validate = mx.io.CSVIter(data_csv="./validate-64x64-data.csv", data_shape=(30, 64, 64),
                              batch_size=1)
 
systole_model = mx.model.FeedForward(ctx=devs,
        symbol             = network,
        num_epoch          = 51,
        learning_rate      = 0.001,
        wd                 = 0.00001,
        momentum           = 0.9)


systole_model.fit(X=data_train, eval_metric = mx.metric.np(CRPS))


# # Predict systole

# In[5]:

systole_prob = systole_model.predict(data_validate)


# # Training the diastole net

# In[6]:
print ( 'diastole cycle' )
network = get_symbol()
batch_size = 5
devs = [mx.gpu(0)]
data_train = mx.io.CSVIter(data_csv="./train-64x64-data.csv", data_shape=(30, 64, 64),
                           label_csv="./train-diastole.csv", label_shape=(600,),
                           batch_size=batch_size)

diastole_model = mx.model.FeedForward(ctx=devs,
        symbol             = network,
        num_epoch          = 52,
        learning_rate      = 0.001,
        wd                 = 0.00001,
        momentum           = 0.9)


diastole_model.fit(X=data_train, eval_metric = mx.metric.np(CRPS))


# # Predict diastole

# In[7]:

diastole_prob = diastole_model.predict(data_validate)


# # Generate Submission

# In[8]:

def accumulate_result(validate_lst, prob):
    sum_result = {}
    cnt_result = {}
    size = prob.shape[0]
    fi = csv.reader(open(validate_lst))
    for i in range(size):
        line = fi.__next__() # Python2: line = fi.next()
        idx = int(line[0])
        if idx not in cnt_result:
            cnt_result[idx] = 0.
            sum_result[idx] = np.zeros((1, prob.shape[1]))
        cnt_result[idx] += 1
        sum_result[idx] += prob[i, :]
    for i in cnt_result.keys():
        sum_result[i][:] /= cnt_result[i]
    return sum_result


# In[9]:

systole_result = accumulate_result("./validate-label.csv", systole_prob)
diastole_result = accumulate_result("./validate-label.csv", diastole_prob)


# In[10]:

# we have 2 person missing due to frame selection, use udibr's hist result instead
def doHist(data):
    h = np.zeros(600)
    for j in np.ceil(data).astype(int):
        h[j:] += 1
    h /= len(data)
    return h
train_csv = np.genfromtxt("./train-label.csv", delimiter=',')
hSystole = doHist(train_csv[:, 1])
hDiastole = doHist(train_csv[:, 2])


# In[11]:

def submission_helper(pred):
    p = np.zeros(600)
    pred.resize(p.shape)
    p[0] = pred[0]
    for j in range(1, 600):
        a = p[j - 1]
        b = pred[j]
        if b < a:
            p[j] = a
        else:
            p[j] = b
    return p



# In[12]:
print ( 'submission cycle' )
fi = csv.reader(open("data/sample_submission_validate.csv"))
f = open("submission.csv", "w")
fo = csv.writer(f, lineterminator='\n')
fo.writerow(fi.__next__())
for line in fi:
    idx = line[0]
    key, target = idx.split('_')
    key = int(key)
    out = [idx]
    if key in systole_result:
        if target == 'Diastole':
            out.extend(list(submission_helper(diastole_result[key])))
        else:
            out.extend(list(submission_helper(systole_result[key])))
    else:
        print("Miss: %s" % idx)
        if target == 'Diastole':
            out.extend(hDiastole)
        else:
            out.extend(hSystole)
    fo.writerow(out)
f.close()
