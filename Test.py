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

import mxnet as mx


# In[0]: -- Please supply test.csv via data/
data_test = mx.io.CSVIter(data_csv="data/test.csv", data_shape=(30, 64, 64), batch_size=1)


# In[1]:
print ( 'systole evaluation cycle' )
systole_model = mx.model.FeedForward.load('ejection_fraction_detection_systole_model', 51)
systole_model.fit(X=data_train, eval_metric = mx.metric.np(CRPS))
systole_prob = systole_model.predict(data_test)
systole_result = accumulate_result("./validate-label.csv", systole_prob)

# In[2]:
print ( 'diastole evaluation cycle' )
diastole_model = mx.model.FeedForward.load('ejection_fraction_detection_diastole_model', 52)
diastole_model.fit(X=data_train, eval_metric = mx.metric.np(CRPS))
diastole_prob = diastole_model.predict(data_test)
diastole_result = accumulate_result("./validate-label.csv", diastole_prob)



# # Generate Submission on evaluation boundaries a priori
# In[3]:
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


# In[5]:
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

# In[6]:
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
    fo.writerow(out)
f.close()
