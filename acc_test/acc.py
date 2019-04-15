import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
from keras import backend as K


print('Analyse accuracy of: cb513_test_prob_'+str(sys.argv[1])+'.npy')

m1 = np.load('cb513_test_prob_'+str(sys.argv[1])+'.npy',)
#m1 = np.load('val_pred_20190412-102316-augmodel92_unet_768_cb513_raw_pred.npy')

print (m1.shape)


# warped_order_1 = ['NoSeq', 'H', 'E', 'L','T', 'S', 'G', 'B',  'I']
#                    0        1    2    3   4    5    6    7     8
#                  ['L',     'B', 'E', 'G','I', 'H', 'S', 'T', 'NoSeq'] # new order
order_list = [8,5,2,0,7,6,3,1,4]
labels = ['L', 'B', 'E', 'G','I', 'H', 'S', 'T', 'NoSeq']

m1p = np.zeros_like(m1)
for count, i in enumerate(order_list):
	m1p[:,:,i] = m1[:,:m1.shape[1],count]


summed_probs = m1p

length_list = [len(line.strip().split(',')[2]) for line in open('cb513test_solution.csv').readlines()]
print ('max protein seq length is', np.max(length_list))

ensemble_predictions = []
for protein_idx, i in enumerate(length_list):
    new_pred = ''
    for j in range(i):
        new_pred += labels[np.argmax(summed_probs[protein_idx, j ,:])]
    ensemble_predictions.append(new_pred)

        
# calculating accuracy 
def get_acc(gt,pred):
    #assert len(gt)== len(pred)
    correct = 0
    for i in range(len(gt)):
        if gt[i]==pred[i]:
            correct+=1
            
    return (1.0*correct)/len(gt)


gt_all = [line.strip().split(',')[3] for line in open('cb513test_solution.csv').readlines()]
acc_list = []
equal_counter = 0
total = 0

for gt,pred in zip(gt_all,ensemble_predictions):
	if len(gt) == len(pred):
		acc = get_acc(gt,pred)
		acc_list.append(acc)
		equal_counter +=1
	else :  
		acc = get_acc(gt,pred)
		acc_list.append(acc)
	#print(len(gt), gt)
	#print(len(pred), pred)
	#print(' ')
	total += 1

print ('mean accuracy is', np.mean(acc_list))
print (str(equal_counter)+' from '+str(total))
