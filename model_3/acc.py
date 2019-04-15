import numpy as np
import matplotlib.pyplot as plt

m1p = np.load('cb513_test_prob_1.npy')

print (m1p.shape)

# warped_order_1 = ['NoSeq', 'H', 'E', 'L','T', 'S', 'G', 'B',  'I']
#                    0        1    2    3   4    5    6    7     8
#                  ['L',     'B', 'E', 'G','I', 'H', 'S', 'T', 'NoSeq'] # new order
order_list = [8,5,2,0,7,6,3,1,4]
labels = ['L', 'B', 'E', 'G','I', 'H', 'S', 'T', 'NoSeq']

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
    assert len(gt)== len(pred)
    correct = 0
    for i in range(len(gt)):
        if gt[i]==pred[i]:
            correct+=1
            
    return (1.0*correct)/len(gt)

gt_all = [line.strip().split(',')[3] for line in open('cb513test_solution.csv').readlines()]
acc_list = []

for gt,pred in zip(gt_all,m1p):
	if len(gt) == len(pred):
		acc = get_acc(gt,pred)
		acc_list.append(acc)
	else : print('not equal')
print(acc_list)
print ('mean accuracy is', np.mean(acc_list))
