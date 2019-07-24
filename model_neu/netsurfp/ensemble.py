import numpy as np
import matplotlib.pyplot as plt
from utils import get_data, onehot_to_seq, onehot_to_seq2, get_acc, get_acc2
data_file = "preds/Q8/"

m1 = np.load(data_file+'cb513_700_pred_1.npy')
#m2 = np.load(data_file+'cb513_608_pred_2.npy')
m3 = np.load(data_file+'cb513_700_pred_3.npy')
m4 = np.load(data_file+'cb513_700_pred_4.npy')
m5 = np.load(data_file+'cb513_700_pred_5.npy')
m6 = np.load(data_file+'cb513_700_pred_6.npy')
mask = np.load(data_file+'cb513_700_len.npy')
length_list = np.sum(mask, axis=1)

print(m1.shape, m2.shape, m3.shape, m4.shape, m5.shape, m6.shape)

# warped_order_1 = ['NoSeq', 'H', 'E', 'L','T', 'S', 'G', 'B',  'I']
#                    0        1    2    3   4    5    6    7     8
#                  ['L',     'B', 'E', 'G','I', 'H', 'S', 'T', 'NoSeq'] # new order

# for decoding one-hot-encoding
order_list = [8, 5, 2, 0, 7, 6, 3, 1, 4]
labels = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T', 'NoSeq']

q8_list = list('-GHIBESTC')
q3_list = list('-HHHEECCC')

m1p = np.zeros_like(m4)
m2p = np.zeros_like(m4)
m3p = np.zeros_like(m4)
m4p = np.zeros_like(m4)
m5p = np.zeros_like(m4)
m6p = np.zeros_like(m4)

_, y_true = get_data('cb513_700', hmm=True, normalize=False, standardize=True)

# change one-hot encoding order st. it corresponds to list of labels and ensure same length
for count, i in enumerate(order_list):
    m1p[:, :, i] = m1[:, :700, count]
    m2p[:, :, i] = m2[:, :700, count]
    m3p[:, :, i] = m3[:, :700, count]
    m4p[:, :, i] = m4[:, :700, count]
    m5p[:, :, i] = m5[:, :700, count]
    m6p[:, :, i] = m6[:, :700, count]


# check that prediction is prob distribution
def check_softmax(T):
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            try:
                assert np.abs(np.sum(T[i, j, :]) - 1.0) < 0.0001
            except:
                print(np.sum(T[i, j, :]), 'failed')
                return
    print('outputs are softmaxed')


# check_softmax(m1)

summed_probs = m1 + m3 + m4 + m5 + m6

#length_list = [len(line.strip().split(',')[2]) for line in open('cb513test_solution.csv').readlines()]
print('max protein seq length is', np.max(length_list))


def get_ensemble_pred(labels, summed_probs):

    # create new prediciton as highest scorer in sum of all other predictions
    ensemble_predictions = []
    for protein_idx, i in enumerate(length_list):
        new_pred = ''
        for j in range(i):
            new_pred += labels[np.argmax(summed_probs[protein_idx, j, :])]
        ensemble_predictions.append(new_pred)

    return ensemble_predictions

q3_pred = 0
q8_pred = 0
q3_len = 0
q8_len = 0

q8_accs = []
q3_accs = []

y_ensemble = get_ensemble_pred()

for true, pred in zip(y_true, y_ensemble):
    seq3 = onehot_to_seq(pred, q3_list)
    seq8 = onehot_to_seq(pred, q8_list)
    seq_true_3 = onehot_to_seq2(true, q3_list)
    seq_true_8 = onehot_to_seq2(true, q8_list)

    if i:
        print('Q3 prediction, first pred then true: ')
        print(seq3[:60])
        print(seq_true_3[:60])

        print('Q8 prediction, first pred then true: ')
        print(seq8[:60])
        print(seq_true_8[:60])

        i = False

    corr3, len3 = get_acc(seq_true_3, seq3)
    corr8, len8 = get_acc(seq_true_8, seq8)
    q8_accs.append(get_acc2(seq_true_8, seq8))
    q3_accs.append(get_acc2(seq_true_3, seq3))
    q3_pred += corr3
    q8_pred += corr8
    q3_len += len3
    q8_len += len8

print("Accuracy #sum(correct per proteins)/#sum(len_proteins):")
print("Q3 " + test + " test accuracy: " + str(q3_pred / q3_len))
print("Q8 " + test + " test accuracy: " + str(q8_pred / q8_len))
print("\nAccuracy mean(#correct per protein/#len_protein):")
print("Q3 " + test + " test accuracy: " + str(np.mean(q3_accs)))
print("Q8 " + test + " test accuracy: " + str(np.mean(q8_accs)))

