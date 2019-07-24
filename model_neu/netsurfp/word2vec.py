import pandas as pd
import numpy as np
import time

from utils import get_data

start_time = time.time()

data_root = '/nosave/lange/cu-ssp/data/netsurfp/'
filename= "train_700"
seq_list = list('ACDEFGHIKLMNPQRSTVWY')

def seq2ngrams(seqs, n = 3):
    return np.array( [[seq[i:i+n] for i in range(int(len(seq)-2))] for seq in seqs])


def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        m = np.max(o)
        if m != 0:
            i = np.argmax(o)
            s += index[i]
        else:
            break
    return s


# load protvec data
protVec = pd.read_csv(data_root+'protVec_100d_3grams.csv', sep="\t")
print(protVec.head())

words = list(protVec['words'])

# change protVec tabel to word(3-gram):embedding dictionary
embeddings = protVec.iloc[:,1:].values
word_embed = dict(zip(protVec['words'], embeddings))

# load input data
train_input = np.load(data_root+filename+'_input.npy')
print("train input shape: "+str(train_input.shape))

# decode input data from one hot to sequences

prim_seq = []
for i, oh in enumerate(train_input):
    seq = onehot_to_seq(oh, seq_list)
    prim_seq.append(seq)


# create overlapping? 3 grams from sequences
n_grams = seq2ngrams(prim_seq)
print('n-gram shape:'+str(n_grams.shape))
print('first n gram:')
print(n_grams[0])

gram_time = time.time()-start_time
m, s = divmod(gram_time, 60)
print("Needed {:.0f}min {:.0f}s to create the 3 grams.".format(m, s))

# replace 3 gramm with 100 dim embedding
embed_seq = np.zeros((train_input.shape[0], train_input.shape[1]*100))

for i, grams in enumerate(n_grams):
    for j, g in enumerate(grams):
        embed_seq[i,j:j+100]=word_embed[g]

embed_time = time.time() -gram_time
m, s = divmod(embed_time, 60)
print("Needed {:.0f}min {:.0f}s to embed the sequences.".format(m, s))

print(embed_seq[0])
print(embed_seq[-1])

# save embedding to disk

np.save(data_root+'input_word2vec.npy', embed_seq)
print("Saved to disk.")

# reduce dimension with umap

# save to disk

end_time = time.time() -start_time
m, s = divmod(end_time, 60)
print("Needed {:.0f}min {:.0f}s in total.".format(m, s))
