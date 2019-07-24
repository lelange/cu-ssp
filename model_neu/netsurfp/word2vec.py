import pandas as pd
import numpy as np
import time
import umap

data_root = '/nosave/lange/cu-ssp/data/netsurfp/'
filename= "cb513_700"
seq_list = list('ACDEFGHIKLMNPQRSTVWY')
MAXLEN_SEQ = 700

if MAXLEN_SEQ is None:
    ending = "full"
else:
    ending= str(MAXLEN_SEQ)

file_train = 'train_' + ending
file_test = ['cb513_'+ ending, 'ts115_'+ ending, 'casp12_'+ ending]


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

start_time = time.time()

# load protvec data
protVec = pd.read_csv(data_root+'protVec_100d_3grams.csv', sep="\t")
print(protVec.head())
words = list(protVec['words'])

# change protVec tabel to word(3-gram):embedding dictionary
embeddings = protVec.iloc[:,1:].values
word_embed = dict(zip(protVec['words'], embeddings))


for filename in file_test:

    # load input data
    train_input = np.load(data_root + filename + '_input.npy')
    print("train input shape: " + str(train_input.shape))

    # decode input data from one hot to sequences

    prim_seq = []
    for i, oh in enumerate(train_input):
        seq = onehot_to_seq(oh, seq_list)
        prim_seq.append(seq)

    # create overlapping 3 grams from sequences
    n_grams = seq2ngrams(prim_seq)
    print('n-gram shape:' + str(n_grams.shape))
    print('first n gram:')
    print(n_grams[0])

    gram_time = time.time() - start_time
    m, s = divmod(gram_time, 60)
    print("Needed {:.0f}min {:.0f}s to create the 3 grams.".format(m, s))

    # replace 3 gramm with 100 dim embedding
    embed_seq = np.zeros((train_input.shape[0], train_input.shape[1] * 100))

    for i, grams in enumerate(n_grams):
        for j, g in enumerate(grams):
            embed_seq[i, j:j + 100] = word_embed[g]

    embed_time = time.time() - gram_time
    m, s = divmod(embed_time, 60)
    print("Needed {:.0f}min {:.0f}s to embed the sequences.".format(m, s))


    # save embedding to disk

    np.save(data_root + filename + "_word2vec_input.npy", embed_seq)
    print("Saved to " + data_root + filename + "_word2vec_input.npy")

    # ----------- dim. reduction with umap ----

    #embed_seq =np.load(data_root + filename + "_word2vec_input.npy")
    print(embed_seq.shape)

    nb_components = 500

    print("Start UMAP reduction...")

    # reduce dimension with umap
    reducer = umap.UMAP(n_components=nb_components, verbose=True)
    embedding = reducer.fit_transform(embed_seq)

    print(embedding.shape)

    # save to disk
    np.save(data_root + filename + '_umap_' + str(nb_components) + '_input.npy', embedding)
    print("Saved to " + data_root + filename + '_umap_' + str(nb_components) + '_input.npy')

end_time = time.time() - start_time
m, s = divmod(end_time, 60)
print("Needed {:.0f}min {:.0f}s in total.".format(m, s))
