from allennlp.commands.elmo import ElmoEmbedder
from keras.preprocessing import text, sequence
from pathlib import Path
import torch
import sys
import os
import argparse
import time
import numpy as np
import telegram

model_dir = Path('../../seqVec')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'
seqvec  = ElmoEmbedder(options,weights,cuda_device=0) # cuda_device=-1 for CPU

#inputs: primary structure
train_input = np.load('../data/netsurfp/train_input.npy')
test_input = np.load('../data/netsurfp/cb513_input.npy')

def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        m = np.max(o)
        if m != 0:
            i = np.argmax(o)
            s += index[i]
        else: #for unknown proteins all-zero encoding, will be called X
            s += index[-1]
    return s

data_root = '../data/netsurfp/'
data_train = np.load(data_root+'Train_HHblits.npz')
data_cb513 = np.load(data_root+'CB513_HHblits.npz')

maxlen_seq = 600
minlen_seq= 100

def get_seq_mask(data):
    mask = data['data'][:,:,50]
    seq_range = [minlen_seq<=mask[i].sum()<=maxlen_seq for i in range(len(mask))]
    return mask[seq_range]

#create sequence representation
'''
train_q8 = data['data'][:,:,57:65]
seq_mask = data['data'][:,:,50]

test = onehot_to_seq(train_q8[0][seq_mask[0]==1], q8_list)
q8_seq = []
for i, is_seq in enumerate(seq_mask):
    seq = onehot_to_seq(train_q8[i][seq_mask[i]>0], q8_list)
    q8_seq.append(seq)

'''
def telegram_me(m, s, model_name, test_acc = None, hmm=False, standardize=False, normalize = False):
    Token = "806663548:AAEJIMIBEQ9eKdyF8_JYnxUhUsDQZls1w7w"
    chat_ID = "69661085"
    bot = telegram.Bot(token=Token)
    msg = '{} ist erfolgreich durchgelaufen! \U0001F60B \n\n' \
          '(Gesamtlaufzeit {:.0f}min {:.0f}s)'.format(model_name, m, s)
    if hmm:
        verb = ''
        if standardize:
            verb += 'standardisierte '
        if normalize:
            verb += 'und normalisierte '
        msg+='\nEs wurden '+verb+'HMM Profile verwendet.'
    if test_acc is not None:
        msg += '\nTest accuracy: {:.3%}'.format(test_acc)
    bot.send_message(chat_id=chat_ID, text=msg)

def calculate_and_save_embedding(input, mask_seq):
    # Get embedding for amino acid sequence:
    input_embedding = []
    times = []

    for i, seq in enumerate(input):
        t1 = time.time()
        print('\n \n----------------------')
        print('----------------------')
        print('Sequence ', (i + 1), '/', len(input))
        print('----------------------')

        input_seq = onehot_to_seq(seq[mask_seq[i]>0], list('ACEDGFIHKMLNQPSRTWVYX') )
        embedding = seqvec.embed_sentence(list(input_seq))  # List-of-Lists with shape [3,L,1024]

        # Get 1024-dimensional embedding for per-residue predictions:
        residue_embd = torch.tensor(embedding).sum(dim=0) # Tensor with shape [L,1024]
        # Get 1024-dimensional embedding for per-protein predictions:
        #protein_embd = torch.tensor(embedding).sum(dim=0).mean(dim=0)  # Vector with shape [1024]
        residue_embd_pad = torch.nn.ConstantPad2d((0, 0, 0, (maxlen_seq-len(input_seq) )), 0)(residue_embd)
        residue_embd_np = residue_embd_pad.cpu().detach().numpy()
        print(residue_embd_np.shape)
        input_embedding.append(residue_embd_np)
        t = time.time() - t1
        times.append(t)
        print("For {} residues {:.0f}s needed.".format(len(input_seq), t))

    end_time = time.time() - start_time
    m, s = divmod(end_time, 60)
    print("The embedding calculation needed {:.0f}min {:.0f}s in total.".format(m, s))
    return input_embedding, times

start_time = time.time()
mask = get_seq_mask(data_train)
train_input_embedding, train_times = calculate_and_save_embedding(train_input, mask_seq=mask[:,:600])
np.save(data_root+'train_netsurfp_times_residue.npy', train_times)
np.save(data_root+'train_netsurfp_input_embedding_residue.npy', train_input_embedding)

time_end = time.time() - start_time
m, s = divmod(time_end, 60)
telegram_me(m, s, sys.argv[0])

start_time = time.time()
mask = get_seq_mask(data_cb513)
test_input_embedding, test_times = calculate_and_save_embedding(test_input, mask_seq=mask[:,:600])
np.save(data_root+'cb513_netsurfp_times_residue.npy', test_times)
np.save(data_root+'cb513_netsurfp_input_embedding_residue.npy', test_input_embedding)

time_end = time.time() - start_time
m, s = divmod(time_end, 60)
telegram_me(m, s, sys.argv[0])
