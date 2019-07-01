from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
import torch
import sys
import os
import argparse
import time
import numpy as np

start_time = time.time()

model_dir = Path('../../seqVec')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'
seqvec  = ElmoEmbedder(options,weights,cuda_device=0) # cuda_device=-1 for CPU
#inputs: primary structure
train_input = np.load('../data/train_input.npy')
test_input = np.load('../data/test_input.npy')

#Get embedding for amino acid sequence:
train_input_embedding = []
times = []
for i,seq in enumerate(train_input):
    print('\n \n----------------------')
    print('----------------------')
    print(i, '/', len(train_input))
    print('----------------------')
    embedding = seqvec.embed_sentence( list(seq) ) # List-of-Lists with shape [3,L,1024]

    #Get 1024-dimensional embedding for per-residue predictions:
    #residue_embd = torch.tensor(embedding).sum(dim=0) # Tensor with shape [L,1024]
    #Get 1024-dimensional embedding for per-protein predictions:
    protein_embd = torch.tensor(embedding).sum(dim=0).mean(dim=0) # Vector with shape [1024]
    protein_embd_np = protein_embd.cpu().detach().numpy()
    train_input_embedding.append(protein_embd_np)
    time = time.time()-start_time
    times.append(time)
    print("For {} residues {:.0f}s needed.".format(len(train_input), time))

end_time = time.time() - start_time
m, s = divmod(end_time, 60)
print("The embedding calculation needed {:.0f}min {:.0f}s in total.".format(m, s))

np.save('times.npy', times)
np.save('train_input_embedding.npy', train_input_embedding)