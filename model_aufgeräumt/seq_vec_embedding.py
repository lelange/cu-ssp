from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
import torch
import sys
import os
import argparse
import time
import numpy as np



model_dir = Path('../../seqVec')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'
seqvec  = ElmoEmbedder(options,weights,cuda_device=0) # cuda_device=-1 for CPU
#inputs: primary structure
train_input = np.load('../data/train_input.npy')
test_input = np.load('../data/test_input.npy')

def calculate_and_save_embedding(input):
    # Get embedding for amino acid sequence:
    input_embedding = []
    times = []
    for i, seq in enumerate(input):
        print('\n \n----------------------')
        print('----------------------')
        print('Sequence ', (i + 1), '/', len(input))
        print('----------------------')
        embedding = seqvec.embed_sentence(list(seq))  # List-of-Lists with shape [3,L,1024]

        # Get 1024-dimensional embedding for per-residue predictions:
        # residue_embd = torch.tensor(embedding).sum(dim=0) # Tensor with shape [L,1024]
        # Get 1024-dimensional embedding for per-protein predictions:
        protein_embd = torch.tensor(embedding).sum(dim=0).mean(dim=0)  # Vector with shape [1024]
        protein_embd_np = protein_embd.cpu().detach().numpy()
        input_embedding.append(protein_embd_np)
        t = time.time() - start_time
        times.append(t)
        print("For {} residues {:.0f}s needed.".format(len(input), t))

    end_time = time.time() - start_time
    m, s = divmod(end_time, 60)
    print("The embedding calculation needed {:.0f}min {:.0f}s in total.".format(m, s))
    return input_embedding, times

start_time = time.time()
train_input_embedding, train_times = calculate_and_save_embedding(train_input)
np.save('../data/train_times.npy', train_times)
np.save('../data/train_input_embedding.npy', train_input_embedding)


start_time = time.time()
test_input_embedding, test_times = calculate_and_save_embedding(test_input)
np.save('../data/test_times.npy', test_times)
np.save('../data/test_input_embedding.npy', test_input_embedding)
