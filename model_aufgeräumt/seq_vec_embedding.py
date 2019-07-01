from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
import torch

model_dir = Path('../../seqVec')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'
seqvec  = ElmoEmbedder(options,weights,cuda_device=0) # cuda_device=-1 for CPU
#Get embedding for amino acid sequence:
seq = 'SEQWENCE' # your amino acid sequence
embedding = seqvec.embed_sentence( list(seq) ) # List-of-Lists with shape [3,L,1024]

#Get 1024-dimensional embedding for per-residue predictions:
residue_embd = torch.tensor(embedding).sum(dim=0) # Tensor with shape [L,1024]
#Get 1024-dimensional embedding for per-protein predictions:
protein_embd = torch.tensor(embedding).sum(dim=0).mean(dim=0) # Vector with shape [1024]

print(embedding.shape)
print(residue_embd.shape)
print(protein_embd.shape)