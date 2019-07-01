from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
import torch

model_dir = Path('../../seqVec')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'
seqvec  = ElmoEmbedder(options,weights,cuda_device=0) # cuda_device=-1 for CPU
#inputs: primary structure
train_input = np.load('../data/train_input.npy')
test_input = np.load('../data/test_input.npy')

#Get embedding for amino acid sequence:
seq = train_input[0]
print(seq.shape)
embedding = seqvec.embed_sentence( list(seq) ) # List-of-Lists with shape [3,L,1024]

#Get 1024-dimensional embedding for per-residue predictions:
#residue_embd = torch.tensor(embedding).sum(dim=0) # Tensor with shape [L,1024]
#Get 1024-dimensional embedding for per-protein predictions:
protein_embd = torch.tensor(embedding).sum(dim=0).mean(dim=0) # Vector with shape [1024]
test = protein_embd.cpu().detach().numpy()
print(test)
print(embedding.shape)
print(protein_embd.shape)
print(test.shape)