import numpy as np
import pandas as pd

data_root = '/nosave/lange/cu-ssp/data/'

def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        if np.max(o) != 0:
            i = np.argmax(o)
            s += index[i]
        else:
            break
    return s

def standard(data):
    mean = np.mean(data)
    std = np.std(data)
    data_ = (data - mean) / std
    return data_


def normal(data):
    min = np.min(data)
    max = np.max(data)
    data_ = (data-min)/(max-min)
    return data_

def get_netsurf_data(filename, max_len=None):
    # filename =
    seq_list = list('ACDEFGHIKLMNPQRSTVWY')
    path = data_root+'netsurfp/'

    input_onehot = np.load(path+filename + '_input.npy')
    q8_onehot = np.load(path+filename + '_q9.npy')
    profiles = np.load(path+filename + '_hmm.npy')

    prim_seq = []
    for i, oh in enumerate(input_onehot):
        seq = onehot_to_seq(oh, seq_list)
        prim_seq.append(seq)

    out = pd.DataFrame({'input_AA': prim_seq, 'input_onehot': input_onehot,
                        'output_onehot': q8_onehot,
                        })

    return out, profiles

sifat = pd.read_pickle(data_root+'chemical_features.pkl')
sifat['Molekülmasse']=normal(sifat['Molekülmasse'])
sifat['Volumen']=normal(sifat['Volumen'])

def get_input_features(filename):

    data,_ = get_netsurf_data(filename)

    input_data = data['input_AA']

    input_features = np.zeros((len(input_data), 700, len(sifat.columns)))

    for i, seq in enumerate(input_data):
        for j, aa in enumerate(seq):
            input_features[i, j, :] = sifat.loc[aa].values

    np.save(data_root + 'netsurfp/' + filename+'_input_features.npy', input_features)
    print('File saved at '+data_root + 'netsurfp/' + filename+'_input_features.npy.')


get_input_features('train_700')
get_input_features('cb513_700')
get_input_features('ts115_700')
get_input_features('casp12_700')










