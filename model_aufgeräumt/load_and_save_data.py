import numpy as np
import pandas as pd

maxlen_seq = 700

cullpdb =np.load("../data/cullpdb_train.npy").item()
data13=np.load("../data/casp13.npy").item()
cullpdb_df = pd.DataFrame(cullpdb)
data13_df = pd.DataFrame(data13)

#load train, test, pssm-profiles and hmm-profiles which are not longer than maxlen_seq
print('load data...')
train_dssp = cullpdb_df['dssp'][cullpdb_df['seq'].apply(len)<=maxlen_seq].values
train_pssm_list = cullpdb_df['pssm'][(cullpdb_df['seq'].apply(len)<=maxlen_seq)].values
train_hmm_list = cullpdb_df['hhm'][(cullpdb_df['seq'].apply(len)<=maxlen_seq)].values


test_dssp = data13_df['dssp'][data13_df['seq'].apply(len)<=maxlen_seq].values
test_pssm_list, test_hmm_list = data13_df[['pssm', 'hhm']][(data13_df['seq'].apply(len)<=maxlen_seq)].values

def make_q8(dssp):
    q8_beta = []
    q8 = []
    for item in dssp:
        q8_beta.append(item.replace('-', 'L'))
    for item in q8_beta:
        q8.append(item.replace('_', 'L'))
    return q8
print("change q8 sequences... ")
train_dssp_q8 = make_q8(train_dssp)
test_dssp_q8 = make_q8(test_dssp)

def reshape_and_pad(list):
    number_seq = len(list)
    len_profiles = list[0].shape[1]
    data = np.zeros([number_seq, maxlen_seq, len_profiles])
    for i in range(number_seq):
        for j in range(len(list[i])):
            for k in range(len_profiles):
                data[i][j][k]=list[i][j][k]
    return data

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    data = (data-mean)/std
    return data

def standardize(data):
    min = np.min(data)
    max = np.max(data)
    data = (data-min)(max-min)
    return data
print("reshape and pad profiles...")
train_pssm = reshape_and_pad(train_pssm_list)
train_hmm = reshape_and_pad(train_hmm_list)
test_pssm = reshape_and_pad(test_pssm_list)
test_hmm = reshape_and_pad(test_hmm_list)

train_profiles = np.concatenate((train_pssm, train_hmm),axis=2)
test_profiles = np.concatenate((test_pssm, test_hmm),axis=2)
print("normalize profiles...")
train_pssm_norm = normalize(train_pssm)
train_hmm_norm = normalize(train_hmm)
test_pssm_norm = normalize(test_pssm)
test_hmm_norm = normalize(test_hmm)

train_profiles_norm = np.concatenate((train_pssm_norm, train_hmm_norm),axis=2)
test_profiles_norm = np.concatenate((test_pssm_norm, test_hmm_norm),axis=2)
print("standardize profiles ...")
train_pssm_stan = standardize(train_pssm)
train_hmm_stan = standardize(train_hmm)
test_pssm_stan = standardize(test_pssm)
test_hmm_stan = standardize(test_hmm)

train_profiles_stan = np.concatenate((train_pssm_stan, train_hmm_stan),axis=2)
test_profiles_stan = np.concatenate((test_pssm_stan, test_hmm_stan),axis=2)
print("save data...")
np.save('../data/train_q8.npy', train_dssp_q8)
np.save('../data/test_q8.npy', test_dssp_q8)
np.save('../data/train_profiles.npy', train_profiles)
np.save('../data/test_profiles.npy', test_profiles)
np.save('../data/train_profiles_norm.npy', train_profiles_norm)
np.save('../data/test_profiles_norm.npy', test_profiles_norm)
np.save('../data/train_profiles_stan.npy', train_profiles_stan)
np.save('../data/test_profiles_stan.npy', test_profiles_stan)