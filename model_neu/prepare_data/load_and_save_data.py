import numpy as np
import pandas as pd

maxlen_seq = 700
minlen_seq= 100

cullpdb =np.load("/nosave/lange/cu-ssp/data/data_qzlshy/cullpdb_train.npy").item()
data13=np.load("/nosave/lange/cu-ssp/data/data_qzlshy/cb513_hmm.npy").item()
cullpdb_df = pd.DataFrame(cullpdb)
data13_df = pd.DataFrame(data13)

#select sequence lengths
seq_train=cullpdb_df['seq']
seq_test=data13_df['seq']
seq_range_train = [minlen_seq<=len(seq_train[i])<=maxlen_seq for i in range(len(seq_train))]
seq_range_test = [minlen_seq<=len(seq_test[i])<=maxlen_seq for i in range(len(seq_test))]

#load train, test, pssm-profiles and hmm-profiles which are not longer than maxlen_seq
print('load data...')
#primary structure
train_input = cullpdb_df[['seq']][seq_range_train].values.squeeze()
test_input= data13_df[['seq']][seq_range_test].values.squeeze()
#secondary structure
train_dssp = cullpdb_df['dssp'][seq_range_train].values.squeeze()
test_dssp = data13_df['dssp'][seq_range_test].values.squeeze()
#pssm profiles
train_pssm_list = cullpdb_df['pssm'][seq_range_train].values.squeeze()
test_pssm_list = data13_df['pssm'][seq_range_test].values.squeeze()
#hmm profiles
train_hmm_list = cullpdb_df['hhm'][seq_range_train].values.squeeze()
test_hmm_list = data13_df['hhm'][seq_range_test].values.squeeze()

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

print("reshape and pad profiles...")
train_pssm = reshape_and_pad(train_pssm_list)
test_pssm = reshape_and_pad(test_pssm_list)
train_hmm = reshape_and_pad(train_hmm_list)
test_hmm = reshape_and_pad(test_hmm_list)

print("save data...")
np.save('/nosave/lange/cu-ssp/data/data_qzlshy/train_input.npy', train_input)
np.save('/nosave/lange/cu-ssp/data/data_qzlshy/test_input.npy', test_input)

np.save('/nosave/lange/cu-ssp/data/data_qzlshy/train_q8.npy', train_dssp_q8)
np.save('/nosave/lange/cu-ssp/data/data_qzlshy/test_q8.npy', test_dssp_q8)

np.save('/nosave/lange/cu-ssp/data/data_qzlshy/data/train_pssm.npy', train_pssm)
np.save('/nosave/lange/cu-ssp/data/data_qzlshy/data/test_pssm.npy', test_pssm)

np.save('/nosave/lange/cu-ssp/data/data_qzlshy/data/train_hmm.npy', train_hmm)
np.save('/nosave/lange/cu-ssp/data/data_qzlshy/data/test_hmm.npy', test_hmm)


