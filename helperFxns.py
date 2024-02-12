import numpy as np
from scipy.sparse import csr_matrix as sparsify


def lett2num(msa_lett, code='ACDEFGHIKLMNPQRSTVWY-',lett2index=None):
    ''' Translate an alignment from a representation where the 20 natural amino
    acids are represented by letters to a representation where they are
    represented by the numbers 1,...,20, with any symbol not corresponding to an
    amino acid represented by 0.

    :Example:
       >>> msa_num = lett2num(msa_lett, code='ACDEFGHIKLMNPQRSTVWY')

    '''
    if not lett2index:
        lett2index = {aa: i+1 for i, aa in enumerate(code)}
    [Nseq, Npos] = [len(msa_lett), len(msa_lett[0])]
    msa_num = np.zeros((Nseq, Npos)).astype(int)
    for s, seq in enumerate(msa_lett):
        for i, lett in enumerate(seq):
            if lett in code:
                msa_num[s, i] = lett2index[lett]
    return msa_num

    

def alg2bin(alg, N_aa=21):
    ''' Translate an alignment of size M x L where the amino acids are represented
    by numbers between 0 and N_aa (obtained using lett2num) to a sparse binary
    array of size M x (N_aa x L).

    :Example:
      >>> Abin = alg2bin(alg, N_aa=20) '''

    [N_seq, N_pos] = alg.shape
    Abin_tens = np.zeros((N_aa, N_pos, N_seq))
    for ia in range(N_aa):
        Abin_tens[ia, :, :] = (alg == ia+1).T
    Abin = sparsify(Abin_tens.reshape(N_aa*N_pos, N_seq, order='F').T)
    return Abin


def filterAln(hd, seq):
    '''
    Given a numeric (but not yet binarized) alignment, filter out highly
    gapped positions and sequences according to a cutoff. Returns the filtered
    set of headers and sequences
    '''
    seqPosFilter = seq[:, np.sum(seq == 21, 0)/len(seq) < 0.5]
    hdFilter, ixKeep = [], []
    for i in range(len(seqPosFilter)):
        # print(np.sum(seqPosFilter[i] == 21)/len(seqPosFilter[i]))
        if (np.sum(seqPosFilter[i] == 21)/len(seqPosFilter[i]) < 0.5):
            # print(np.sum(seqPosFilter[i] == 21))
            # print(len(seqPosFilter[i]))

            hdFilter.append(hd[i])
            ixKeep.append(i)
    seqFilter = seqPosFilter[ixKeep, :]

    return hdFilter, seqFilter


def simMat(binalg, Npos):
    ''' Given a binarized alignment (from alg2bin) and the number of alignment positions,
    compute a sequence identity matrix'''
    smat = (binalg.dot(binalg.T)).todense()/Npos
    return smat.A


def num2lett(msa_num, code='ACDEFGHIKLMNPQRSTVWY-'):
    ''' Translate an alignment from a representation where the 20 natural amino
    acids are represented by numbers to a representation where they are
    represented by the letters A,...,Y, with any symbol not corresponding to an
    amino acid represented by '-'.

    :Example:
       >>> msa_lett = num2lett(msa_num, code='ACDEFGHIKLMNPQRSTVWY')

    '''
    index2lett = {i+1: aa for i, aa in enumerate(code)}
    [Nseq, Npos] = [len(msa_num), len(msa_num[0])]
    msa_lett = np.zeros((Nseq, Npos)).astype(str)
    for s, seq in enumerate(msa_num):
        for i, num in enumerate(seq):
            if num > 0:
                msa_lett[s, i] = index2lett[num]
            else:
                msa_lett[s, i] = '-'

    # convert to list of strings
    msa_lett = [''.join(msa_lett[i, :]) for i in range(Nseq)]
    return msa_lett
