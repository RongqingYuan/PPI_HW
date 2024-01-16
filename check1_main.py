import os

import scipy
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO

from helperFxns import lett2num, alg2bin, filterAln, simMat


files = [file for file in os.listdir() if file.endswith('.fasta')]
print(files)
for file in files:
    # read the file with the Bio.SeqIO module
    encoding_file = file.split('.')[0] + '_one-hot.npy'
    header_file = file.split('.')[0] + '_header.txt'
    filter_header_file = file.split('.')[0] + '_filter_header.txt'
    headers = []
    seqs = []
    for seq_record in SeqIO.parse(file, "fasta"):
        header = '>' + str(seq_record.description)
        headers.append(header)
        seq = str(seq_record.seq)
        seqs.append(seq)

        # print(len(seq))
        # print(seq_record.id)
        # print(repr(seq_record.seq))
        # print(len(seq_record))
        # print("\n")

    with open(header_file, 'w') as f:
        for header in headers:
            f.write(header)
            f.write('\n')

    num = lett2num(seqs, code='ACDEFGHIKLMNPQRSTVWY-')
    # print(num.shape)
    # print(type(num))

    bin = alg2bin(num, N_aa=21)
    # print(bin.shape)
    # print(type(bin))

    # filter out highly gapped positions and sequences according to the cutoff
    hdFilter, seqFilter = filterAln(headers, num)
    # print(seqFilter.shape)
    # print(type(seqFilter))

    print(seqFilter.shape)
    print(len(hdFilter))

    with open(filter_header_file, 'w') as f:
        for header in hdFilter:
            f.write(header)
            f.write('\n')

    filter_bin = alg2bin(seqFilter, N_aa=21)
    # print(filter_bin.shape)
    # print(type(filter_bin))

    # compute a sequence identity matrix
    sim = simMat(filter_bin, seqFilter.shape[1])
    # print(sim.shape)
    # print(type(sim))

    # sim is a symmetric matrix, get the values above the diagonal
    sim_diag = sim[np.triu_indices(sim.shape[0], k=1)]

    # convert to a shape (n,) array
    sim_diag = np.array(sim_diag).reshape(-1)
    print(sim_diag.shape)

    # plot the histogram of sequence identities
    plt.figure()
    plt.hist(sim_diag, bins=100)
    plt.xlabel('Sequence identity')
    plt.ylabel('Counts')
    plt.title('Histogram of sequence identities of proteins in MSA_{}'.format(
        file.split('.')[0]))
    # plt.show()
    plt.savefig(file.split('.')[0] + '_hist.png')
