# Adapted from "Neural ODE Processes" by Alexander Norcliffe, Cristian Bodnar,
# Ben Day, Jacob Moss, Pietro Liò

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class HafnerData(Dataset):
    '''
    Dataset of GSE100099
    MCF7 cells gamma-irradiated over 24 hours
    p53 is typically the protein of interest
    t=0,1,2,3,4,5,6,7,8,9,10,11,12,24
    '''

    def __init__(self, data_dir):
        target_genes = [
            'KAZN', 'PMAIP1', 'PRKAB1', 'CSNK1G1', 'E2F7', 'SLC30A1',
            'PTP4A1', 'RAP2B', 'SUSD6', 'UBR5-AS1', 'RNF19B', 'AEN', 'ZNF79', 'XPC',
            'FAM212B', 'SESN2', 'DCP1B', 'MDM2', 'GADD45A', 'SESN1', 'CDKN1A', 'BTG2'
        ]
        target_genes.extend([
            'DSCAM', 'C14orf93', 'RPL23AP64', 'RPS6KA5', 'MXD1', 'LINC01560', 'THNSL2',
            'EPAS1', 'ARSD', 'NACC2', 'NEDD9', 'GATS', 'ABHD4', 'BBS1', 'TXNIP',
            'KDM4A', 'ZNF767P', 'LTB4R', 'PI4K2A', 'ZNF337', 'PRKX', 'MLLT11',
            'HSPA4L', 'CROT', 'BAX', 'ORAI3', 'CES2', 'PVT1', 'ZFYVE1', 'PIK3R3',
            'TSPYL2', 'PROM2', 'ZBED5-AS1', 'CCNG1', 'STOM', 'IER5', 'STEAP3',
            'TYMSOS', 'TMEM198B', 'TIGAR', 'ASTN2', 'ANKRA2', 'RRM2B', 'TAP1', 'TP53I3', 'PNRC1',
            'GLS2', 'TMEM229B', 'IKBIP', 'ERCC5', 'KIAA1217', 'DDIT4', 'DDB2', 'TP53INP1'
        ])
        np.random.shuffle(target_genes)
        tfs = ['TP53']

        with open(data_dir+'/t0to24.tsv', 'r', 1) as f:
            contents = f.buffer
            df = pd.read_table(contents, sep='\t', index_col=0)

        columns = ['MCF7, t='+str(t)+' h, IR 10Gy, rep1' for t in range(13)]

        self.genes_df = df[df.index.isin(target_genes)][columns]
        self.genes_df = self.genes_df.reindex(target_genes)
        self.tfs_df = df[df.index.isin(tfs)][columns]

        m = self.genes_df.values
        genes_norm = 1/m.shape[0] * \
            np.linalg.norm(m, axis=1, ord=None)  # l2 norm
        self.genes = torch.FloatTensor(
            m / np.sqrt(genes_norm.reshape(-1, 1)), dtype=torch.float32).unsqueeze(-1)

        f = self.tfs_df.values
        tfs_norm = 1/f.shape[0] * \
            np.linalg.norm(f, axis=1, ord=None)  # l2 norm
        self.tfs = f / np.sqrt(tfs_norm.reshape(-1, 1))

        self.t = torch.FloatTensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.float32).view(-1, 1)
        self.t = self.t.repeat([self.genes.shape[0], 1, 1])
        self.data = list(zip(self.t, self.genes))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.genes.shape[0]
