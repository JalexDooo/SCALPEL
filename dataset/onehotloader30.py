import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import h5py
import pandas as pd
import numpy as np

from transformers import BertModel, BertTokenizer
from utils import FixRNABert

ohc_seq = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'N': [0, 0, 0, 0]}

ohc_fold = {
    '(': [1, 0, 0],
    ')': [0, 1, 0],
    '.': [0, 0, 1],
    'N': [0, 0, 0]}

def seq_one_hot_code(seq):
    seq = seq.upper()
    lst_seq = list(seq)
    return [ohc_seq[i] for i in lst_seq]

def fold_one_hot_code(seq):
    lst_seq = list(seq)
    return [ohc_fold[i] for i in lst_seq]

class OnehotLoader30(DataLoader):
    def __init__(self, data, max_len=30):
        self.max_len = max_len

        # self.data = pd.read_csv(data_path, sep=',', encoding='GBK', header=None)
        # self.data[11] = self.data[11].replace('na', float(0.)) # h_mfe
        # self.data[24] = self.data[24].replace('na', float(0.)) # RPKM
        # self.data = self.data.iloc[1:]
        self.data = data

        model = 0
        gRNA_id = 1
        cell_line = 2
        target_gene = 3
        gRNA_seq = 4
        tRNA_seq = 5
        trans_id = 6
        ext_tseq = 7
        DR_seq = 8
        gRNA_mfe = 9
        DR_gRNA_fold = 10
        h_mfe = 11
        icshape = 12
        prt_binding_p = 13
        lfc = 14
        p1 = 15
        p2 = 16
        ext_p1 = 17
        ext_p2 = 18
        ms_match = 19
        ms_match_idt = 20
        norm_lfc = 21
        parental_gRNA = 22
        parental_gRNA_lfc = 23
        RPKM = 24
        st_relate_len = 25
        ed_relate_len = 26
        st_tar_trans = 27
        ed_tar_trans = 28
        cds_st = 29
        cds_ed = 30
        utr5 = 31
        cds = 32
        utr3 = 33
        utr5_rate = 34
        cds_rate = 35
        utr3_rate = 36
        read_depth = 37

        

        # useful feature
        self.g_seq = self.data[gRNA_seq].to_numpy()
        self.t_seq = self.data[tRNA_seq].to_numpy()
        self.extt_seq = self.data[ext_tseq].to_numpy()
        self.dr_seq = self.data[DR_seq].to_numpy()
        self.g_fold = self.data[DR_gRNA_fold].to_numpy()
        self.g_mfe = self.data[gRNA_mfe].to_numpy().astype(np.float32).reshape(-1,1)
        self.h_mfe = self.data[h_mfe].to_numpy().astype(np.float32).reshape(-1,1)
        self.icshape = self.data[icshape].to_numpy()
        self.binding_p = self.data[prt_binding_p].to_numpy()
        self.norm_label = self.data[norm_lfc].to_numpy().astype(np.float32).reshape(-1,1)
        self.p1 = self.data[p1].to_numpy().astype(np.float32).reshape(-1,1)
        self.p2 = self.data[p2].to_numpy().astype(np.float32).reshape(-1,1)
        self.rpkm = self.data[RPKM].to_numpy().astype(np.float32).reshape(-1,1)
        self.st_relate_len = self.data[st_relate_len].to_numpy().astype(np.float32).reshape(-1,1)
        self.ed_relate_len = self.data[ed_relate_len].to_numpy().astype(np.float32).reshape(-1,1)
        self.utr5_rate = self.data[utr5_rate].to_numpy().astype(np.float32).reshape(-1,1)
        self.cds_rate = self.data[cds_rate].to_numpy().astype(np.float32).reshape(-1,1)
        self.utr3_rate = self.data[utr3_rate].to_numpy().astype(np.float32).reshape(-1,1)
        self.read_depth = self.data[read_depth].to_numpy()

        # debug feature
        self.cell_line = self.data[cell_line].to_numpy()
        self.target_gene = self.data[target_gene].to_numpy()

        # csv output
        self.csv_model = self.data[model].to_numpy()
        self.csv_gRNA_id = self.data[gRNA_id].to_numpy()
        self.csv_ms_match_idt = self.data[ms_match_idt].to_numpy()
        self.csv_gRNA_seq = self.data[gRNA_seq].to_numpy()
        self.csv_tRNA_seq = self.data[tRNA_seq].to_numpy()
        self.csv_g_fold = self.data[DR_gRNA_fold].to_numpy()
        self.csv_icshape = self.data[icshape].to_numpy()
        self.csv_binding_p = self.data[prt_binding_p].to_numpy()
        self.csv_rpkm = self.data[RPKM].to_numpy().astype(np.float32)
        self.csv_g_mfe = self.data[gRNA_mfe].to_numpy().astype(np.float32)
        self.csv_h_mfe = self.data[h_mfe].to_numpy().astype(np.float32)
        self.csv_st_relate_len = self.data[st_relate_len].to_numpy().astype(np.float32)
        self.csv_ed_relate_len = self.data[ed_relate_len].to_numpy().astype(np.float32)
        self.csv_utr5_rate = self.data[utr5_rate].to_numpy().astype(np.float32)
        self.csv_cds_rate = self.data[cds_rate].to_numpy().astype(np.float32)
        self.csv_utr3_rate = self.data[utr3_rate].to_numpy().astype(np.float32)
        self.csv_read_depth = self.data[read_depth].to_numpy()
        self.csv_norm_label = self.data[norm_lfc].to_numpy().astype(np.float32)

        # print('cell_line: ', np.unique(self.cell_line))
        # print('target_gene: ', np.unique(self.target_gene))
        for idx in range(len(self.icshape)):
            if self.icshape[idx] != 'na' and self.icshape[idx] != 'n,a':
                ics = self.icshape[idx].split(',')
                ics = ics[int(self.p1[idx][0])-1:int(self.p2[idx][0])]
                icss = [float(i) for i in ics]
                if len(icss) < self.max_len:
                    for i in range(self.max_len-len(icss)):
                        icss.append(float(-1.))
                self.icshape[idx] = np.array(icss)
            else:
                icss = []
                for i in range(self.max_len):
                    icss.append(float(-2.))
                self.icshape[idx] = np.array(icss)


        


    def __len__(self):
        return len(self.g_seq)
    
    def __getitem__(self, idx):
        max_len = 30

        binding = self.binding_p[idx].split(',')
        bindings = [float(i) for i in binding]
        binding_p = np.array(bindings)

        g_seq = self.g_seq[idx]
        t_seq = self.t_seq[idx]
        g_fold = self.g_fold[idx]

        if len(self.g_seq[idx]) < max_len:
            g_seq = self.g_seq[idx] + 'N'*(max_len-len(self.g_seq[idx]))
        
        if len(self.t_seq[idx]) < max_len:
            t_seq = self.t_seq[idx] + 'N'*(max_len-len(self.t_seq[idx]))
        
        if len(self.g_fold[idx]) < len(self.dr_seq[idx]) + max_len:
            g_fold = self.g_fold[idx] + 'N'*(len(self.dr_seq[idx]) + max_len - len(self.g_fold[idx]))
        g_fold = g_fold[len(self.dr_seq[idx]):]
        
        if len(g_fold) != max_len:
            assert False
        # oh encoding
        g_seq = np.array(seq_one_hot_code(g_seq))
        t_seq = np.array(seq_one_hot_code(t_seq))
        g_fold = np.array(fold_one_hot_code(g_fold))

        g_seq = torch.from_numpy(g_seq).float()
        t_seq = torch.from_numpy(t_seq).float()
        g_fold = torch.from_numpy(g_fold).float()

        g_mfe = torch.from_numpy(self.g_mfe[idx]).float()
        h_mfe = torch.from_numpy(self.h_mfe[idx]).float()
        icshape = torch.from_numpy(self.icshape[idx]).float()
        binding_p = torch.from_numpy(binding_p).float()
        norm_label = torch.from_numpy(self.norm_label[idx]).float()
        p1 = torch.from_numpy(self.p1[idx]).float()
        p2 = torch.from_numpy(self.p2[idx]).float()
        rpkm = torch.from_numpy(self.rpkm[idx]).float()
        st_relate_len = torch.from_numpy(self.st_relate_len[idx]).float()
        ed_relate_len = torch.from_numpy(self.ed_relate_len[idx]).float()
        utr5_rate = torch.from_numpy(self.utr5_rate[idx]).float()
        cds_rate = torch.from_numpy(self.cds_rate[idx]).float()
        utr3_rate = torch.from_numpy(self.utr3_rate[idx]).float()

        # self.g_seq[idx] = self.g_seq[idx].reshape(-1) # 120
        # self.t_seq[idx] = self.t_seq[idx].reshape(-1) # 120
        # self.g_fold[idx] = self.g_fold[idx].reshape(-1) # 90

        




        return g_seq, t_seq, self.extt_seq[idx], self.dr_seq[idx], g_fold, \
                g_mfe, h_mfe, icshape, binding_p, \
                norm_label, \
                p1, p2, \
                rpkm, \
                st_relate_len, ed_relate_len, \
                utr5_rate, cds_rate, utr3_rate, self.read_depth[idx], \
                self.cell_line[idx], self.target_gene[idx], \
                \
                self.csv_model[idx],self.csv_gRNA_id[idx], self.csv_ms_match_idt[idx], self.csv_gRNA_seq[idx], \
                self.csv_tRNA_seq[idx], self.csv_g_fold[idx], self.csv_icshape[idx], self.csv_binding_p[idx], \
                self.csv_rpkm[idx], self.csv_g_mfe[idx], self.csv_h_mfe[idx], \
                self.csv_st_relate_len[idx], self.csv_ed_relate_len[idx], \
                self.csv_utr5_rate[idx], self.csv_cds_rate[idx], self.csv_utr3_rate[idx], \
                self.csv_read_depth[idx], self.csv_norm_label[idx]

class BertOnehotLoader30(DataLoader):
    def __init__(self, data, max_len=30):
        self.max_len = max_len
        self.data = data

        model = 0
        gRNA_id = 1
        cell_line = 2
        target_gene = 3
        gRNA_seq = 4
        tRNA_seq = 5
        trans_id = 6
        ext_tseq = 7
        DR_seq = 8
        gRNA_mfe = 9
        DR_gRNA_fold = 10
        h_mfe = 11
        icshape = 12
        prt_binding_p = 13
        lfc = 14
        p1 = 15
        p2 = 16
        ext_p1 = 17
        ext_p2 = 18
        ms_match = 19
        ms_match_idt = 20
        norm_lfc = 21
        parental_gRNA = 22
        parental_gRNA_lfc = 23
        RPKM = 24
        st_relate_len = 25
        ed_relate_len = 26
        st_tar_trans = 27
        ed_tar_trans = 28
        cds_st = 29
        cds_ed = 30
        utr5 = 31
        cds = 32
        utr3 = 33
        utr5_rate = 34
        cds_rate = 35
        utr3_rate = 36
        read_depth = 37

        

        # useful feature
        self.g_seq = self.data[gRNA_seq].to_numpy()
        self.t_seq = self.data[tRNA_seq].to_numpy()
        self.extt_seq = self.data[ext_tseq].to_numpy()
        self.dr_seq = self.data[DR_seq].to_numpy()
        self.g_fold = self.data[DR_gRNA_fold].to_numpy()
        self.g_mfe = self.data[gRNA_mfe].to_numpy().astype(np.float32).reshape(-1,1)
        self.h_mfe = self.data[h_mfe].to_numpy().astype(np.float32).reshape(-1,1)
        self.icshape = self.data[icshape].to_numpy()
        self.binding_p = self.data[prt_binding_p].to_numpy()
        self.norm_label = self.data[norm_lfc].to_numpy().astype(np.float32).reshape(-1,1)
        self.p1 = self.data[p1].to_numpy().astype(np.float32).reshape(-1,1)
        self.p2 = self.data[p2].to_numpy().astype(np.float32).reshape(-1,1)
        self.rpkm = self.data[RPKM].to_numpy().astype(np.float32).reshape(-1,1)
        self.st_relate_len = self.data[st_relate_len].to_numpy().astype(np.float32).reshape(-1,1)
        self.ed_relate_len = self.data[ed_relate_len].to_numpy().astype(np.float32).reshape(-1,1)
        self.utr5_rate = self.data[utr5_rate].to_numpy().astype(np.float32).reshape(-1,1)
        self.cds_rate = self.data[cds_rate].to_numpy().astype(np.float32).reshape(-1,1)
        self.utr3_rate = self.data[utr3_rate].to_numpy().astype(np.float32).reshape(-1,1)
        self.read_depth = self.data[read_depth].to_numpy()
        self.label = self.data[lfc].to_numpy().astype(np.float32).reshape(-1,1)

        

        # debug feature
        self.cell_line = self.data[cell_line].to_numpy()
        self.target_gene = self.data[target_gene].to_numpy()

        # csv output
        self.csv_model = self.data[model].to_numpy()
        self.csv_gRNA_id = self.data[gRNA_id].to_numpy()
        self.csv_ms_match_idt = self.data[ms_match_idt].to_numpy()
        self.csv_gRNA_seq = self.data[gRNA_seq].to_numpy()
        self.csv_tRNA_seq = self.data[tRNA_seq].to_numpy()
        self.csv_g_fold = self.data[DR_gRNA_fold].to_numpy()
        self.csv_icshape = self.data[icshape].to_numpy()
        self.csv_binding_p = self.data[prt_binding_p].to_numpy()
        self.csv_rpkm = self.data[RPKM].to_numpy().astype(np.float32)
        self.csv_g_mfe = self.data[gRNA_mfe].to_numpy().astype(np.float32)
        self.csv_h_mfe = self.data[h_mfe].to_numpy().astype(np.float32)
        self.csv_st_relate_len = self.data[st_relate_len].to_numpy().astype(np.float32)
        self.csv_ed_relate_len = self.data[ed_relate_len].to_numpy().astype(np.float32)
        self.csv_utr5_rate = self.data[utr5_rate].to_numpy().astype(np.float32)
        self.csv_cds_rate = self.data[cds_rate].to_numpy().astype(np.float32)
        self.csv_utr3_rate = self.data[utr3_rate].to_numpy().astype(np.float32)
        self.csv_read_depth = self.data[read_depth].to_numpy()
        self.csv_norm_label = self.data[norm_lfc].to_numpy().astype(np.float32)

        # print('cell_line: ', np.unique(self.cell_line))
        # print('target_gene: ', np.unique(self.target_gene))
        device = torch.device('cuda:0')

        # extent sequence
        for idx in range(len(self.icshape)):
            p11 = int(self.p1[idx][0])-1
            p22 = int(self.p2[idx][0])
            if p22 - p11 < self.max_len:
                midp = self.max_len - (p22-p11)
                if p11 - midp >= 0:
                    p11 = p11 - midp
                    p22 = p11 + self.max_len
                else:
                    p11 = 0
                    p22 = self.max_len
            else:
                p11 = p11
                p22 = p11+self.max_len

            # icshape
            if self.icshape[idx] != 'na' and self.icshape[idx] != 'n,a':
                ics = self.icshape[idx].split(',')
                ics = ics[p11:p22]
                icss = [float(i) for i in ics]
                self.icshape[idx] = np.array(icss)
            else:
                icss = []
                for i in range(self.max_len):
                    icss.append(float(-1.))
                self.icshape[idx] = np.array(icss)

            # target RNA -> Bert
            self.t_seq[idx] = self.extt_seq[idx][p11:p22]

        tokenizer = BertTokenizer.from_pretrained('./BERT_Model', do_lower_case=False)
        model = BertModel.from_pretrained('./BERT_Model')
        model = model.to(device)
        model = model.eval()
        # print(self.t_seq.shape)
        
        self.bert_embedding = FixRNABert(list(self.t_seq), model, tokenizer, device, 3, self.max_len)

        # print(bert_embedding.shape) # (N, 28, 768)

    def __len__(self):
        return len(self.g_seq)
    
    def __getitem__(self, idx):
        max_len = self.max_len

        # print('self.binding_p: ', self.binding_p[idx])

        binding = self.binding_p[idx].split(',')
        bindings = [float(i) for i in binding]
        binding_p = np.array(bindings)

        g_seq = self.g_seq[idx]
        t_seq = self.t_seq[idx]
        extt_seq = self.extt_seq[idx]
        g_fold = self.g_fold[idx]

        if len(self.g_seq[idx]) < max_len:
            g_seq = self.g_seq[idx] + 'N'*(max_len-len(self.g_seq[idx]))
        
        # if len(self.t_seq[idx]) < max_len:
        #     t_seq = self.t_seq[idx] + 'N'*(max_len-len(self.t_seq[idx]))
        
        if len(self.g_fold[idx]) < len(self.dr_seq[idx]) + max_len:
            g_fold = self.g_fold[idx] + 'N'*(len(self.dr_seq[idx]) + max_len - len(self.g_fold[idx]))
        g_fold = g_fold[len(self.dr_seq[idx]):]
        
        if len(g_fold) != max_len:
            assert False
        # oh encoding
        g_seq = np.array(seq_one_hot_code(g_seq))
        t_seq = np.array(seq_one_hot_code(t_seq))
        # extt_seq = np.array(seq_one_hot_code(extt_seq))[:100]
        # dr_seq = np.array(seq_one_hot_code(self.dr_seq[idx]))[:30]
        g_fold = np.array(fold_one_hot_code(g_fold))

        g_seq = torch.from_numpy(g_seq).float()
        t_seq = torch.from_numpy(t_seq).float()
        # extt_seq = torch.from_numpy(extt_seq).float()
        # dr_seq = torch.from_numpy(dr_seq).float()
        g_fold = torch.from_numpy(g_fold).float()

        # print(g_seq.shape, t_seq.shape, g_fold.shape)

        g_mfe = torch.from_numpy(self.g_mfe[idx]).float()
        h_mfe = torch.from_numpy(self.h_mfe[idx]).float()
        icshape = torch.from_numpy(self.icshape[idx]).float()
        binding_p = torch.from_numpy(binding_p).float()
        norm_label = torch.from_numpy(self.norm_label[idx]).float()
        p1 = torch.from_numpy(self.p1[idx]).float()
        p2 = torch.from_numpy(self.p2[idx]).float()
        rpkm = torch.from_numpy(self.rpkm[idx]).float()
        st_relate_len = torch.from_numpy(self.st_relate_len[idx]).float()
        ed_relate_len = torch.from_numpy(self.ed_relate_len[idx]).float()
        utr5_rate = torch.from_numpy(self.utr5_rate[idx]).float()
        cds_rate = torch.from_numpy(self.cds_rate[idx]).float()
        utr3_rate = torch.from_numpy(self.utr3_rate[idx]).float()

        # self.g_seq[idx] = self.g_seq[idx].reshape(-1) # 120
        # self.t_seq[idx] = self.t_seq[idx].reshape(-1) # 120
        # self.g_fold[idx] = self.g_fold[idx].reshape(-1) # 90

        return g_seq, t_seq, self.extt_seq[idx], self.dr_seq[idx], g_fold, \
                g_mfe, h_mfe, icshape, binding_p, \
                norm_label, \
                p1, p2, \
                rpkm, \
                st_relate_len, ed_relate_len, \
                utr5_rate, cds_rate, utr3_rate, self.read_depth[idx], \
                self.cell_line[idx], self.target_gene[idx], self.bert_embedding[idx], self.label[idx], \
                \
                self.csv_model[idx],self.csv_gRNA_id[idx], self.csv_ms_match_idt[idx], self.csv_gRNA_seq[idx], \
                self.csv_tRNA_seq[idx], self.csv_g_fold[idx], self.csv_icshape[idx], self.csv_binding_p[idx], \
                self.csv_rpkm[idx], self.csv_g_mfe[idx], self.csv_h_mfe[idx], \
                self.csv_st_relate_len[idx], self.csv_ed_relate_len[idx], \
                self.csv_utr5_rate[idx], self.csv_cds_rate[idx], self.csv_utr3_rate[idx], \
                self.csv_read_depth[idx], self.csv_norm_label[idx]
        # return g_seq, t_seq, extt_seq, dr_seq, g_fold, \
        #         g_mfe, h_mfe, icshape, binding_p, \
        #         norm_label, \
        #         p1, p2, \
        #         rpkm, \
        #         st_relate_len, ed_relate_len, \
        #         utr5_rate, cds_rate, utr3_rate, 0, \
        #         0, 0, self.bert_embedding[idx], self.label[idx], \
        #         \
        #         self.csv_model[idx],self.csv_gRNA_id[idx], self.csv_ms_match_idt[idx], self.csv_gRNA_seq[idx], \
        #         self.csv_tRNA_seq[idx], self.csv_g_fold[idx], self.csv_icshape[idx], self.csv_binding_p[idx], \
        #         self.csv_rpkm[idx], self.csv_g_mfe[idx], self.csv_h_mfe[idx], \
        #         self.csv_st_relate_len[idx], self.csv_ed_relate_len[idx], \
        #         self.csv_utr5_rate[idx], self.csv_cds_rate[idx], self.csv_utr3_rate[idx], \
        #         self.csv_read_depth[idx], self.csv_norm_label[idx]


class BertOnehotLoader300(DataLoader):
    def __init__(self, data, max_len=30):
        self.max_len = max_len
        self.data = data

        model = 0
        gRNA_id = 1
        cell_line = 2
        target_gene = 3
        gRNA_seq = 4
        tRNA_seq = 5
        trans_id = 6
        ext_tseq = 7
        DR_seq = 8
        gRNA_mfe = 9
        DR_gRNA_fold = 10
        h_mfe = 11
        icshape = 12
        prt_binding_p = 13
        lfc = 14
        p1 = 15
        p2 = 16
        ext_p1 = 17
        ext_p2 = 18
        ms_match = 19
        ms_match_idt = 20
        norm_lfc = 21
        parental_gRNA = 22
        parental_gRNA_lfc = 23
        RPKM = 24
        st_relate_len = 25
        ed_relate_len = 26
        st_tar_trans = 27
        ed_tar_trans = 28
        cds_st = 29
        cds_ed = 30
        utr5 = 31
        cds = 32
        utr3 = 33
        utr5_rate = 34
        cds_rate = 35
        utr3_rate = 36
        read_depth = 37

        

        # useful feature
        self.g_seq = self.data[gRNA_seq].to_numpy()
        self.t_seq = self.data[tRNA_seq].to_numpy()
        self.extt_seq = self.data[ext_tseq].to_numpy()
        self.dr_seq = self.data[DR_seq].to_numpy()
        self.g_fold = self.data[DR_gRNA_fold].to_numpy()
        self.g_mfe = self.data[gRNA_mfe].to_numpy().astype(np.float32).reshape(-1,1)
        self.h_mfe = self.data[h_mfe].to_numpy().astype(np.float32).reshape(-1,1)
        self.icshape = self.data[icshape].to_numpy()
        self.binding_p = self.data[prt_binding_p].to_numpy()
        self.norm_label = self.data[norm_lfc].to_numpy().astype(np.float32).reshape(-1,1)
        self.p1 = self.data[p1].to_numpy().astype(np.float32).reshape(-1,1)
        self.p2 = self.data[p2].to_numpy().astype(np.float32).reshape(-1,1)
        self.rpkm = self.data[RPKM].to_numpy().astype(np.float32).reshape(-1,1)
        self.st_relate_len = self.data[st_relate_len].to_numpy().astype(np.float32).reshape(-1,1)
        self.ed_relate_len = self.data[ed_relate_len].to_numpy().astype(np.float32).reshape(-1,1)
        self.utr5_rate = self.data[utr5_rate].to_numpy().astype(np.float32).reshape(-1,1)
        self.cds_rate = self.data[cds_rate].to_numpy().astype(np.float32).reshape(-1,1)
        self.utr3_rate = self.data[utr3_rate].to_numpy().astype(np.float32).reshape(-1,1)
        self.read_depth = self.data[read_depth].to_numpy()
        self.label = self.data[lfc].to_numpy().astype(np.float32).reshape(-1,1)

        

        # debug feature
        self.cell_line = self.data[cell_line].to_numpy()
        self.target_gene = self.data[target_gene].to_numpy()

        # csv output
        self.csv_model = self.data[model].to_numpy()
        self.csv_gRNA_id = self.data[gRNA_id].to_numpy()
        self.csv_ms_match_idt = self.data[ms_match_idt].to_numpy()
        self.csv_gRNA_seq = self.data[gRNA_seq].to_numpy()
        self.csv_tRNA_seq = self.data[tRNA_seq].to_numpy()
        self.csv_g_fold = self.data[DR_gRNA_fold].to_numpy()
        self.csv_icshape = self.data[icshape].to_numpy()
        self.csv_binding_p = self.data[prt_binding_p].to_numpy()
        self.csv_rpkm = self.data[RPKM].to_numpy().astype(np.float32)
        self.csv_g_mfe = self.data[gRNA_mfe].to_numpy().astype(np.float32)
        self.csv_h_mfe = self.data[h_mfe].to_numpy().astype(np.float32)
        self.csv_st_relate_len = self.data[st_relate_len].to_numpy().astype(np.float32)
        self.csv_ed_relate_len = self.data[ed_relate_len].to_numpy().astype(np.float32)
        self.csv_utr5_rate = self.data[utr5_rate].to_numpy().astype(np.float32)
        self.csv_cds_rate = self.data[cds_rate].to_numpy().astype(np.float32)
        self.csv_utr3_rate = self.data[utr3_rate].to_numpy().astype(np.float32)
        self.csv_read_depth = self.data[read_depth].to_numpy()
        self.csv_norm_label = self.data[norm_lfc].to_numpy().astype(np.float32)

        # print('cell_line: ', np.unique(self.cell_line))
        # print('target_gene: ', np.unique(self.target_gene))
        device = torch.device('cuda:0')

        minn = 300
        # extent sequence
        for idx in range(len(self.icshape)):
            p11 = int(self.p1[idx][0])-1
            p22 = int(self.p2[idx][0])
            # if p22 - p11 < self.max_len:
            #     midp = self.max_len - (p22-p11)
            #     if p11 - midp >= 0:
            #         p11 = p11 - midp
            #         p22 = p11 + self.max_len
            #     else:
            #         p11 = 0
            #         p22 = self.max_len
            # else:
            #     p11 = p11
            #     p22 = p11+self.max_len

            '''
            minn = 230
            # target seq            
            for _ in range(minn - len(self.extt_seq[idx])):
                self.extt_seq[idx] = self.extt_seq[idx] + 'N'
            
            self.t_seq[idx] = self.extt_seq[idx]
            # icshape
            if self.icshape[idx] != 'na' and self.icshape[idx] != 'n,a':
                ics = self.icshape[idx].split(',')
                if len(ics) < minn:
                    for _ in range(minn - len(ics)):
                        ics.append(float(-1.))
                else:
                    ics = ics[:minn]

                icss = [float(i) for i in ics]
                self.icshape[idx] = np.array(icss)
            else:
                icss = []
                for i in range(minn):
                    icss.append(float(-1.))
                self.icshape[idx] = np.array(icss)
            '''
            # 101 AUPRC 0.7359 0.8879
            minn = 111
            if self.icshape[idx] != 'na' and self.icshape[idx] != 'n,a':
                ics = self.icshape[idx].split(',')
                if p22 <= minn:
                    ics = ics[:minn]
                elif p11 <= minn:
                    ics = ics[p22-minn:p22]
                else:
                    ics = ics[len(ics)-minn:]
                # ics = ics[p11:p22]
                icss = [float(i) for i in ics]
                self.icshape[idx] = np.array(icss)
            else:
                icss = []
                for i in range(minn):
                    icss.append(float(-1.))
                self.icshape[idx] = np.array(icss)

            if p22 <= minn:
                # ics = ics[:minn]
                self.t_seq[idx] = self.extt_seq[idx][:minn]
            elif p11 <= minn:
                # ics = ics[p22-minn:p22]
                self.t_seq[idx] = self.extt_seq[idx][p22-minn:p22]
            else:
                # ics = ics[len(ics)-minn:]
                self.t_seq[idx] = self.extt_seq[idx][len(self.t_seq[idx])-minn:]
            # self.t_seq[idx] = self.extt_seq[idx][p11:p22]

        tokenizer = BertTokenizer.from_pretrained('./BERT_Model', do_lower_case=False)
        model = BertModel.from_pretrained('./BERT_Model')
        model = model.to(device)
        model = model.eval()
        # print(self.t_seq.shape)
        
        self.bert_embedding = FixRNABert(list(self.g_seq), model, tokenizer, device, 3, 30)

        # print(bert_embedding.shape) # (N, 28, 768)

    def __len__(self):
        return len(self.g_seq)
    
    def __getitem__(self, idx):
        max_len = self.max_len

        # print('self.binding_p: ', self.binding_p[idx])

        binding = self.binding_p[idx].split(',')
        bindings = [float(i) for i in binding]
        binding_p = np.array(bindings)

        g_seq = self.g_seq[idx]
        t_seq = self.t_seq[idx]
        extt_seq = self.extt_seq[idx]
        g_fold = self.g_fold[idx]

        if len(self.g_seq[idx]) < max_len:
            g_seq = self.g_seq[idx] + 'N'*(max_len-len(self.g_seq[idx]))
        
        # if len(self.t_seq[idx]) < max_len:
        #     t_seq = self.t_seq[idx] + 'N'*(max_len-len(self.t_seq[idx]))
        
        if len(self.g_fold[idx]) < len(self.dr_seq[idx]) + max_len:
            g_fold = self.g_fold[idx] + 'N'*(len(self.dr_seq[idx]) + max_len - len(self.g_fold[idx]))
        g_fold = g_fold[len(self.dr_seq[idx]):]
        
        if len(g_fold) != max_len:
            assert False
        # oh encoding
        g_seq = np.array(seq_one_hot_code(g_seq))
        t_seq = np.array(seq_one_hot_code(t_seq))
        # extt_seq = np.array(seq_one_hot_code(extt_seq))[:100]
        # dr_seq = np.array(seq_one_hot_code(self.dr_seq[idx]))[:30]
        g_fold = np.array(fold_one_hot_code(g_fold))

        g_seq = torch.from_numpy(g_seq).float()
        t_seq = torch.from_numpy(t_seq).float()
        # extt_seq = torch.from_numpy(extt_seq).float()
        # dr_seq = torch.from_numpy(dr_seq).float()
        g_fold = torch.from_numpy(g_fold).float()

        # print(g_seq.shape, t_seq.shape, g_fold.shape)

        g_mfe = torch.from_numpy(self.g_mfe[idx]).float()
        h_mfe = torch.from_numpy(self.h_mfe[idx]).float()
        icshape = torch.from_numpy(self.icshape[idx]).float()
        binding_p = torch.from_numpy(binding_p).float()
        norm_label = torch.from_numpy(self.norm_label[idx]).float()
        p1 = torch.from_numpy(self.p1[idx]).float()
        p2 = torch.from_numpy(self.p2[idx]).float()
        rpkm = torch.from_numpy(self.rpkm[idx]).float()
        st_relate_len = torch.from_numpy(self.st_relate_len[idx]).float()
        ed_relate_len = torch.from_numpy(self.ed_relate_len[idx]).float()
        utr5_rate = torch.from_numpy(self.utr5_rate[idx]).float()
        cds_rate = torch.from_numpy(self.cds_rate[idx]).float()
        utr3_rate = torch.from_numpy(self.utr3_rate[idx]).float()

        # self.g_seq[idx] = self.g_seq[idx].reshape(-1) # 120
        # self.t_seq[idx] = self.t_seq[idx].reshape(-1) # 120
        # self.g_fold[idx] = self.g_fold[idx].reshape(-1) # 90
        # print(g_seq.shape)
        # print(t_seq.shape)
        # print(self.extt_seq[idx].shape)
        # assert False

        return g_seq, t_seq, self.extt_seq[idx], self.dr_seq[idx], g_fold, \
                g_mfe, h_mfe, icshape, binding_p, \
                norm_label, \
                p1, p2, \
                rpkm, \
                st_relate_len, ed_relate_len, \
                utr5_rate, cds_rate, utr3_rate, self.read_depth[idx], \
                self.cell_line[idx], self.target_gene[idx], self.bert_embedding[idx], self.label[idx], \
                \
                self.csv_model[idx],self.csv_gRNA_id[idx], self.csv_ms_match_idt[idx], self.csv_gRNA_seq[idx], \
                self.csv_tRNA_seq[idx], self.csv_g_fold[idx], self.csv_icshape[idx], self.csv_binding_p[idx], \
                self.csv_rpkm[idx], self.csv_g_mfe[idx], self.csv_h_mfe[idx], \
                self.csv_st_relate_len[idx], self.csv_ed_relate_len[idx], \
                self.csv_utr5_rate[idx], self.csv_cds_rate[idx], self.csv_utr3_rate[idx], \
                self.csv_read_depth[idx], self.csv_norm_label[idx]
        # return g_seq, t_seq, extt_seq, dr_seq, g_fold, \
        #         g_mfe, h_mfe, icshape, binding_p, \
        #         norm_label, \
        #         p1, p2, \
        #         rpkm, \
        #         st_relate_len, ed_relate_len, \
        #         utr5_rate, cds_rate, utr3_rate, 0, \
        #         0, 0, self.bert_embedding[idx], self.label[idx], \
        #         \
        #         self.csv_model[idx],self.csv_gRNA_id[idx], self.csv_ms_match_idt[idx], self.csv_gRNA_seq[idx], \
        #         self.csv_tRNA_seq[idx], self.csv_g_fold[idx], self.csv_icshape[idx], self.csv_binding_p[idx], \
        #         self.csv_rpkm[idx], self.csv_g_mfe[idx], self.csv_h_mfe[idx], \
        #         self.csv_st_relate_len[idx], self.csv_ed_relate_len[idx], \
        #         self.csv_utr5_rate[idx], self.csv_cds_rate[idx], self.csv_utr3_rate[idx], \
        #         self.csv_read_depth[idx], self.csv_norm_label[idx]

