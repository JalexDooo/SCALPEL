import torch
import torch.utils.data
import numpy as np
import pandas as pd
import os, sys, h5py
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score, confusion_matrix
from scipy.stats import ks_2samp, norm, pearsonr, spearmanr

import os, sys
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.transform import resize as imresize
from PIL import Image


def roc_and_prc_from_lfc(observed_label, predictions):
    """
    Compute ROC and PRC points
    :param observed_label: observed response
    :param predictions: predictions
    :return: ROC and PRC
    """
    if observed_label is None or len(np.unique(observed_label)) != 2:
        return (None, None), (None, None)

    # ensure predictions positively correlate with labels or metrics will break (e.g. LFC needs to be sign flipped)
    predictions = np.sign(pearsonr(predictions, observed_label)[0]) * predictions
    # ROC and PRC values
    fpr, tpr, _ = roc_curve(observed_label, predictions)
    precision, recall, _ = precision_recall_curve(observed_label, predictions)

    return (fpr, tpr), (precision, recall)

def tfnp(label, prediction):
    try:
        tn, fp, fn, tp = confusion_matrix(label, prediction).ravel()
    except Exception:
        tp, tn, fp, fn =0,0,0,0
    
    return tp, tn, fp, fn

def Sensitivity(predict, target, par=0.8):
    ndim = np.ndim(target)
    if ndim == 2:
        predict=predict[:,0]
        target = target[:,0]
    predict = (predict>par)*int(1)
    target = (target>par)*int(1)
    tp, tn, fp, fn = tfnp(target, predict)
    sen = tp/(tp+fn+1e-7)
    return sen

def Specificity(predict, target, par=0.8):
    ndim = np.ndim(target)
    if ndim == 2:
        predict=predict[:,0]
        target = target[:,0]
    predict = (predict>par)*int(1)
    target = (target>par)*int(1)
    tp, tn, fp, fn = tfnp(target, predict)
    spe = tn/(tn+fp+1e-7)
    return spe

def Accuracy(predict, target, par=0.8):
    ndim = np.ndim(target)
    if ndim == 2:
        predict=predict[:,0]
        target = target[:,0]
    predict = (predict>par)*int(1)
    target = (target>par)*int(1)

    inters = predict*target # predict=1 & label=1
    uninters = (predict==0)*(target==0) # predict=0 & label=0
    acc = (inters.sum()+uninters.sum())/(inters.shape[0])
    return acc

def AUC_SC(predict, target):
    ndim = np.ndim(target)
    if ndim == 2:
        predict=predict[:,0]
        target = target[:,0]
    fpr, tpr, thretholds = roc_curve(target, predict)
    score = auc(fpr, tpr)
    curve = [fpr, tpr]
    return score, curve


def inference(args, model, device, test_loader):
    model.eval()
    p_all = []
    with torch.no_grad():
        for batch_idx, (x0, y0) in enumerate(test_loader):
            x, y = x0.float().to(device), y0.to(device).float()
            output = model(x)
            prob = torch.sigmoid(output)

            p_np = prob.to(device='cpu').numpy()
            p_all.append(p_np)

    p_all = np.concatenate(p_all)
    return p_all


def normalize_pwm(pwm, factor=None, MAX=None):
    if MAX is None:
        MAX = np.max(np.abs(pwm))
    pwm = pwm/MAX
    if factor:
        pwm = np.exp(pwm*factor)
    norm = np.outer(np.ones(pwm.shape[0]), np.sum(np.abs(pwm), axis=0))
    return pwm/norm

def get_nt_height(pwm, height, norm):

    def entropy(p):
        s = 0
        for i in range(len(p)):
            if p[i] > 0:
                s -= p[i]*np.log2(p[i])
        return s

    num_nt, num_seq = pwm.shape
    heights = np.zeros((num_nt,num_seq))
    for i in range(num_seq):
        if norm == 1:
            total_height = height
        else:
            total_height = (np.log2(num_nt) - entropy(pwm[:, i]))*height
        
        heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height*2))

    return heights.astype(int)

def seq_logo(pwm, height=30, nt_width=10, norm=0, alphabet='rna', colormap='standard'):
    acgu_path = './acgu.npz'
    chars = np.load(acgu_path,allow_pickle=True)['data']
    heights = get_nt_height(pwm, height, norm)
    num_nt, num_seq = pwm.shape
    width = np.ceil(nt_width*num_seq).astype(int)
    
    max_height = height*2
    logo = np.ones((max_height, width, 3)).astype(int)*255
    for i in range(num_seq):
        nt_height = np.sort(heights[:,i])
        index = np.argsort(heights[:,i])
        remaining_height = np.sum(heights[:,i])
        offset = max_height-remaining_height

        for j in range(num_nt):
            if nt_height[j] <=0 :
                continue
            # resized dimensions of image
            nt_img = imresize(chars[index[j]], output_shape=(nt_height[j], nt_width))*255
            # if j==0:
            #     print(nt_img[:,:,0])
             #   print(chars[index[j]][:,:,0])
            # determine location of image
            height_range = range(remaining_height-nt_height[j], remaining_height)
            width_range = range(i*nt_width, i*nt_width+nt_width)
            # 'annoying' way to broadcast resized nucleotide image
            if height_range:
                for k in range(3):
                    for m in range(len(width_range)):
                        logo[height_range+offset, width_range[m],k] = nt_img[:,m,k]

            remaining_height -= nt_height[j]

    return logo.astype(np.uint8)

def plot_saliency(X, W, nt_width=100, norm_factor=3, str_null=None, outdir="results/"):
    # filter out zero-padding
    plot_index = np.where(np.sum(X[:4,:], axis=0)!=0)[0]
    num_nt = len(plot_index)
    print('nt: ', num_nt)
    trace_width = num_nt*nt_width # 96*100
    trace_height = 400
    
    seq_str_mode = False
    if X.shape[0]>4:
        seq_str_mode = True
        assert str_null is not None, "Null region is not provided."

    # sequence logo
    img_seq_raw = seq_logo(X[:4, plot_index], height=nt_width, nt_width=nt_width)

    if seq_str_mode:
        # structure line
        str_raw = X[4, plot_index]
        if str_null.sum() > 0:
            str_raw[str_null.T==1] = -0.01

        line_str_raw = np.zeros(trace_width)
        for v in range(str_raw.shape[0]):
            line_str_raw[v*nt_width:(v+1)*nt_width] = (1-str_raw[v])*trace_height 
            # i+=1
    
    # sequence saliency logo
    seq_sal = normalize_pwm(W[:4, plot_index], factor=norm_factor)
    print('seq_sal: ', seq_sal.shape)
    img_seq_sal_logo = seq_logo(seq_sal, height=nt_width*5, nt_width=nt_width)
    print('img_logo: ', img_seq_sal_logo.shape)
    img_seq_sal = imresize(W[:4, plot_index], output_shape=(trace_height, trace_width))

    if seq_str_mode:
        # structure saliency logo
        str_sal = W[-1, plot_index].reshape(1,-1)
        img_str_sal = imresize(str_sal, output_shape=(trace_height, trace_width))

    # plot    
    fig = plt.figure(figsize=(10.1,2))
    gs = gridspec.GridSpec(nrows=4, ncols=1, height_ratios=[2.5, 1, 0.5, 1])
    cmap_reversed = mpl.cm.get_cmap('jet')

    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    ax.imshow(img_seq_sal_logo)
    plt.text(x=trace_width-400,y=10, s='CreatNet', fontsize=4)

    ax = fig.add_subplot(gs[1, 0])
    ax.axis('off')
    ax.imshow(img_seq_sal, cmap=cmap_reversed)

    ax = fig.add_subplot(gs[2, 0]) 
    ax.axis('off')
    ax.imshow(img_seq_raw)

    if seq_str_mode:
        ax = fig.add_subplot(gs[3, 0]) 
        ax.axis('off')
        ax.imshow(img_str_sal, cmap=cmap_reversed)
        ax.plot(line_str_raw, '-', color='r', linewidth=1, scalex=False, scaley=False)
        
        # plot balck line to hide the -1(NULL structure score)
        x = (np.zeros(trace_width) + (1+0.01))*trace_height  +1.5
        ax.plot(x, '-', color='white', linewidth=1.2, scalex=False, scaley=False)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # save figure
    filepath = outdir
    fig.savefig(filepath, format='png', dpi=300, bbox_inches='tight')
    plt.close('all')






def read_csv(path):
    print('Read csv file: ', path)
    df = pd.read_csv(path, sep=',', encoding='GBK', header=None)
    # gRNA_id,Gene,gRNA_sequence,target_sequence,transcript_ID,extended_target,DR_sequence,gRNA_MFE,DR_gRNA_fold,hybrid_MFE,icshape score,protein_binding_probability,LFC,p1,p2
    # models,gRNA_id,cell_line,target_gene,gRNA_sequence,targetsequence,transcript_id,extend_target,DR_sequence,gRNA_MFE,DR_gRNA_fold,hybridMFE,icshape score,protein_binding_p,LFC,p1(start_1),p2,extend_p1,extend_p2,refseq,raw ratio,relative_ratio,binary_relative_ratio_061f,binary_relative_ratio_075f,ratio045_cutoff_binary_relative_ratio,ratio,old_relative_ratio,old_binary_relative_ratio_gene20,position,is_5UTR,UTR5_position,is_CDS,CDS_position,is_3UTR,UTR3_position,RNAseq2_relative,RNAseq3_relative,RNAseq7_relative,RNAseq8_relative,np_vivo_ic_has_data,np_vivo_ic_sum,blast_f24_mis3_e1_20_match_num,pos,vienna_2_T37,vienna_2_T60,vienna_2_T70,contrafold_2,eternafold,refseq_target_transcript_percent,ensembl_target_transcript_percent,absolute_position_start,nearby_seq_all_5,nearby_seq_all_10,nearby_seq_all_15,nearby_seq_all_20,GC_content,linearfold_dr_flag,linearfold_vals,linearfold_vals_7win,linearfold_vals_23win,target unfold energy,target unfold energy_7win,target unfold energy_23win,bad guide_bottom20%,bad guide_bottom10%,bad guide_90th_pct,bad guide_95th_pct,bins,ParentalGuide,end,start,GuideMidPoint,Type,OffTarget,Perturbation,Rfon_GuideScore,Rfon_StandardizedGuideScore,Rfon_Quartile,Annotation,Exon,Dist2Junction_5p,Dist2Junction_3p,fold.gene,fold.targetsite,fold.gRNA,TargetSeqContext,Mismatch,MatchPos,loc,g_quad,direct_repeat,hybrid_mfe_1_23,hybMFE_15.9,hybMFE_3.12,log_unpaired,log10_unpaired_p11,log10_unpaired_p19,log10_unpaired_p25,Titration,Other,RBP,TF,Controls,HAP1.gene.TPM,HEKTE.gene.TPM,MismatchPosition,MismatchIdentity

    # a = 'models,gRNA_id,cell_line,target_gene,gRNA_sequence,targetsequence,transcript_id,extend_target,DR_sequence,gRNA_MFE,DR_gRNA_fold,hybridMFE,icshape score,protein_binding_p,LFC,p1(start_1),p2,extend_p1,extend_p2,refseq,raw ratio,relative_ratio,binary_relative_ratio_061f,binary_relative_ratio_075f,ratio045_cutoff_binary_relative_ratio,ratio,old_relative_ratio,old_binary_relative_ratio_gene20,position,is_5UTR,UTR5_position,is_CDS,CDS_position,is_3UTR,UTR3_position,RNAseq2_relative,RNAseq3_relative,RNAseq7_relative,RNAseq8_relative,np_vivo_ic_has_data,np_vivo_ic_sum,blast_f24_mis3_e1_20_match_num,pos,vienna_2_T37,vienna_2_T60,vienna_2_T70,contrafold_2,eternafold,refseq_target_transcript_percent,ensembl_target_transcript_percent,absolute_position_start,nearby_seq_all_5,nearby_seq_all_10,nearby_seq_all_15,nearby_seq_all_20,GC_content,linearfold_dr_flag,linearfold_vals,linearfold_vals_7win,linearfold_vals_23win,target unfold energy,target unfold energy_7win,target unfold energy_23win,bad guide_bottom20%,bad guide_bottom10%,bad guide_90th_pct,bad guide_95th_pct,bins,ParentalGuide,end,start,GuideMidPoint,Type,OffTarget,Perturbation,Rfon_GuideScore,Rfon_StandardizedGuideScore,Rfon_Quartile,Annotation,Exon,Dist2Junction_5p,Dist2Junction_3p,fold.gene,fold.targetsite,fold.gRNA,TargetSeqContext,Mismatch,MatchPos,loc,g_quad,direct_repeat,hybrid_mfe_1_23,hybMFE_15.9,hybMFE_3.12,log_unpaired,log10_unpaired_p11,log10_unpaired_p19,log10_unpaired_p25,Titration,Other,RBP,TF,Controls,HAP1.gene.TPM,HEKTE.gene.TPM,MismatchPosition,MismatchIdentity,Normalized_LFC'
    # a = a.split(',')
    # print(a)
    # print(len(a))
    # print(a[4])

    # assert False

    df = df.iloc[1:]
    
    g_seq = 4
    t_seq = 5
    ext_seq = 7
    gRNA_MFE = 9
    # DR_gRNA_fold = 10
    hybrid_MFE = 11
    icshape = 12
    binding_prob = 13
    label = 14
    st = 15
    ed = 16
    norm_label = 107


    seq1 = df[g_seq].to_numpy()
    seq2 = df[t_seq].to_numpy()
    seq3 = df[ext_seq].to_numpy()
    mfe1 = df[gRNA_MFE].to_numpy().astype(np.float32).reshape(-1,1)
    mfe2 = df[hybrid_MFE].to_numpy().astype(np.float32).reshape(-1,1)
    shape = df[icshape].to_numpy()
    prob = df[binding_prob].to_numpy()
    eff = df[norm_label].to_numpy().astype(np.float32).reshape(-1,1)
    p1 = df[st].to_numpy().astype(np.float32).reshape(-1,1)
    p2 = df[ed].to_numpy().astype(np.float32).reshape(-1,1)

    # rnac_set  = df[Type].to_numpy()
    # sequences = df[Seq].to_numpy()
    # structs  = df[Str].to_numpy()
    # targets   = df[Score].to_numpy().astype(np.float32).reshape(-1,1)
    return seq1, seq2, seq3, mfe1, mfe2, shape, prob, eff, p1, p2

def read_csv_rpkm(path):
    print('Read csv file: ', path)
    df = pd.read_csv(path, sep=',', encoding='GBK', header=None)
    # gRNA_id,Gene,gRNA_sequence,target_sequence,transcript_ID,extended_target,DR_sequence,gRNA_MFE,DR_gRNA_fold,hybrid_MFE,icshape score,protein_binding_probability,LFC,p1,p2
    # models,gRNA_id,cell_line,target_gene,gRNA_sequence,targetsequence,transcript_id,extend_target,DR_sequence,gRNA_MFE,DR_gRNA_fold,hybridMFE,icshape score,protein_binding_p,LFC,p1(start_1),p2,extend_p1,extend_p2,refseq,raw ratio,relative_ratio,binary_relative_ratio_061f,binary_relative_ratio_075f,ratio045_cutoff_binary_relative_ratio,ratio,old_relative_ratio,old_binary_relative_ratio_gene20,position,is_5UTR,UTR5_position,is_CDS,CDS_position,is_3UTR,UTR3_position,RNAseq2_relative,RNAseq3_relative,RNAseq7_relative,RNAseq8_relative,np_vivo_ic_has_data,np_vivo_ic_sum,blast_f24_mis3_e1_20_match_num,pos,vienna_2_T37,vienna_2_T60,vienna_2_T70,contrafold_2,eternafold,refseq_target_transcript_percent,ensembl_target_transcript_percent,absolute_position_start,nearby_seq_all_5,nearby_seq_all_10,nearby_seq_all_15,nearby_seq_all_20,GC_content,linearfold_dr_flag,linearfold_vals,linearfold_vals_7win,linearfold_vals_23win,target unfold energy,target unfold energy_7win,target unfold energy_23win,bad guide_bottom20%,bad guide_bottom10%,bad guide_90th_pct,bad guide_95th_pct,bins,ParentalGuide,end,start,GuideMidPoint,Type,OffTarget,Perturbation,Rfon_GuideScore,Rfon_StandardizedGuideScore,Rfon_Quartile,Annotation,Exon,Dist2Junction_5p,Dist2Junction_3p,fold.gene,fold.targetsite,fold.gRNA,TargetSeqContext,Mismatch,MatchPos,loc,g_quad,direct_repeat,hybrid_mfe_1_23,hybMFE_15.9,hybMFE_3.12,log_unpaired,log10_unpaired_p11,log10_unpaired_p19,log10_unpaired_p25,Titration,Other,RBP,TF,Controls,HAP1.gene.TPM,HEKTE.gene.TPM,MismatchPosition,MismatchIdentity

    # a = 'models,gRNA_id,cell_line,target_gene,gRNA_sequence,targetsequence,transcript_id,extend_target,DR_sequence,gRNA_MFE,DR_gRNA_fold,hybridMFE,icshape score,protein_binding_p,LFC,p1(start_1),p2,extend_p1,extend_p2,refseq,raw ratio,relative_ratio,binary_relative_ratio_061f,binary_relative_ratio_075f,ratio045_cutoff_binary_relative_ratio,ratio,old_relative_ratio,old_binary_relative_ratio_gene20,position,is_5UTR,UTR5_position,is_CDS,CDS_position,is_3UTR,UTR3_position,RNAseq2_relative,RNAseq3_relative,RNAseq7_relative,RNAseq8_relative,np_vivo_ic_has_data,np_vivo_ic_sum,blast_f24_mis3_e1_20_match_num,pos,vienna_2_T37,vienna_2_T60,vienna_2_T70,contrafold_2,eternafold,refseq_target_transcript_percent,ensembl_target_transcript_percent,absolute_position_start,nearby_seq_all_5,nearby_seq_all_10,nearby_seq_all_15,nearby_seq_all_20,GC_content,linearfold_dr_flag,linearfold_vals,linearfold_vals_7win,linearfold_vals_23win,target unfold energy,target unfold energy_7win,target unfold energy_23win,bad guide_bottom20%,bad guide_bottom10%,bad guide_90th_pct,bad guide_95th_pct,bins,ParentalGuide,end,start,GuideMidPoint,Type,OffTarget,Perturbation,Rfon_GuideScore,Rfon_StandardizedGuideScore,Rfon_Quartile,Annotation,Exon,Dist2Junction_5p,Dist2Junction_3p,fold.gene,fold.targetsite,fold.gRNA,TargetSeqContext,Mismatch,MatchPos,loc,g_quad,direct_repeat,hybrid_mfe_1_23,hybMFE_15.9,hybMFE_3.12,log_unpaired,log10_unpaired_p11,log10_unpaired_p19,log10_unpaired_p25,Titration,Other,RBP,TF,Controls,HAP1.gene.TPM,HEKTE.gene.TPM,MismatchPosition,MismatchIdentity,Normalized_LFC'
    # a = a.split(',')
    # print(a)
    # print(len(a))
    # print(a[4])

    # assert False

    df = df.iloc[1:]
    
    g_seq = 4
    t_seq = 5
    ext_seq = 7
    gRNA_MFE = 9
    # DR_gRNA_fold = 10
    hybrid_MFE = 11
    icshape = 12
    binding_prob = 13
    label = 14
    st = 15
    ed = 16
    norm_label = 107


    seq1 = df[g_seq].to_numpy()
    seq2 = df[t_seq].to_numpy()
    seq3 = df[ext_seq].to_numpy()
    mfe1 = df[gRNA_MFE].to_numpy().astype(np.float32).reshape(-1,1)
    mfe2 = df[hybrid_MFE].to_numpy().astype(np.float32).reshape(-1,1)
    shape = df[icshape].to_numpy()
    prob = df[binding_prob].to_numpy()
    eff = df[norm_label].to_numpy().astype(np.float32).reshape(-1,1)
    p1 = df[st].to_numpy().astype(np.float32).reshape(-1,1)
    p2 = df[ed].to_numpy().astype(np.float32).reshape(-1,1)

    # rnac_set  = df[Type].to_numpy()
    # sequences = df[Seq].to_numpy()
    # structs  = df[Str].to_numpy()
    # targets   = df[Score].to_numpy().astype(np.float32).reshape(-1,1)
    return seq1, seq2, seq3, mfe1, mfe2, shape, prob, eff, p1, p2





class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


# (bert_embedding1, bert_embedding2, structure, probs, mfe1, mfe2, valid_frac=0.2)
def split_dataset(bert_embedding1, bert_embedding2, structure, probs, mfe1, mfe2, eff, valid_frac=0.2):
    
    ind0 = np.where(eff<-0.5)[0]
    ind1 = np.where(eff>=-0.5)[0]
    
    n_neg = int(len(ind0)*valid_frac)
    n_pos = int(len(ind1)*valid_frac)

    shuf_neg = np.random.permutation(len(ind0))
    shuf_pos = np.random.permutation(len(ind1))

    X_train_bert1 = np.concatenate((bert_embedding1[ind1[shuf_pos[n_pos:]]], bert_embedding1[ind0[shuf_neg[n_neg:]]]))
    X_train_bert2 = np.concatenate((bert_embedding2[ind1[shuf_pos[n_pos:]]], bert_embedding2[ind0[shuf_neg[n_neg:]]]))
    X_train_structure = np.concatenate((structure[ind1[shuf_pos[n_pos:]]], structure[ind0[shuf_neg[n_neg:]]]))
    X_train_probs = np.concatenate((probs[ind1[shuf_pos[n_pos:]]], probs[ind0[shuf_neg[n_neg:]]]))
    X_train_mfe1 = np.concatenate((mfe1[ind1[shuf_pos[n_pos:]]], mfe1[ind0[shuf_neg[n_neg:]]]))
    X_train_mfe2 = np.concatenate((mfe2[ind1[shuf_pos[n_pos:]]], mfe2[ind0[shuf_neg[n_neg:]]]))

    Y_train = np.concatenate((eff[ind1[shuf_pos[n_pos:]]], eff[ind0[shuf_neg[n_neg:]]]))
    train = (X_train_bert1, X_train_bert2, X_train_structure, X_train_probs, X_train_mfe1, X_train_mfe2, Y_train)

    X_test_bert1 = np.concatenate((bert_embedding1[ind1[shuf_pos[:n_pos]]], bert_embedding1[ind0[shuf_neg[:n_neg]]]))
    X_test_bert2 = np.concatenate((bert_embedding2[ind1[shuf_pos[:n_pos]]], bert_embedding2[ind0[shuf_neg[:n_neg]]]))
    X_test_structure = np.concatenate((structure[ind1[shuf_pos[:n_pos]]], structure[ind0[shuf_neg[:n_neg]]]))
    X_test_probs = np.concatenate((probs[ind1[shuf_pos[:n_pos]]], probs[ind0[shuf_neg[:n_neg]]]))
    X_test_mfe1 = np.concatenate((mfe1[ind1[shuf_pos[:n_pos]]], mfe1[ind0[shuf_neg[:n_neg]]]))
    X_test_mfe2 = np.concatenate((mfe2[ind1[shuf_pos[:n_pos]]], mfe2[ind0[shuf_neg[:n_neg]]]))

    Y_test = np.concatenate((eff[ind1[shuf_pos[:n_pos]]], eff[ind0[shuf_neg[:n_neg]]]))
    test = (X_test_bert1, X_test_bert2, X_test_structure, X_test_probs, X_test_mfe1, X_test_mfe2, Y_test)

    return train, test

def split_dataset1(bert_embedding1, structure, probs, mfe1, mfe2, eff, valid_frac=0.2):
    
    ind0 = np.where(eff<-0.5)[0]
    ind1 = np.where(eff>=-0.5)[0]
    
    n_neg = int(len(ind0)*valid_frac)
    n_pos = int(len(ind1)*valid_frac)

    shuf_neg = np.random.permutation(len(ind0))
    shuf_pos = np.random.permutation(len(ind1))

    X_train_bert1 = np.concatenate((bert_embedding1[ind1[shuf_pos[n_pos:]]], bert_embedding1[ind0[shuf_neg[n_neg:]]]))
    # X_train_bert2 = np.concatenate((bert_embedding2[ind1[shuf_pos[n_pos:]]], bert_embedding2[ind0[shuf_neg[n_neg:]]]))
    X_train_structure = np.concatenate((structure[ind1[shuf_pos[n_pos:]]], structure[ind0[shuf_neg[n_neg:]]]))
    X_train_probs = np.concatenate((probs[ind1[shuf_pos[n_pos:]]], probs[ind0[shuf_neg[n_neg:]]]))
    X_train_mfe1 = np.concatenate((mfe1[ind1[shuf_pos[n_pos:]]], mfe1[ind0[shuf_neg[n_neg:]]]))
    X_train_mfe2 = np.concatenate((mfe2[ind1[shuf_pos[n_pos:]]], mfe2[ind0[shuf_neg[n_neg:]]]))

    Y_train = np.concatenate((eff[ind1[shuf_pos[n_pos:]]], eff[ind0[shuf_neg[n_neg:]]]))
    train = (X_train_bert1, X_train_structure, X_train_probs, X_train_mfe1, X_train_mfe2, Y_train)

    X_test_bert1 = np.concatenate((bert_embedding1[ind1[shuf_pos[:n_pos]]], bert_embedding1[ind0[shuf_neg[:n_neg]]]))
    # X_test_bert2 = np.concatenate((bert_embedding2[ind1[shuf_pos[:n_pos]]], bert_embedding2[ind0[shuf_neg[:n_neg]]]))
    X_test_structure = np.concatenate((structure[ind1[shuf_pos[:n_pos]]], structure[ind0[shuf_neg[:n_neg]]]))
    X_test_probs = np.concatenate((probs[ind1[shuf_pos[:n_pos]]], probs[ind0[shuf_neg[:n_neg]]]))
    X_test_mfe1 = np.concatenate((mfe1[ind1[shuf_pos[:n_pos]]], mfe1[ind0[shuf_neg[:n_neg]]]))
    X_test_mfe2 = np.concatenate((mfe2[ind1[shuf_pos[:n_pos]]], mfe2[ind0[shuf_neg[:n_neg]]]))

    Y_test = np.concatenate((eff[ind1[shuf_pos[:n_pos]]], eff[ind0[shuf_neg[:n_neg]]]))
    test = (X_test_bert1, X_test_structure, X_test_probs, X_test_mfe1, X_test_mfe2, Y_test)

    return train, test




def seq2kmer_bert(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    seq_length = len(seq)
    kmer = [seq[x:x + k] for x in range(seq_length - k + 1)]
    kmers = " ".join(kmer)
    return kmers

def seq2kmer_length_bert(seq, k, max_length):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    seq_length = len(seq)
    kmer = [seq[x:x + k] for x in range(seq_length - k + 1)]
    
    if len(kmer) < max_length:
        for _ in range(max_length-len(kmer)):
            kmer.append('[PAD]')
    kmers = " ".join(kmer)
    return kmers

def circRNA_Bert(dataloader, model, tokenizer, device):
    features = []
    seq = []
    for sequences in dataloader:
        seq.append(sequences)

        ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        token_type_ids = torch.tensor(ids['token_type_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
            # print(embedding.shape)
        embedding = embedding.cpu().numpy()

        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len - 1]
            # print(seq_emd.shape)
            features.append(seq_emd)
    return features

def circRNABert(protein, model, tokenizer, device, k):
    sequences1 = protein
    sequences = []
    Bert_Feature = []
    for seq in sequences1:
        seq = seq.strip()
        ss = seq2kmer_bert(seq, k)
        sequences.append(ss)
    dataloader = torch.utils.data.DataLoader(sequences, batch_size=2048, shuffle=False)
    Features = circRNA_Bert(dataloader, model, tokenizer, device)
    for i in Features:
        Feature = np.array(i)
        Bert_Feature.append(Feature)
    bb = np.array(Bert_Feature)
    data = bb
    return data

def FixRNABert(protein, model, tokenizer, device, k, max_length=30):
    sequences1 = protein
    sequences = []
    Bert_Feature = []
    for seq in sequences1:
        seq = seq.strip()
        ss = seq2kmer_length_bert(seq, k, max_length-2)
        sequences.append(ss) # 1191
    dataloader = torch.utils.data.DataLoader(sequences, batch_size=2048, shuffle=False)
    Features = circRNA_Bert(dataloader, model, tokenizer, device)
    for i in Features:
        Feature = np.array(i)
        Bert_Feature.append(Feature)
    bb = np.array(Bert_Feature)
    data = bb
    return data


