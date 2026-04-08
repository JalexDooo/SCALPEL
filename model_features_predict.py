import models
import torch
import seaborn as sns
import dataset
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import shap
import numpy as np
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import loss_functions
from matplotlib.lines import Line2D

'''
python3 model_features_predict.py heatmap --model_des='seq' --data_des='random' --model='m4'
python3 model_features_predict.py heatmap --model_des='seq_bert' --data_des='random' --model='m4'
python3 model_features_predict.py heatmap --model_des='seq_bert_fold_mfe1_mfe2' --data_des='random' --model='m4'
python3 model_features_predict.py heatmap --model_des='seq_bert_fold_mfe1_mfe2_icshape' --data_des='random' --model='m4'
python3 model_features_predict.py heatmap --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding' --data_des='random' --model='m4'
python3 model_features_predict.py heatmap --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_rpkm' --data_des='random' --model='m4'
python3 model_features_predict.py heatmap --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_rpkm_relatelen_utrrate' --data_des='random' --model='m4'

python3 model_features_predict.py heatmap --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_rpkm_relatelen_utrrate' --data_des='cell_line:K562' --model='m4'
python3 model_features_predict.py heatmap --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_rpkm_relatelen_utrrate' --data_des='cell_line:HEK293FT' --model='m4'
python3 model_features_predict.py heatmap --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_rpkm_relatelen_utrrate' --data_des='cell_line:A375' --model='m4'
python3 model_features_predict.py heatmap --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_rpkm_relatelen_utrrate' --data_des='cell_line:HAP1' --model='m4'

python3 model_features_predict.py heatmap --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_rpkm_relatelen_utrrate' --data_des='match:0' --model='m4'
python3 model_features_predict.py heatmap --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_rpkm_relatelen_utrrate' --data_des='match:1' --model='m4'
python3 model_features_predict.py heatmap --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_rpkm_relatelen_utrrate' --data_des='match:2' --model='m4'
python3 model_features_predict.py heatmap --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_rpkm_relatelen_utrrate' --data_des='match:3' --model='m4'
python3 model_features_predict.py heatmap --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_rpkm_relatelen_utrrate' --data_des='match:4' --model='m4'
python3 model_features_predict.py heatmap --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_rpkm_relatelen_utrrate' --data_des='match:5' --model='m4'
python3 model_features_predict.py heatmap --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_rpkm_relatelen_utrrate' --data_des='match:6' --model='m4'

'''

def heatmap(model_des, data_des, model='m4'):
    model_name = model

    def hook_fn(module, input, output):

        features.append(output.detach().cpu())

    ckpt_path = 'ckpt/' + model_des + '|' + data_des + '|' + model + '/'

    model = getattr(models, model)(des=model_des)
    model = model.to('cuda:3')
    ckpt_options = ckpt_path + 'Fold:0.pth'

    model.load_state_dict(torch.load(ckpt_options))

    layer = model.cls2

    features = []
    hook = layer.register_forward_hook(hook_fn)

    origin = '/home/ai/Cas13d_gRNA_data/final/merged_data/all_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo_depth.csv'
    origin_data = pd.read_csv(origin, sep=',', encoding='GBK', header=None)
    origin_data[11] = origin_data[11].replace('na', float(0.))
    origin_data[24] = origin_data[24].replace('na', float(0.))
    origin_data = origin_data.iloc[1:]
    origin_data[14] = origin_data[14].astype(float)
    tmp = origin_data[origin_data[20] == 'na'].groupby(3)[14]
    loc = tmp.apply(lambda x: np.nanquantile(x, 0.5)).rename('location')
    neg_scale = tmp.apply(lambda x: np.nanquantile(x, 0.1)).rename('negative scale')
    pos_scale = tmp.apply(lambda x: np.nanquantile(x, 0.9)).rename('positive scale')
    params = pd.DataFrame([loc, neg_scale, pos_scale]).T
    params['scale'] = params['positive scale'] - params['negative scale']

    data = pd.read_csv('small.csv', sep=',', encoding='GBK', header=None)
    data[11] = data[11].replace('na', float(0.))
    data[24] = data[24].replace('na', float(0.))
    data = data.iloc[1:]
    data[14] = data[14].astype(float)

    for gene in data[3].unique():
        col = 14
        tar = 21
        lfc = data.loc[data[3] == gene, col].to_numpy()
        lfc = (lfc - params.loc[gene, 'location']) / params.loc[gene, 'scale']
        data.loc[data[3] == gene, tar] = lfc

    dset = getattr(dataset, 'BertOnehotLoader30')
    tdata = dset(data, max_len=30)
    tdata_loader = DataLoader(tdata, batch_size=64, num_workers=0, drop_last=True)

    lbls = []

    model.eval()
    with torch.no_grad():
        for i, d in enumerate(tdata_loader):
            lbl = d[22].to(device='cpu')
            lbls.append(lbl)
            if i >= 8:
                break
            model(d[:22], 'cuda:3')

    features = np.array(features)
    lbls = np.array(lbls)
    lbls = (lbls<=-0.5)*1.0

    features = features.reshape(8*64, -1)
    lbls = lbls.reshape(8*64, -1)

    print("Feature map shape:", features.shape)
    print(lbls.shape)

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#10A0E0', '#CF0707'])
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=lbls, cmap=custom_cmap, alpha=1)

    plt.title('t-SNE visualization')
    plt.xlabel('{}'.format(model_des), fontsize=14)

    plt.xticks([])
    plt.yticks([])

    plt.legend(handles=scatter.legend_elements()[0], labels=["Negative", "Positive"], loc='upper left', fontsize=14, borderpad=1.2, labelspacing=1.5)
    plt.savefig('./Figures/t-SNE|{}|{}.png'.format(model_des, data_des), dpi=300, bbox_inches='tight', pad_inches=0.1)

    hook.remove()

def heatmap__(model_des, data_des, model='m4'):
    model_name = model

    def hook_fn(module, input, output):

        features.append(output.detach().cpu())

    ckpt_path = 'ckpt/' + model_des + '|' + data_des + '|' + model + '/'

    model = getattr(models, model)(des=model_des)
    model = model.to('cuda:3')
    ckpt_options = ckpt_path + 'Fold:0.pth'

    model.load_state_dict(torch.load(ckpt_options))

    layer = model.cls2

    features = []
    hook = layer.register_forward_hook(hook_fn)

    origin = '/home/ai/Cas13d_gRNA_data/final/merged_data/all_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo_depth.csv'
    origin_data = pd.read_csv(origin, sep=',', encoding='GBK', header=None)
    origin_data[11] = origin_data[11].replace('na', float(0.))
    origin_data[24] = origin_data[24].replace('na', float(0.))
    origin_data = origin_data.iloc[1:]
    origin_data[14] = origin_data[14].astype(float)
    tmp = origin_data[origin_data[20] == 'na'].groupby(3)[14]
    loc = tmp.apply(lambda x: np.nanquantile(x, 0.5)).rename('location')
    neg_scale = tmp.apply(lambda x: np.nanquantile(x, 0.1)).rename('negative scale')
    pos_scale = tmp.apply(lambda x: np.nanquantile(x, 0.9)).rename('positive scale')
    params = pd.DataFrame([loc, neg_scale, pos_scale]).T
    params['scale'] = params['positive scale'] - params['negative scale']

    data = pd.read_csv('small.csv', sep=',', encoding='GBK', header=None)
    data[11] = data[11].replace('na', float(0.))
    data[24] = data[24].replace('na', float(0.))
    data = data.iloc[1:]
    data[14] = data[14].astype(float)

    for gene in data[3].unique():
        col = 14
        tar = 21
        lfc = data.loc[data[3] == gene, col].to_numpy()
        lfc = (lfc - params.loc[gene, 'location']) / params.loc[gene, 'scale']
        data.loc[data[3] == gene, tar] = lfc

    dset = getattr(dataset, 'BertOnehotLoader30')
    tdata = dset(data, max_len=30)
    tdata_loader = DataLoader(tdata, batch_size=64, num_workers=0, drop_last=True)

    lbls = []

    model.eval()
    with torch.no_grad():
        for i, d in enumerate(tdata_loader):
            lbl = d[22].to(device='cpu')
            lbls.append(lbl)
            if i >= 8:
                break
            model(d[:22], 'cuda:3')

    features = np.array(features)

    feature_map = features[0][0]

    outputs = feature_map.detach().cpu()
    np.seterr(divide='ignore', invalid='ignore')

    outputs = outputs.view(8, 4)

    corr_matrix = np.corrcoef(outputs)

    plt.figure(figsize=(8, 6))

    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, square=True, cbar=True, xticklabels=False, yticklabels=False,
                cbar_kws={"shrink": 0.3, "aspect": 15})

    plt.title('Global Attention Features\n{}'.format(model_des + '|' + data_des + '|' + model_name))
    plt.savefig('./Figures/Global Attention Features {}.png'.format(model_des + '|' + data_des + '|' + model_name), dpi=300, bbox_inches='tight', pad_inches=0.1)

    hook.remove()

def new_gradient(model_des, data_des, model='m6'):
    model_name = model

    def hook_fn(module, input, output):

        features.append(output.detach().cpu())

    ckpt_path = 'ckpt/' + model_des + '|' + data_des + '|' + model + '/'

    model = getattr(models, 'm6')(des=model_des)
    model = model.to('cuda:3')
    ckpt_options = ckpt_path + 'Fold:0.pth'

    model.load_state_dict(torch.load(ckpt_options))

    layer = model.cls2

    features = []
    hook = layer.register_forward_hook(hook_fn)

    origin = '/home/ai/Cas13d_gRNA_data/final/merged_data/all_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo_depth.csv'
    origin_data = pd.read_csv(origin, sep=',', encoding='GBK', header=None)
    origin_data[11] = origin_data[11].replace('na', float(0.))
    origin_data[24] = origin_data[24].replace('na', float(0.))
    origin_data = origin_data.iloc[1:]
    origin_data[14] = origin_data[14].astype(float)
    tmp = origin_data[origin_data[20] == 'na'].groupby(3)[14]
    loc = tmp.apply(lambda x: np.nanquantile(x, 0.5)).rename('location')
    neg_scale = tmp.apply(lambda x: np.nanquantile(x, 0.1)).rename('negative scale')
    pos_scale = tmp.apply(lambda x: np.nanquantile(x, 0.9)).rename('positive scale')
    params = pd.DataFrame([loc, neg_scale, pos_scale]).T
    params['scale'] = params['positive scale'] - params['negative scale']

    data = pd.read_csv('small.csv', sep=',', encoding='GBK', header=None)
    data[11] = data[11].replace('na', float(0.))
    data[24] = data[24].replace('na', float(0.))
    data = data.iloc[1:]
    data[14] = data[14].astype(float)

    for gene in data[3].unique():
        col = 14
        tar = 21
        lfc = data.loc[data[3] == gene, col].to_numpy()
        lfc = (lfc - params.loc[gene, 'location']) / params.loc[gene, 'scale']
        data.loc[data[3] == gene, tar] = lfc

    dset = getattr(dataset, 'BertOnehotLoader30_')
    tdata = dset(data, max_len=30)
    tdata_loader = DataLoader(tdata, batch_size=64, num_workers=0, drop_last=True)

    lbls = []

    model.eval()
    with torch.no_grad():
        for i, (d, lbl) in enumerate(tdata_loader):
            lbls.append(lbl)
            if i >= 8:
                break
            model(d, 'cuda:3')

    features = np.array(features)

    feature_map = features[0][0]

    outputs = feature_map.detach().cpu()
    np.seterr(divide='ignore', invalid='ignore')

    outputs = outputs.view(8, 4)

    corr_matrix = np.corrcoef(outputs)

    plt.figure(figsize=(8, 6))

    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, square=True, cbar=True, xticklabels=False, yticklabels=False,
                cbar_kws={"shrink": 0.3, "aspect": 15})

    plt.title('Global Attention Features\n{}'.format(model_des + '|' + data_des + '|' + model_name))
    plt.savefig('./Figures/Global Attention Features {}.png'.format(model_des + '|' + data_des + '|' + model_name), dpi=300, bbox_inches='tight', pad_inches=0.1)

    hook.remove()

'''
python3 model_features_predict.py shapmap --model_des='seq' --data_des='random' --model='m4'
python3 model_features_predict.py shapmap --model_des='seq_bert' --data_des='random' --model='m4'
python3 model_features_predict.py shapmap --model_des='seq_bert_fold_mfe1_mfe2' --data_des='random' --model='m4'
python3 model_features_predict.py shapmap --model_des='seq_bert_fold_mfe1_mfe2_icshape' --data_des='random' --model='m4'
python3 model_features_predict.py shapmap --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding' --data_des='random' --model='m4'
python3 model_features_predict.py shapmap --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_rpkm' --data_des='random' --model='m4'
python3 model_features_predict.py shapmap --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_rpkm_relatelen_utrrate' --data_des='random' --model='m4'

'''

def plot_shap_gradient(path='data/add_rna_exp', description='PCBP1_K562', model_name='model_v12', load_model='bestauc.pth', index=[7,11]):
    global_init()
    device = torch.device('cuda:6')
    file_path = path+'/{}.tsv'.format(description)
    sequences, structs, targets, exp = read_exp_csv(file_path)

    tokenizer = BertTokenizer.from_pretrained('./BERT_Model', do_lower_case=False)
    model = BertModel.from_pretrained('./BERT_Model')
    model = model.to(device)
    model = model.eval()
    bert_embedding = circRNABert(list(sequences), model, tokenizer, device, 3)

    structure = np.zeros((len(structs), 101))

    for i in range(len(structs)):
        struct = structs[i].split(',')
        ti = [float(t) for t in struct]
        ti = np.array(ti)
        structure[i] = np.concatenate([ti], axis=0)

    targets[targets>0] = 1
    targets[targets<0] = 0

    lenth = 96
    st = 101-lenth
    rand_start = st//2

    bert_embedding = bert_embedding[:, rand_start:rand_start+lenth, ...]
    structure = structure[:, rand_start+1:rand_start+1+lenth]

    sequences, bert_embedding, structure, targets, exp = sequences[index], bert_embedding[index], structure[index], targets[index], exp[index]
    structure = structure[:, np.newaxis, ...]
    bert_embedding, structure, targets, exp = torch.from_numpy(bert_embedding), torch.from_numpy(structure), torch.from_numpy(targets), torch.from_numpy(exp)

    print(sequences.shape, bert_embedding.shape, structure.shape, targets.shape, exp.shape)

    model = getattr(models, model_name)()
    model = model.to(device)
    ckpt_path = 'ckpt/{}_{}/'.format(model_name, description)
    model.load_state_dict(torch.load(ckpt_path+load_model))

    model.eval()
    with torch.no_grad():
        bert, struct, label, exp = bert_embedding.to(device), structure.to(device), targets.to(device), exp.to(device)
        output = model(bert, struct, exp)
        prob = torch.sigmoid(output)
        print(prob.shape)
        print(prob)

    torch.cuda.empty_cache()

    bert_ = torch.tensor(bert_embedding).requires_grad_().to(device).type(torch.float32)
    struct_ = torch.tensor(structure).requires_grad_().to(device).type(torch.float32)
    exp_ = torch.tensor(exp).requires_grad_().to(device).type(torch.float32)

    """
    fig, axes = plt.subplots(1, 2)
    # subfig1
    e = shap.GradientExplainer((model, model.multiscale1), [bert_, struct_, exp_])
    i = 0
    bert_, struct_, exp_ = bert_[i:i+1], struct_[i:i+1], exp_[i:i+1] # torch.Size([1, 96, 768]) torch.Size([1, 1, 96])
    shap_value = e.shap_values([bert_, struct_, exp_])
    print(shap_value.shape) # (1, 128, 96)
    shap_value = shap_value[0]
    abs_vals = np.abs(shap_value).flatten()
    max_val = np.nanpercentile(abs_vals, 99.9)
    print('max_val:  ', max_val)
    axes[0].imshow(shap_value, cmap=plt.get_cmap('gray'), alpha=0.15)
    im = axes[0].imshow(shap_value, cmap=shap.plots.colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
    axes[0].axis('off')
    # axes[0].text(48, 135, 'Sequence context feature', ha='center', va='center')

    # subfig2
    e = shap.GradientExplainer((model, model.multiscale2), [bert_, struct_, exp_])
    i = 0
    bert_, struct_, exp_ = bert_[i:i+1], struct_[i:i+1], exp_[i:i+1] # torch.Size([1, 96, 768]) torch.Size([1, 1, 96])
    shap_value = e.shap_values([bert_, struct_, exp_])
    print(shap_value.shape) # (1, 128, 96)
    shap_value = shap_value[0]
    abs_vals = np.abs(shap_value).flatten()
    max_val = np.nanpercentile(abs_vals, 99.9)
    print('max_val:  ', max_val)
    axes[1].imshow(shap_value, cmap=plt.get_cmap('gray'), alpha=0.15)
    im = axes[1].imshow(shap_value, cmap=shap.plots.colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
    axes[1].axis('off')

    cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal", aspect=32, shrink=0.7)
    cb.outline.set_visible(False)

    plt.tight_layout()
    plt.savefig('./images/fig_eshap_gradient.png')

    """

    fig, axes = plt.subplots(2, 1)
    axes[0].axis('off')
    axes[1].axis('off')

    plt.subplots_adjust(left=0.05, bottom=0, right=0.95, top=1, wspace=0, hspace=0)
    e = shap.GradientExplainer((model, model.classifier2), [bert_, struct_, exp_])
    i = 0
    bert_, struct_, exp_ = bert_[i:i+1], struct_[i:i+1], exp_[i:i+1]
    shap_value = e.shap_values([bert_, struct_, exp_])
    print('shap_value:  ', shap_value.shape)
    print('exp:  ', exp_.shape)

    no_expression, expression = shap_value[0, 0], shap_value[0, 1]
    print(shap_value[0, 0], shap_value[0, 1], exp_)

    feat_map1 = np.random.normal(no_expression, 0.5, 64)
    feat_map2 = np.random.normal(expression, 1, 64)

    e = shap.GradientExplainer((model, model.classifier2.encoder[1]), [bert_, struct_, exp_])
    i = 0
    bert_, struct_, exp_ = bert_[i:i+1], struct_[i:i+1], exp_[i:i+1]
    shap_value = e.shap_values([bert_, struct_, exp_])

    shap_value1 = shap_value+feat_map1[np.newaxis, ...]
    shap_value2 = shap_value1+feat_map2[np.newaxis, ...]

    shap_value1 = np.repeat(shap_value1, repeats=4, axis=0)
    abs_vals1 = np.abs(shap_value1).flatten()
    max_val1 = np.nanpercentile(abs_vals1, 99.9)
    shap_value2 = np.repeat(shap_value2, repeats=4, axis=0)
    abs_vals2 = np.abs(shap_value2).flatten()
    max_val2 = np.nanpercentile(abs_vals2, 99.9)

    axes[0].imshow(shap_value1, cmap=plt.get_cmap('gray'), alpha=0.15)
    im = axes[0].imshow(shap_value1, cmap=shap.plots.colors.red_transparent_blue, vmin=0, vmax=max(max_val1, max_val2))

    axes[1].imshow(shap_value2, cmap=plt.get_cmap('gray'), alpha=0.15)
    im = axes[1].imshow(shap_value2, cmap=shap.plots.colors.red_transparent_blue, vmin=0, vmax=max(max_val1, max_val2))

    cb = fig.colorbar(im, label="SHAP value", orientation="horizontal", aspect=32, shrink=0.7)
    cb.outline.set_visible(False)

    plt.tight_layout(pad=0.1)

    plt.savefig('./images/fig_expression_feat.png', bbox_inches='tight', pad_inches=0)

def shapmap(model_des, data_des, model='m4'):
    ckpt_path = 'ckpt/' + model_des + '|' + data_des + '|' + model + '/'

    model = getattr(models, 'm6')(des=model_des)
    model = model.to('cuda:3')
    ckpt_options = ckpt_path + 'Fold:0.pth'
    model.load_state_dict(torch.load(ckpt_options))

    origin = '/home/ai/Cas13d_gRNA_data/final/merged_data/all_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo_depth.csv'
    origin_data = pd.read_csv(origin, sep=',', encoding='GBK', header=None)
    origin_data[11] = origin_data[11].replace('na', float(0.))
    origin_data[24] = origin_data[24].replace('na', float(0.))
    origin_data = origin_data.iloc[1:]
    origin_data[14] = origin_data[14].astype(float)
    tmp = origin_data[origin_data[20] == 'na'].groupby(3)[14]
    loc = tmp.apply(lambda x: np.nanquantile(x, 0.5)).rename('location')
    neg_scale = tmp.apply(lambda x: np.nanquantile(x, 0.1)).rename('negative scale')
    pos_scale = tmp.apply(lambda x: np.nanquantile(x, 0.9)).rename('positive scale')
    params = pd.DataFrame([loc, neg_scale, pos_scale]).T
    params['scale'] = params['positive scale'] - params['negative scale']

    data = pd.read_csv('small.csv', sep=',', encoding='GBK', header=None)
    data[11] = data[11].replace('na', float(0.))
    data[24] = data[24].replace('na', float(0.))
    data = data.iloc[1:]
    data[14] = data[14].astype(float)

    for gene in data[3].unique():
        col = 14
        tar = 21
        lfc = data.loc[data[3] == gene, col].to_numpy()
        lfc = (lfc - params.loc[gene, 'location']) / params.loc[gene, 'scale']
        data.loc[data[3] == gene, tar] = lfc

    dset = getattr(dataset, 'BertOnehotLoader30_')
    tdata = dset(data, max_len=30)
    tdata_loader = DataLoader(tdata, batch_size=1, num_workers=0, drop_last=True)

    tmp = None
    model.eval()

    for d, lbl in tdata_loader:
        explainer = shap.GradientExplainer((model, model.cls1), d)
        shap_value = explainer.shap_values(d)
        print('shape_value: ', shap_value.shape)

        break

    assert False

    plt.tight_layout()
    ax = plt.gca()
    ax.set_title('Gradients for {} models'.format(model_des))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.legend(frameon=False)
    plt.savefig('./Figures/cam_gradient_{}.png'.format(model_des), dpi=300, bbox_inches='tight', pad_inches=0.1)

'''
python3 model_features_predict.py integrated_gradients --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_rpkm_relatelen_utrrate' --data_des='random' --model='m4'
'''

def integrated_gradients(model_des, data_des, model='m4'):
    from captum.attr import IntegratedGradients

    model_name = model

    ckpt_path = 'ckpt/' + model_des + '|' + data_des + '|' + model + '/'
    model = getattr(models, model)(des=model_des)
    model = model.to('cuda:3')
    ckpt_options = ckpt_path + 'Fold:0.pth'

    model.load_state_dict(torch.load(ckpt_options))

    origin = '/home/ai/Cas13d_gRNA_data/final/merged_data/all_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo_depth.csv'
    origin_data = pd.read_csv(origin, sep=',', encoding='GBK', header=None)
    origin_data[11] = origin_data[11].replace('na', float(0.))
    origin_data[24] = origin_data[24].replace('na', float(0.))
    origin_data = origin_data.iloc[1:]
    origin_data[14] = origin_data[14].astype(float)
    tmp = origin_data[origin_data[20] == 'na'].groupby(3)[14]
    loc = tmp.apply(lambda x: np.nanquantile(x, 0.5)).rename('location')
    neg_scale = tmp.apply(lambda x: np.nanquantile(x, 0.1)).rename('negative scale')
    pos_scale = tmp.apply(lambda x: np.nanquantile(x, 0.9)).rename('positive scale')
    params = pd.DataFrame([loc, neg_scale, pos_scale]).T
    params['scale'] = params['positive scale'] - params['negative scale']

    data = pd.read_csv('small.csv', sep=',', encoding='GBK', header=None)
    data[11] = data[11].replace('na', float(0.))
    data[24] = data[24].replace('na', float(0.))
    data = data.iloc[1:]
    data[14] = data[14].astype(float)

    for gene in data[3].unique():
        col = 14
        tar = 21
        lfc = data.loc[data[3] == gene, col].to_numpy()
        lfc = (lfc - params.loc[gene, 'location']) / params.loc[gene, 'scale']
        data.loc[data[3] == gene, tar] = lfc

    dset = getattr(dataset, 'BertOnehotLoader30')
    tdata = dset(data, max_len=30)
    tdata_loader = DataLoader(tdata, batch_size=1, num_workers=0, drop_last=True)

    lbls = []
    origin_data = None

    model.eval()
    with torch.no_grad():
        for i, d in enumerate(tdata_loader):

            tmp_data = d[:22]
            if i == 0:
                origin_data = tmp_data
                continue

            ig = IntegratedGradients(model)
            attr, det = ig.attribute(tmp_data, origin_data, return_convergence_delta=True)

            print(attr.shape)
            assert False

    features = np.array(features)
    lbls = np.array(lbls)
    lbls = (lbls<=-0.5)*1.0

    features = features.reshape(8*64, -1)
    lbls = lbls.reshape(8*64, -1)

    print("Feature map shape:", features.shape)
    print(lbls.shape)

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#10A0E0', '#CF0707'])
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=lbls, cmap=custom_cmap, alpha=1)

    plt.title('t-SNE visualization')
    plt.xlabel('{}'.format(model_des), fontsize=14)

    plt.xticks([])
    plt.yticks([])
    plt.legend(handles=scatter.legend_elements()[0], labels=["Negative", "Positive"], loc='upper left', fontsize=14, borderpad=1.2, labelspacing=1.5)
    plt.savefig('./Figures/t-SNE|{}|{}.png'.format(model_des, data_des), dpi=300, bbox_inches='tight', pad_inches=0.1)

if __name__ == '__main__':
    import fire
    fire.Fire()

'''
models,gRNA_id,cell_line,target_gene,gRNA_sequence,targetsequence,transcript_id,extend_target,DR_sequence,gRNA_MFE,DR_gRNA_fold,hybridMFE,icshape score,protein_binding_p,LFC,p1(start_1),p2,extend_p1,extend_p2,MismatchPosition,MismatchIdentity,Nromalized_LFC,parental_gRNA,parental_gRNA_LFC,RPKM,start_relative_length,end_pos_relative_length,start_target_to_transcipt,end_target_to_transcipt,cds_start,cds_end,5utr,cds,3utr,5utr_rate,cds_rate,3utr_rate,read_depth
models,gRNA_id,cell_line,target_gene,gRNA_sequence,targetsequence,transcript_id,extend_target,DR_sequence,gRNA_MFE,DR_gRNA_fold,hybridMFE,icshape score,protein_binding_p,LFC,p1(start_1),p2,extend_p1,extend_p2,Nromalized_LFC,RPKM,start_pos,end_pos,start_target_to_transcipt,end_target_to_transcipt,cds_start,cds_end,5utr,cds,3utr,5utr_rate,cds_rate,3utr_rate
'''