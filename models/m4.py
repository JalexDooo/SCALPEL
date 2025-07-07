import math
import copy

# import fm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



'''
ssh -p 36586 root@connect.beijinga.seetacloud.com

scp -rP 36586 all96_merge.h5 root@connect.beijinga.seetacloud.com:/root/autodl-tmp

'''

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Conv1d(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,),
                 dilation=(1,), if_bias=False, relu=True, same_padding=True, bn=True):
        super(Conv1d, self).__init__()
        p0 = int((kernel_size[0] - 1)/2) if same_padding else 0
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p0,
                              dilation=dilation, bias=True if if_bias else False)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        x = F.dropout(x, 0.3, training=self.training)
        return x

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0
        p1 = int((kernel_size[1] - 1) / 2) if same_padding else 0
        padding = (p0, p1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ResidualBlock2D(nn.Module):

    def __init__(self, planes, kernel_size=(11,5), padding=(5,2), downsample=True):
        super(ResidualBlock2D, self).__init__()
        self.c1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.b1 = nn.BatchNorm2d(planes)
        self.c2 = nn.Conv2d(planes, planes*2, kernel_size=kernel_size, stride=1,
                     padding=padding, bias=False)
        self.b2 = nn.BatchNorm2d(planes*2)
        self.c3 = nn.Conv2d(planes*2, planes*4, kernel_size=1, stride=1, bias=False)
        self.b3 = nn.BatchNorm2d(planes * 4)
        self.downsample = nn.Sequential(
            nn.Conv2d(planes, planes*4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes*4),
        )
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class ResidualBlock1D(nn.Module):

    def __init__(self, planes, downsample=True):
        super(ResidualBlock1D, self).__init__()
        kn = 32
        self.c1 = nn.Conv1d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.b1 = nn.BatchNorm1d(planes)
        self.c2 = nn.Conv1d(planes, planes*2, kernel_size=11, stride=1,
                     padding=5, bias=False)
        self.b2 = nn.BatchNorm1d(planes*2)
        self.c3 = nn.Conv1d(planes*2, planes*kn, kernel_size=1, stride=1, bias=False)
        self.b3 = nn.BatchNorm1d(planes * kn)
        self.downsample = nn.Sequential(
            nn.Conv1d(planes, planes*kn, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(planes*kn),
        )
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class multiscale(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(multiscale, self).__init__()

        self.conv0 = Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False)

        self.conv1 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False, bn=False),
            Conv1d(out_channel, out_channel, kernel_size=(3,), same_padding=True),
        )

        self.conv2 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False),
            Conv1d(out_channel, out_channel, kernel_size=(5,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(5,), same_padding=True),
        )

        self.conv3 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True)
        )

    def forward(self, x):

        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x4 = torch.cat([x0, x1, x2, x3], dim=1)
        return x4 + x

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class ChannelAttention1d(nn.Module):
    def __init__(self, channel, reduction=2):
        super(ChannelAttention1d, self).__init__()
        self.apool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.apool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return y

class ValueAttention1d(nn.Module):
    def __init__(self, deep, reduction=2):
        super(ValueAttention1d, self).__init__()
        self.apool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(deep, deep // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(deep // reduction, deep),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.transpose(-1, -2)
        b, d, c = x.size()
        y = self.apool(x).view(b, d)
        y = self.fc(y).view(b, 1, d)
        return y

class SEBlock1d(nn.Module):
    def __init__(self, deep, reduction=8):
        super(SEBlock1d, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(deep, deep//reduction),
            nn.ReLU(True),
            nn.Linear(deep//reduction, deep),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.fc(x)
        return x

class Encoder1(nn.Module):
    def __init__(self, deep, N, heads, dropout):
        super(Encoder1, self).__init__()
        self.N = N
        self.embed = FloatEmbedding(deep)
        self.pe = PositionalEncoder(deep, dropout=dropout)
        self.layers = get_clones(EncoderLayer(deep, heads, dropout), N)
        self.norm = Norm(deep)

    def forward(self, x, mask):
        x = self.embed(x)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class Encoder2(nn.Module):
    '''No embedding.
    
    '''
    def __init__(self, deep, N, heads, dropout):
        super(Encoder2, self).__init__()
        self.N = N
        self.embed = FloatEmbedding(deep)
        self.pe = PositionalEncoder(deep, dropout=dropout)
        self.layers = get_clones(EncoderLayer(deep, heads, dropout), N)
        self.norm = Norm(deep)

    def forward(self, x, mask):
        # x = self.embed(x)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

# utils class
def attention(q, k, v, deep_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(deep_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask==0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)
    
    output = torch.matmul(scores, v)
    return output

class PositionalEncoder(nn.Module):
    def __init__(self, deep, max_seq_len=2048, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        self.deep = deep
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, deep)
        for pos in range(max_seq_len):
            for i in range(0, deep, 2):
                pe[pos, i] = math.sin(pos/(10000 ** ((2*i)/deep) ))
                pe[pos, i+1] = math.cos(pos/(10000 ** ((2*(i+1))/deep) ))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x * math.sqrt(self.deep)
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe = pe.to(x.device)
        x = x + pe
        return self.dropout(x)

class Norm(nn.Module):
    def __init__(self, deep, eps=1e-6):
        super(Norm, self).__init__()
        self.deep = deep 

        self.alpha = nn.Parameter(torch.ones(self.deep))
        self.bias = nn.Parameter(torch.zeros(self.deep))

        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x-x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True)+self.eps) + self.bias
        return norm

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, deep, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.deep = deep
        self.deep_k = deep // heads
        self.head = heads

        self.q_linear = nn.Linear(deep, deep)
        self.v_linear = nn.Linear(deep, deep)
        self.k_linear = nn.Linear(deep, deep)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(deep, deep)

    def forward(self, q, k, v, mask=None):
        bz = q.size(0)

        k = self.k_linear(k).view(bz, -1, self.head, self.deep_k)
        q = self.q_linear(q).view(bz, -1, self.head, self.deep_k)
        v = self.v_linear(v).view(bz, -1, self.head, self.deep_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention(q, k, v, self.deep_k, mask, self.dropout)

        concat = scores.transpose(1, 2).contiguous().view(bz, -1, self.deep)
        output = self.out(concat)
        return output

class FeedForward(nn.Module):
    def __init__(self, deep, deep_feedforward=2048, dropout=0.1):
        super(FeedForward, self).__init__()

        self.linear_1 = nn.Linear(deep, deep_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(deep_feedforward, deep)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class FloatEmbedding(nn.Module):
    def __init__(self, deep):
        super(FloatEmbedding, self).__init__()
        self.linear_1 = nn.Linear(1, deep//2)
        self.linear_2 = nn.Linear(deep//2, deep)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x

# ---------------------Encoder--------------------------
class EncoderLayer(nn.Module):
    def __init__(self, deep, heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm_1 = Norm(deep)
        self.norm_2 = Norm(deep)
        self.attn = MultiHeadAttention(heads, deep, dropout=dropout)
        self.ff = FeedForward(deep, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, deep, heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm_1 = Norm(deep)
        self.norm_2 = Norm(deep)
        self.norm_3 = Norm(deep)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, deep, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, deep, dropout=dropout)
        self.ff = FeedForward(deep, dropout=dropout)

    def forward(self, x, encoder_output, in_mask, out_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(encoder_output, encoder_output, x2, out_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(encoder_output, encoder_output, x2, in_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

# --------------------Classifier------------------------
class DownBlock(nn.Module):
    '''
    Referring to HDRNet.
    '''
    def __init__(self, channel=1280, dropout=0.1):
        super(DownBlock, self).__init__()
        self.cpool = nn.ConstantPad1d((0, 1), 0)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(channel),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(channel, channel, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(channel),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.cpool(x)
        res = self.maxpool(x)
        x = self.conv(res)
        x = x + res
        return x


class AddFeatEncoder(nn.Module):
    def __init__(self, in_channel=1, out_channel=64):
        super(AddFeatEncoder, self).__init__()

    def forward(self, x):

        return x


class Classifier(nn.Module):
    def __init__(self, deep=128, in_channel=514, out_channel=1, feat_k=4, dropout=0.1, size=103):
        super(Classifier, self).__init__()

    def forward(self, x): # [64, 514, 28]
        print(x.shape)
        return x

class SCALPEL(nn.Module):
    def __init__(self, des='seq'):
        super(SCALPEL, self).__init__()
        # seq_fold_mfe1_2_icshape_binding_rpkm_relate_len_utr__rate_read_depth
        self.fold, self.mfe1, self.mfe2, self.icshape, self.binding, \
        self.rpkm, self.rt1, self.rt2, \
        self.utr1, self.utr2, self.utr3, self.read_depth, self.bert  = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        self.feat_len = 1920

        if 'bert' in des:
            self.bert = 1
            self.feat_len += 1792
        if 'fold' in des:
            self.fold = 1
            self.feat_len += 180
        if 'mfe1' in des:
            self.mfe1 = 1
            self.feat_len += 1
        if 'mfe2' in des:
            self.mfe2 = 1
            self.feat_len += 1
        if 'icshape' in des:
            self.icshape = 1
            self.feat_len += 30
        if 'binding' in des:
            self.binding = 1
            self.feat_len += 172
        if 'rpkm' in des:
            self.rpkm = 1
            self.feat_len += 1
        if 'relatelen' in des:
            self.rt1 = 1
            self.rt2 = 1
            self.feat_len += 2
        if 'utrrate' in des:
            self.utr1, self.utr2, self.utr3 = 1, 1, 1
            self.feat_len += 3
        # if 'read_depth' in des:
        #     self.read_depth = 1
        
        self.seq1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=1, padding='same'),
            nn.ReLU()
        )

        self.fold1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding='same'),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=4, stride=1, padding='same'),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding='same'),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.flat = nn.Flatten()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)

        self.cls1 = nn.Sequential(
            nn.Linear(self.feat_len, 128),
            # nn.Sigmoid(),
            nn.Dropout(0.1)
        )
        self.cls2 = nn.Sequential(
            nn.Linear(128, 32),
            # nn.Sigmoid(),
            nn.Dropout(0.1)
        )
        self.out = nn.Sequential(
            nn.Linear(32, 1)
            # nn.Sigmoid()
        )

        self.bert1 = Conv1d(768, 64, kernel_size=(1,), stride=1)
        self.bert2 = multiscale(64, 16)

        self.seq_se = SEBlock1d(1920, 32)
        self.glob_att = SEBlock1d(self.feat_len, 32)
        

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, device='cuda:0'):
        g_seq, t_seq, extt_seq, dr_seq, g_fold, \
        g_mfe, h_mfe, icshape, binding_p, \
        norm_label, p1, p2, \
        rpkm, \
        st_relate_len, ed_relate_len, \
        utr5_rate, cds_rate, utr3_rate, read_depth, \
        cell_line, target_gene, bert_embedding = x

        bert_embedding = bert_embedding.to(device) # torch.Size([64, 28, 768])
        g_seq = g_seq.to(device) # torch.Size([64, 30, 4])
        t_seq = t_seq.to(device) # torch.Size([64, 30, 4])
        g_fold = g_fold.to(device)
        g_mfe = g_mfe.to(device)
        h_mfe = h_mfe.to(device)
        icshape = icshape.to(device)
        binding_p = binding_p.to(device)
        rpkm = rpkm.to(device)
        st_relate_len = st_relate_len.to(device)
        ed_relate_len = ed_relate_len.to(device)
        utr5_rate = utr5_rate.to(device)
        cds_rate = cds_rate.to(device)
        utr3_rate = utr3_rate.to(device)

        g_seq = g_seq.unsqueeze(-1).permute(0, 3, 1, 2)
        t_seq = t_seq.unsqueeze(-1).permute(0, 3, 1, 2)
        g_fold = g_fold.unsqueeze(-1).permute(0, 3, 1, 2)

        seq = torch.cat([g_seq, t_seq], dim=1)

        
        seq1 = self.seq1(seq)
        seq1 = self.conv1(seq1)
        seq1 = self.maxpool(seq1)
        seq_flat = self.flat(seq1) # [N, 1920]
        seq1 = self.dropout1(seq_flat)

        # seblock + residual
        seq_se = self.seq_se(seq1)
        seq_se_res = seq_se * seq1 + seq1
        features = seq_se_res

        # features = seq1

        if self.bert:
            bert_embedding = bert_embedding.permute(0, 2, 1) # torch.Size([64, 768, 28])
            bert1 = self.bert1(bert_embedding)
            bert2 = self.bert2(bert1) # torch.Size([64, 64, 28]) 1792
            bert_feat = self.flat(bert2)
            bert_feat = self.dropout3(bert_feat)
            features = torch.concat([features, bert_feat], dim=1)

        if self.fold:
            fold1 = self.fold1(g_fold)
            fold1 = self.conv2(fold1)
            fold1 = self.maxpool(fold1)
            fold1 = self.flat(fold1) # [N, 180]
            fold1d = self.dropout2(fold1)
            features = torch.concat([features, fold1d], dim=1)
        
        if self.mfe1:
            features = torch.concat([features, g_mfe], dim=1)
        if self.mfe2:
            features = torch.concat([features, h_mfe], dim=1)
        if self.icshape:
            features = torch.concat([features, icshape], dim=1)
        if self.binding:
            features = torch.concat([features, binding_p], dim=1)
        if self.rpkm:
            features = torch.concat([features, rpkm], dim=1)
        if self.rt1 | self.rt2:
            features = torch.concat([features, st_relate_len], dim=1)
            features = torch.concat([features, ed_relate_len], dim=1)
        if self.utr1 | self.utr2 | self.utr3:
            features = torch.concat([features, utr5_rate], dim=1)
            features = torch.concat([features, cds_rate], dim=1)
            features = torch.concat([features, utr3_rate], dim=1)
        
        glob_att = self.glob_att(features)
        glob_att_res = features * glob_att + features
        features = glob_att_res

        cls1 = self.cls1(features)
        cls2 = self.cls2(cls1)
        output = self.out(cls2)

        return output


