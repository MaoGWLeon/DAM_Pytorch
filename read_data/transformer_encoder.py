import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


class ScaledDotProductAttention(nn.Module):
    '''
        Scaled Dot-Product Attention
    '''

    def __init__(self, sqart, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.sqart = sqart
        self.dropout = nn.Dropout(attn_dropout)
        # input:(batch_size,length,word_embedding)
        self.softmax = nn.Softmax(dim=2)

    '''
        K.T:K的转置
        softmax(Q*K.T/sqart(d))*V
    '''

    def forward(self, q, k, v, mask=None, dropout=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))
        attn = attn / self.sqart

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)

        # 需要指定是否需要dropout    默认dropout=0.1
        if dropout is not None:
            attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    # self attention + add residual + layer norm
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # 这里q 和 k 转化成的维度要相同 不然做不了点积
        # 百度的DAM并没有用到multi-head 是否需要对输入的wordembedding做这个转化要看看那边的代码
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        # 正态分布初始化,mean和std自行确定,看后续是否要改动
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(sqart=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, len_q, _ = q.size()
        batch_size, len_k, _ = k.size()
        batch_size, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(batch_size, len_q, self.n_head, self.d_k)
        k = self.w_ks(k).view(batch_size, len_k, self.n_head, self.d_k)
        v = self.w_vs(v).view(batch_size, len_v, self.n_head, self.d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.d_v)  # (n*b) x lv x dv

        mask = mask.repeat(self.n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        # (n*b) x lq x dv-->n x b x lq x dv
        output = output.view(self.n_head, batch_size, len_q, self.d_v)
        # (n,b,lq,dv)-->(b,lq,n,dv)-->(b,lq,n*dv)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        # add residual + layer norm
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.permute(0, 2, 1)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.permute(0, 2, 1)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    # 单层Encoder
    # 由两个模块组成 1、Multi-Head Attention
    #             2、Position-wise FeedForward Network
    def __init__(self, d_model, d_hidden, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_wise_ffn = PositionwiseFeedForward(d_model, d_hidden, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
