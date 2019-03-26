import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init

import PyTorch_models.transformer_util as util

'''
    论文的输入：
    turns:(batch_size,9,50)
    every_turn_len:(batch_size,9)   这里是句子的真实长度，为了做mask
    response:(batch_size,50)
    response_len:(batch_size)   这里是句子的真实长度，为了做mask
    label:(batch_size)
'''


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
            print(f'mask is not None')
            attn = attn.masked_fill(mask, -2 ** 32 + 1)  # 这里用论文的数值

        attn = self.softmax(attn)

        # 需要指定是否需要dropout    默认dropout=0.1
        if dropout is not None:
            attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        return output, attn


class OneHeadAttention(nn.Module):
    # self attention + add residual + layer norm
    def __init__(self, d_model, d_q, dropout=0.1):
        super(OneHeadAttention, self).__init__()
        self.d_q = d_q

        self.attention = ScaledDotProductAttention(sqart=np.power(self.d_q, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, use_dropout=False):
        '''
        Q: a tensor with shape [batch, Q_time, Q_dimension]
        K: a tensor with shape [batch, time, K_dimension]
        V: a tensor with shape [batch, time, V_dimension]

        #暂时不需要这两个参数
        Q_lengths: a tensor with shape [batch]
        K_lengths: a tensor with shape [batch]

        mask:在函数get_attn_key_pad_mask里面计算好了 目前需要改动它
        '''
        residual = q

        # 这里不用repeat（重复）mask了，因为只有one head
        # mask = mask.repeat(self.n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        if use_dropout:
            print(f"OneHead-Attention use dropout")
            output = self.dropout(output)
        # add residual + layer norm
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        # self.w_2 = nn.Conv1d(d_hid, d_in, 1)

        self.linear1 = nn.Linear(d_in, d_hid)
        init.orthogonal_(self.linear1.weight)
        init.zeros_(self.linear1.bias)

        self.linear2 = nn.Linear(d_hid, d_in)
        init.orthogonal_(self.linear2.weight)
        init.zeros_(self.linear2.bias)

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_dropout=False):
        residual = x
        output = F.relu(self.linear1(x))
        output = self.linear2(output)

        if use_dropout:
            output = self.dropout(output)

        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    # 单层Encoder
    # 由两个模块组成 1、Multi-Head Attention
    #             2、Position-wise FeedForward Network
    '''
        d_model:相当于输入的词向量维度
        d_hidden:FFN层里面的隐藏层维度
        d_q:q,k,v中q的维度，用来在dot-product attention 中做scale操作

    '''

    def __init__(self, d_model, d_hidden, d_q, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = OneHeadAttention(d_model, d_q, dropout=dropout)
        self.pos_wise_ffn = PositionwiseFeedForward(d_model, d_hidden, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        # 这里不用non_pad_mask,做了attention后连ffn
        # enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        # enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class OneStep(nn.Module):
    '''
        n_src_vocab:

    '''

    def __init__(self, n_src_vocab, len_max_seq, d_word_vec,
                 n_layers, d_q, d_model, d_hidden, dropout=0.1):
        super(OneStep, self).__init__()
        self.len_max_seq = len_max_seq  # 用来计算mask
        self.src_word_emb = nn.Embedding(n_src_vocab + 1, d_word_vec, padding_idx=util.PAD)  # util.PAD=0
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_hidden, d_q, dropout) for _ in range(n_layers)])

    def forward(self, src_seq, q_lengths, k_lengths, return_attns=False):
        enc_slf_attn_list = []

        slf_attn_mask = util.DAM_get_attn_key_pad_mask(q_lengths, k_lengths, self.len_max_seq)

        enc_output = self.src_word_emb(src_seq)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output
