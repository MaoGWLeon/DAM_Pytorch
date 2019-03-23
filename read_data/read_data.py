import os
import time

import pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
import utils.evaluation as eva


class Config():
    def __init__(self):
        self.data_path = "../data/ubuntu/data.pkl"
        self.word_emb_init = "../data/ubuntu/word_embedding.pkl"
        self.save_path = "../output/ubuntu/temp/"
        self.init_model = None
        self.rand_seed = None

        self.drop_dense = None
        self.drop_attention = None

        self.is_mask = True
        self.is_layer_norm = True
        self.is_positional = False

        self.stack_num = 5
        self.attention_type = "dot"
        self.learning_rate = 1e-3
        self.vocab_size = 434512
        self.emb_size = 200

        self.batch_size = 100  # 模型默认是256 tensorflow的版本设置256跑不起来，先设100
        self.max_turn_num = 9
        self.max_turn_len = 50

        self.max_to_keep = 1
        self.num_scan_data = 2  # 相当于epoch
        self._EOS_ = 28270
        self.final_n_class = 1


def build_batches(data, conf):
    turns_batches = []
    response_batches = []
    label_batches = []

    tt_turns_len_batches = []
    every_turn_len_batches = []
    response_len_batches = []

    # data:('y','u','r')
    batch_len = len(data['y']) / conf.batch_size
    batch_len = int(batch_len)
    print(f'batch_len:{batch_len}')  # batch 的数量

    for batch_index in range(batch_len):
        turns, tt_turns_len, every_turn_len, response, response_len, label = \
            build_one_batch(data, batch_index, conf, turn_cut_type='tail', term_cut_type='tail')

        turns_batches.append(turns)
        response_batches.append(response)
        label_batches.append(label)

        tt_turns_len_batches.append(tt_turns_len)
        every_turn_len_batches.append(every_turn_len)
        response_len_batches.append(response_len)

    final_data={
        "turns":turns_batches,
        "response":response_batches,
        "label":label_batches,
        "tt_turns_len":tt_turns_len_batches,
        "every_turns_len":every_turn_len_batches,
        "response_len":response_len_batches
    }

    return final_data

def build_one_batch(data, batch_index, conf, turn_cut_type='tail', term_cut_type='tail'):
    turns = []
    response = []
    label = []

    tt_turns_len = []
    every_turn_len = []
    response_len = []

    for i in range(conf.batch_size):
        index = batch_index * conf.batch_size + i
        y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len = \
            produce_one_sample(data, index, conf._EOS_, conf.max_turn_num, conf.max_turn_len, turn_cut_type,
                               term_cut_type)

        turns.append(nor_turns_nor_c)  # [[[],[],...],[[],[],...],.....]
        response.append(nor_r)  # [[],[],.....]
        label.append(y)  # []
        every_turn_len.append(term_len)  # [[],[],.....]
        tt_turns_len.append(turn_len)  # []
        response_len.append(r_len)  # []

    return turns, tt_turns_len, every_turn_len, response, response_len, label


def produce_one_sample(data, index, split_id, max_turn_num, max_turn_len, turn_cut_type='tail', term_cut_type='tail'):
    '''
    nor是normalize的意思

    example:
    test_list = [1, 2, 3, 4, 5, 28270, 6, 7, 8, 9, 28270, 10, 11, 12, 13, 28270]
    test_list = split_c(test_list, 28270)
    print(f'{test_list}')
    nor_list, length = normalize_length(test_list, 10)
    print(f'{nor_list},{length}')
    nor_c_list=[]
    nor_c_len_list=[]
    for c in nor_list:
        nor_c,nor_c_len=normalize_length(c,10)
        print(f'{nor_c},{nor_c_len}')
        nor_c_list.append(nor_c)
        nor_c_len_list.append(nor_c_len)
    print(f'{nor_c_list},{nor_c_len_list}')
    '''
    c = data['c'][index]
    # r = data['r'][index][:]
    r = data['r'][index]
    y = data['y'][index]

    turns = split_c(c, split_id)  # 将每个context划分成一个个句子:[]-->[[],[],....]
    # turn_len返回normalize后的句子(utterance)个数
    nor_turns, turn_len = normalize_length(turns, max_turn_num, turn_cut_type)

    nor_turns_nor_c = []  # 对nor_turns里面每一个句子做normalize
    term_len = []  # term_len里面每个数值代表着一句话(utterance)的真实长度
    for c in nor_turns:
        nor_c, nor_c_len = normalize_length(c, max_turn_len, term_cut_type)
        nor_turns_nor_c.append(nor_c)
        term_len.append(nor_c_len)

    nor_r, r_len = normalize_length(r, max_turn_len, term_cut_type)

    return y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len


def normalize_length(turns, length, cut_type='tail'):
    '''
    :param turns: a list or nested list, example turns/r/single turn c
    :param length: the max length of a need list or nested list
    :param cut_type: head or tail, if _list len > length is used
    :return: a list len=length and min(read_length, length)
    '''
    real_length = len(turns)  # 这里的长度指的是context里面有多少句话(utterance的个数)，而不是每句话的长度是多少
    if real_length == 0:
        return [0] * length, 0

    if real_length <= length:
        if not isinstance(turns[0], list):
            turns.extend([0] * (length - real_length))
        else:
            print(f'{type(turns)}')
            turns.extend([[]] * (length - real_length))
        return turns, real_length

    if cut_type == 'head':
        return turns[:length], length
    if cut_type == 'tail':
        return turns[-length:], length


def split_c(c, split_id):
    '''
    这函数将[]-->[[],[],....]
    :param c: context
    :param split_id: _EOS_的index
    :return: turns
    '''
    turns = [[]]
    for id in c:
        if id != split_id:
            turns[-1].append(id)
        else:
            turns.append([])
    if turns[-1] == [] and len(turns) > 1:
        turns.pop()
    return turns


def read_data():
    config = Config()
    '''
        train_data:<class 'dict'>,dict_keys(['y', 'c', 'r'])
        val_data:<class 'dict'>,dict_keys(['y', 'c', 'r'])
        test_data:<class 'dict'>,dict_keys(['y', 'c', 'r'])
    '''

    # data 已经被token化
    '''
        data_small.pkl: train_data:10000    val_data:1000   test_data:1000
        data.pkl: train_data:1000000(pos:500000;neg:500000)    val_data:500000(10个当一组，50000个数据)   test_data:500000(10个当一组，50000个数据)
        data['c']是一个list,用_EOS_作为划分句子的标记，_EOS_的index为28270
    '''
    with open('../data/ubuntu/data_small.pkl', 'rb') as f:
        train_data, val_data, test_data = pickle.load(f)
    # for i in range(10):
    #     print(f"{val_data['y'][i]},{val_data['c'][i][:5]},{val_data['r'][i][:5]}")
    # for i in range(10):
    #     print(f"{test_data['y'][i]},{test_data['c'][i][:5]},{test_data['r'][i][:5]}")
    print(f"{train_data['c'][2][:]}")
    print(f"{train_data['c'][2]}")


if __name__ == '__main__':
    read_data()
