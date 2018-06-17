# -*- coding:utf-8 -*-
"""
.. module:: utils
    :synopsis: utility tools
 

"""

import codecs
import csv
import itertools
from functools import reduce

import numpy as np
import shutil
import torch
import json

import torch.nn as nn
import torch.nn.init

from model.ner_dataset import *

zip = getattr(itertools, 'izip', zip)


def variable(tensor, gpu):
    if gpu >= 0 :
        return torch.autograd.Variable(tensor).cuda()
    else:
        return torch.autograd.Variable(tensor)

def to_scalar(var):
    """change the first element of a tensor to scalar
    """
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    """helper function to calculate argmax of input vector at dimension 1
    """
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum

    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
      
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M


def switch(vec1, vec2, mask):
    """
    switch function for pytorch

    args:
        vec1 (any size) : input tensor corresponding to 0
        vec2 (same to vec1) : input tensor corresponding to 1
        mask (same to vec1) : input tensor, each element equals to 0/1
    return:
        vec (*)
    """
    catvec = torch.cat([vec1.view(-1, 1), vec2.view(-1, 1)], dim=1)
    switched_vec = torch.gather(catvec, 1, mask.long().view(-1, 1))
    return switched_vec.view(-1)


def encode2char_safe(input_lines, char_dict):
    """
    get char representation of lines

    args:
        input_lines (list of strings) : input corpus
        char_dict (dictionary) : char-level dictionary
    return:
        forw_lines
    """
    unk = char_dict['<u>']
    forw_lines = [list(map(lambda m: list(map(lambda t: char_dict.get(t, unk), m)), line)) for line in input_lines]
    return forw_lines


def concatChar(input_lines, char_dict):
    """
    concat char into string

    args:
        input_lines (list of list of char) : input corpus
        char_dict (dictionary) : char-level dictionary
    return:
        forw_lines
    """
    features = [[char_dict[' ']] + list(reduce(lambda x, y: x + [char_dict[' ']] + y, sentence)) + [char_dict['\n']] for sentence in input_lines]
    return features


def encode_safe(input_lines, word_dict, unk):
    """
    encode list of strings into word-level representation with unk
    """
    lines = list(map(lambda t: list(map(lambda m: word_dict.get(m, unk), t)), input_lines))
    return lines


def encode(input_lines, word_dict):
    """
    encode list of strings into word-level representation
    """
    lines = list(map(lambda t: list(map(lambda m: word_dict[m], t)), input_lines))
    return lines


def shrink_features(feature_map, features, thresholds):
    """
    filter un-common features by threshold
    """
    feature_count = {k: 0 for (k, v) in iter(feature_map.items())}
    for feature_list in features:
        for feature in feature_list:
            feature_count[feature] += 1
    shrinked_feature_count = [k for (k, v) in iter(feature_count.items()) if v >= thresholds]
    feature_map = {shrinked_feature_count[ind]: (ind + 1) for ind in range(0, len(shrinked_feature_count))}

    #inserting unk to be 0 encoded
    feature_map['<unk>'] = 0
    #inserting eof
    feature_map['<eof>'] = len(feature_map)
    return feature_map


def generate_corpus(lines, if_shrink_feature=False, thresholds=1):
    """
    generate label, feature, word dictionary and label dictionary

    args:
        lines : corpus
        if_shrink_feature: whether shrink word-dictionary
        threshold: threshold for shrinking word-dictionary
        
    """
    features = list()
    labels = list()
    tmp_fl = list()
    tmp_ll = list()
    feature_map = dict()
    label_map = dict()
    bichar_feature_map = dict()
    bichar_features = list()
    tmp_fb = list()
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            tmp_fl.append(line[0])
            if len(tmp_fl) > 1:
                tmp_fb.append(tmp_fl[-2] + '' + tmp_fl[-1])
                if tmp_fb[-1] not in bichar_feature_map:
                    bichar_feature_map[tmp_fb[-1]] = len(bichar_feature_map) + 1  # 0 is for unk
            if line[0] not in feature_map:
                feature_map[line[0]] = len(feature_map) + 1 #0 is for unk

            tmp_ll.append(line[-1])
            if line[-1] not in label_map:
                label_map[line[-1]] = len(label_map)
        elif len(tmp_fl) > 0:
            tmp_fb.append(tmp_fl[-1] + '-null-')
            bichar_feature_map[tmp_fb[-1]] = len(bichar_feature_map) + 1
            bichar_features.append(tmp_fb)
            features.append(tmp_fl)
            labels.append(tmp_ll)
            tmp_fl = list()
            tmp_ll = list()
            tmp_fb = list()
    if len(tmp_fl) > 0:
        tmp_fb.append(tmp_fl + '-null-')
        bichar_feature_map[tmp_fb[-1]] = len(bichar_feature_map) + 1
        bichar_features.append(tmp_fb)
        features.append(tmp_fl)
        labels.append(tmp_ll)
    label_map['<start>'] = len(label_map)
    label_map['<pad>'] = len(label_map)
    if if_shrink_feature:
        feature_map = shrink_features(feature_map, features, thresholds)
        bichar_feature_map = shrink_features(bichar_feature_map, bichar_features, thresholds)
    else:
        #inserting unk to be 0 encoded
        feature_map['<unk>'] = 0
        #inserting eof
        feature_map['<eof>'] = len(feature_map)

        # inserting unk to be 0 encoded
        bichar_feature_map['<unk>'] = 0
        # inserting eof
        bichar_feature_map['<eof>'] = len(bichar_feature_map)

    return features, labels, feature_map, label_map, bichar_features, bichar_feature_map


def read_corpus(lines):
    """
    convert corpus into features and labels
    """
    features = list()
    labels = list()
    tmp_fl = list()
    tmp_ll = list()
    bichar_features = list()
    tmp_fb = list()
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            if len(line) > 2:
                word = line[0]
                for i in range(1, len(line)-1):
                    word += '_' + line[i]
                tmp_fl.append(word)
            else:
                tmp_fl.append(line[0])
                if len(tmp_fl) > 1:
                    tmp_fb.append(tmp_fl[-2]+''+tmp_fl[-1])
            tmp_ll.append(line[-1])
        elif len(tmp_fl) > 0:
            tmp_fb.append(tmp_fl[-1]+'-null-')
            features.append(tmp_fl)
            labels.append(tmp_ll)
            bichar_features.append(tmp_fb)
            tmp_fl = list()
            tmp_ll = list()
            tmp_fb = list()
    if len(tmp_fl) > 0:
        tmp_fb.append(tmp_fl + '-null-')
        bichar_features.append(tmp_fb)
        features.append(tmp_fl)
        labels.append(tmp_ll)

    return features, labels, bichar_features


def shrink_embedding(feature_map, word_dict, word_embedding, caseless):
    """
    shrink embedding dictionary to in-doc words only
    """
    if caseless:
        feature_map = set([k.lower() for k in feature_map.keys()])
    new_word_list = [k for k in word_dict.keys() if (k in feature_map)]
    new_word_dict = {k:v for (v, k) in enumerate(new_word_list)}
    new_word_list_ind = torch.LongTensor([word_dict[k] for k in new_word_list])
    new_embedding = word_embedding[new_word_list_ind]
    return new_word_dict, new_embedding


def load_embedding_wlm(emb_file, delimiter, feature_map, full_feature_set, caseless, unk, emb_len, shrink_to_train=False, shrink_to_corpus=False):
    """
    load embedding, indoc words would be listed before outdoc words

    args: 
        emb_file: path to embedding file
        delimiter: delimiter of lines
        feature_map: word dictionary
        full_feature_set: all words in the corpus
        caseless: convert into casesless style
        unk: string for unknown token
        emb_len: dimension of embedding vectors
        shrink_to_train: whether to shrink out-of-training set or not
        shrink_to_corpus: whether to shrink out-of-corpus or not
    """
    if caseless:
        feature_set = set([key.lower() for key in feature_map])
        full_feature_set = set([key.lower() for key in full_feature_set])
    else:
        feature_set = set([key for key in feature_map])
        full_feature_set = set([key for key in full_feature_set])
    
    #ensure <unk> is 0
    word_dict = {v:(k+1) for (k,v) in enumerate(feature_set - set(['<unk>']))}
    word_dict['<unk>'] = 0

    in_doc_freq_num = len(word_dict)
    rand_embedding_tensor = torch.FloatTensor(in_doc_freq_num, emb_len)
    init_embedding(rand_embedding_tensor)

    indoc_embedding_array = list()
    indoc_word_array = list()
    outdoc_embedding_array = list()
    outdoc_word_array = list()

    for line in open(emb_file, 'r'):
        line = line.split(delimiter)
        if len(line) > 2:
            vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
            if len(vector) == emb_len:
                if shrink_to_train and line[0] not in feature_set:
                    continue

                if line[0] == unk:
                    rand_embedding_tensor[0] = torch.FloatTensor(vector) #unk is 0
                elif line[0] in word_dict:
                    rand_embedding_tensor[word_dict[line[0]]] = torch.FloatTensor(vector)
                elif line[0] in full_feature_set:
                    indoc_embedding_array.append(vector)
                    indoc_word_array.append(line[0])
                elif not shrink_to_corpus:
                    outdoc_word_array.append(line[0])
                    outdoc_embedding_array.append(vector)
    if len(indoc_embedding_array) > 0:
        embedding_tensor_0 = torch.FloatTensor(np.asarray(indoc_embedding_array))

    if not shrink_to_corpus: #shrink_to_corpus=False
        embedding_tensor_1 = torch.FloatTensor(np.asarray(outdoc_embedding_array))

    if shrink_to_corpus:
        if len(indoc_embedding_array) > 0:
            embedding_tensor = torch.cat([rand_embedding_tensor, embedding_tensor_0], 0)
        else:
            embedding_tensor = rand_embedding_tensor
    else:
        if len(indoc_embedding_array) > 0:
            embedding_tensor = torch.cat([rand_embedding_tensor, embedding_tensor_0, embedding_tensor_1], 0)
        else:
            embedding_tensor = torch.cat([rand_embedding_tensor, embedding_tensor_1], 0)

    for word in indoc_word_array:
        word_dict[word] = len(word_dict)
    in_doc_num = len(word_dict)
    if  not shrink_to_corpus:
        for word in outdoc_word_array:
            word_dict[word] = len(word_dict)

    return word_dict, embedding_tensor, in_doc_num


def construct_bucket_mean_vb(input_features, input_label, lexicons, word_dict, label_dict, bichar_input_features, bichar_word_dict, caseless):
    """
    Construct bucket by mean for viterbi decode, word-level only
    """
    # encode and padding
    if caseless:
        input_features = list(map(lambda t: list(map(lambda x: x.lower(), t)), input_features))
        baichar_input_features = list(map(lambda t: list(map(lambda x: x.lower(), t)), bichar_input_features))

    features = encode_safe(input_features, word_dict, word_dict['<unk>'])
    bichar_features = encode_safe(bichar_input_features, bichar_word_dict, bichar_word_dict['<unk>'])
    labels = encode(input_label, label_dict)
    labels = list(map(lambda t: [label_dict['<start>']] + list(t), labels))
    # thresholds = calc_threshold_mean(features)

    return construct_bucket_vb(features, labels, lexicons, word_dict['<eof>'], label_dict['<pad>'], len(label_dict),bichar_features)


def padding_lexicon_bucket(lexicon_features, lexicon_f_map, gpu):
    for lexicons in lexicon_features:
        if lexicons.max_lex_num == 0:
            lexicons.max_lex_num = 1
        lexicons.lexicons_index_end = encode_safe(lexicons.lexicons_end, lexicon_f_map, lexicon_f_map['<unk>'])
        lexicons.lexicons_index_start = encode_safe(lexicons.lexicons_start, lexicon_f_map, lexicon_f_map['<unk>'])
        for i in range(len(lexicons.lexicons_end)):
            if len(lexicons.lexicons_end[i]) < lexicons.max_lex_num:
                lexicons.lexicons_index_end[i] += [lexicon_f_map['<eof>']] * (lexicons.max_lex_num - len(lexicons.lexicons_end[i]))
            if len(lexicons.lexicons_start[i]) < lexicons.max_lex_num:
                lexicons.lexicons_index_start[i] += [lexicon_f_map['<eof>']] * (lexicons.max_lex_num - len(lexicons.lexicons_start[i]))
        lexicons.lexicons_features_end = variable(torch.LongTensor(np.asarray(lexicons.lexicons_index_end)), gpu)
        lexicons.lexicons_features_start = variable(torch.LongTensor(np.asarray(lexicons.lexicons_index_start)), gpu)
    return lexicon_features


def construct_bucket_vb(input_features, input_labels, input_lexicons, pad_feature, pad_label, label_size, bichar_input_features):
    """
    Construct bucket by thresholds for viterbi decode, word-level only
    """
    buckets = [[], [], [], [], []]
    for feature, label, lexicon, bichar_feature in zip(input_features, input_labels, input_lexicons, bichar_input_features):
        cur_len = len(feature)
        cur_len_1 = cur_len + 1
        buckets[0].append(feature + [pad_feature])
        buckets[1].append([label[ind] * label_size + label[ind + 1] for ind in range(0, cur_len)] + [
            label[cur_len] * label_size + pad_label])
        buckets[2].append([1] * cur_len_1)
        buckets[3].append(lexicon)
        buckets[4].append(bichar_feature + [pad_feature])

    return buckets


def find_length_from_labels(labels, label_to_ix):
    """
    find length of unpadded features based on labels
    """
    end_position = len(labels) - 1
    for position, label in enumerate(labels):
        if label == label_to_ix['<pad>']:
            end_position = position
            break
    return end_position


def revlut(lut):
    return {v: k for k, v in lut.items()}


def save_checkpoint(state, track_list, filename):
    """
    save checkpoint
    """
    with open(filename+'.json', 'w') as f:
        json.dump(track_list, f)
    torch.save(state, filename+'.model')

def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform(input_embedding, -bias, bias)

def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
    
    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def read_corpus_lexicon(lines, sentences, lexicon_feature_map):
    lexicons = []
    end_pos = []
    start_pos = []
    lexicons_len = []
    each = []
    for line in lines:
        line = line.rstrip()
        if len(line) > 0 and line != '<emp-dict>':
            sub_sequence = line.split(' ')
            each.append(sub_sequence[0])
            # print(line)
            start_pos.append(int(sub_sequence[1]))
            end_pos.append(int(sub_sequence[2]))
            lexicons_len.append(len(sub_sequence[0]))
            if sub_sequence[0] not in lexicon_feature_map:
                lexicon_feature_map[sub_sequence[0]] = len(lexicon_feature_map) + 1#0 is for unk
        elif len(line) == 0:
            # if each[0] == '':
            #     lexicons.append([])
            # else:
            #     lexicons.append(each)
            lexicon = Lexicon()
            lexicon.generate_lexi(len(sentences[len(lexicons)]), each, start_pos, end_pos, lexicons_len)
            lexicons.append(lexicon)
            each = []
            end_pos = []
            start_pos = []
            lexicons_len = []

    return lexicons, lexicon_feature_map


def generous_corpus_lexicon(lines, sentences, lexicon_feature_map):
    lexicons = []
    end_pos = []
    start_pos = []
    lexicons_len = []
    each = []
    for line in lines:
        line = line.rstrip()
        if len(line) > 0 and line != '<emp-dict>':
            sub_sequence = line.split(' ')
            each.append(sub_sequence[0])
            start_pos.append(int(sub_sequence[1]))
            end_pos.append(int(sub_sequence[2]))
            lexicons_len.append(len(sub_sequence[0]))
            if sub_sequence[0] not in lexicon_feature_map:
                lexicon_feature_map[sub_sequence[0]] = len(lexicon_feature_map) + 1#0 is for unk
        elif len(line) == 0:
            # if each[0] == '':
            #     lexicons.append([])
            # else:
            #     lexicons.append(each)
            lexicon = Lexicon()
            lexicon.generate_lexi(len(sentences[len(lexicons)]), each, start_pos, end_pos, lexicons_len)
            lexicons.append(lexicon)
            each = []
            end_pos = []
            start_pos = []
            lexicons_len = []

    lexicon_feature_map['<unk>'] = 0
    lexicon_feature_map['<eof>'] = len(lexicon_feature_map)
    return lexicons, lexicon_feature_map


def find_matching_tag(previous, current):
    # inllegal_bies: current_previous
    illegal_bies = ['B_B', 'B_I', 'S_B', 'S_I', 'I_E', 'I_S', 'E_S', 'E_E']
    bies = current + '_' + previous
    if bies in illegal_bies:
        return False
    else:
        return True


def create_matching_labels_matrix(r_l_map):
    T = len(r_l_map) - 2
    matching_tag = np.zeros([T, T])  # 0:illegal, 1:legal
    for t in range(0, T):
        tstr = r_l_map[t]
        ttags = tstr.split('_')

        for tL1 in range(0, T):
            tL1str = r_l_map[tL1]
            tL1tags = tL1str.split('_')
            if len(ttags) != len(tL1tags):
                if len(tL1tags) == 3 \
                        and (tL1tags[1] == "E" or tL1tags[1] == "S") \
                        and (tL1tags[2] == "E" or tL1tags[2] == "S"):
                    if tL1tags[0] == "I" or tL1tags[0] == "B":
                        if len(ttags) == 2 \
                                and find_matching_tag(tL1tags[0], ttags[0]) \
                                and find_matching_tag(tL1tags[1], ttags[1]) \
                                and find_matching_tag(tL1tags[2], ttags[1]):
                            matching_tag[tL1][t] = 1
                    elif tL1tags[0] == "E" or tL1tags[0] == "S":
                        if len(ttags) == 2 \
                                and find_matching_tag(tL1tags[0], ttags[0]) \
                                and find_matching_tag(tL1tags[1], ttags[1]) \
                                and find_matching_tag(tL1tags[2], ttags[1]):
                            matching_tag[tL1][t] = 1
                        elif len(ttags) == 1 \
                                and find_matching_tag(tL1tags[0], ttags[0]) \
                                and find_matching_tag(tL1tags[1], ttags[0]) \
                                and find_matching_tag(tL1tags[2], ttags[0]):
                            matching_tag[tL1][t] = 1
                elif len(tL1tags) == 2 and (tL1tags[1] == "E" or tL1tags[1] == "S"):
                    if tL1tags[0] == "I" or tL1tags[0] == "B":
                        if len(ttags) == 3 \
                                and find_matching_tag(tL1tags[0], ttags[0]) \
                                and find_matching_tag(tL1tags[1], ttags[1]) \
                                and find_matching_tag(tL1tags[1], ttags[2]):
                            matching_tag[tL1][t] = 1
                    elif tL1tags[0] == "E" or tL1tags[0] == "S":
                        if len(ttags) == 3 \
                                and find_matching_tag(tL1tags[0], ttags[0]) \
                                and find_matching_tag(tL1tags[1], ttags[1]) \
                                and find_matching_tag(tL1tags[1], ttags[2]):
                            matching_tag[tL1][t] = 1
                        elif len(ttags) == 1 \
                                and find_matching_tag(tL1tags[0], ttags[0]) \
                                and find_matching_tag(tL1tags[1], ttags[0]) \
                                and find_matching_tag(tL1tags[1], ttags[0]):
                            matching_tag[tL1][t] = 1
                elif len(tL1tags) == 1 and (tL1tags[0] == "E" or tL1tags[0] == "S"):
                    if len(ttags) == 3 \
                            and find_matching_tag(tL1tags[0], ttags[0]) \
                            and find_matching_tag(tL1tags[0], ttags[1]) \
                            and find_matching_tag(tL1tags[0], ttags[2]):
                        matching_tag[tL1][t] = 1
                    elif len(ttags) == 2 \
                            and find_matching_tag(tL1tags[0], ttags[0]) \
                            and find_matching_tag(tL1tags[0], ttags[1]):
                        matching_tag[tL1][t] = 1
            else:
                sz = len(ttags)
                k = 0
                while k < sz:
                    if len(tL1tags) <= k:
                        previous = tL1tags[0]
                    else:
                        previous = tL1tags[k]
                    if len(ttags) <= k:
                        current = ttags[0]
                    else:
                        current = ttags[k]
                    if not find_matching_tag(previous, current):
                        break
                    k += 1
                if k == sz:
                    matching_tag[tL1][t] = 1
    return matching_tag