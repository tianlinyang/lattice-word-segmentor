
from __future__ import print_function
import datetime
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
from model.crf import *
from model.lstm_crf import *
import model.utils as utils
from model.evaluator import eval_w

import argparse
import json
import os
import sys
from tqdm import tqdm
import itertools
import functools

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluating BLSTM-CRF')
    parser.add_argument('--load_arg', default='./checkpoint/lattice_word_seg.json.json', help='arg json file path')
    parser.add_argument('--load_check_point', default='./checkpoint/lattice_word_seg.json.model', help='checkpoint path')
    parser.add_argument('--gpu',type=int, default=0, help='gpu id')
    parser.add_argument('--eva_matrix', choices=['a', 'fa'], default='fa', help='use f1 and accuracy or accuracy alone')
    parser.add_argument('--test_file', default='./data/test.txt', help='path to test file, if set to none, would use test_file path in the checkpoint file')
    parser.add_argument('--lexicon_test_file', default='./data/lexicon.test.txt', help='path to test file, if set to none, would use test_file path in the checkpoint file')
    parser.add_argument('--bichar', type=bool, default=True, help='use bichar or not')

    args = parser.parse_args()

    with open(args.load_arg, 'r') as f:
        jd = json.load(f)
    jd = jd['args']

    checkpoint_file = torch.load(args.load_check_point, map_location=lambda storage, loc: storage)
    f_map = checkpoint_file['f_map']
    l_map = checkpoint_file['l_map']

    lexicon_f_map = checkpoint_file['lexicon_f_map']
    bichar_f_map = checkpoint_file['bichar_f_map']

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    if args.test_file:
        with codecs.open(args.test_file, 'r', 'utf-8') as f:
            test_lines = f.readlines()
    else:
        with codecs.open(jd['test_file'], 'r', 'utf-8') as f:
            test_lines = f.readlines()


    # converting format
    test_features, test_labels, test_bichar_features = utils.read_corpus(test_lines)

    with codecs.open(args.lexicon_test_file, 'r', 'utf-8') as f:
        lexicon_test_lines = f.readlines()
    lexicon_test_features, lexicon_feature_map = utils.read_corpus_lexicon(lexicon_test_lines, test_features,
                                                                           lexicon_f_map)
    lexicon_test_dataset = utils.padding_lexicon_bucket(lexicon_test_features, lexicon_f_map, args.gpu)

    # construct dataset
    test_dataset = utils.construct_bucket_mean_vb(test_features, test_labels, lexicon_test_dataset, f_map, l_map,
                                                  test_bichar_features, bichar_f_map, jd['caseless'])

    # build model
    ner_model = LSTM_CRF(len(f_map), len(bichar_f_map), len(lexicon_f_map), len(l_map), jd['embedding_dim'], jd['hidden'], jd['layers'], jd['drop_out'], args.gpu, jd['bidirectional'], jd['bichar'], large_CRF=jd['small_crf'])

    ner_model.load_state_dict(checkpoint_file['state_dict'])

    if args.gpu >= 0:
        if_cuda = True
        torch.cuda.set_device(args.gpu)
        ner_model.cuda()
        packer = CRFRepack(len(l_map), True)
    else:
        if_cuda = False
        packer = CRFRepack(len(l_map), False)

    evaluator = eval_w(packer, l_map, args.eva_matrix)

    r_l_map = {v: k for k, v in l_map.items()}
    matching_tag = utils.create_matching_labels_matrix(r_l_map).tolist()
    illegal_idx = []
    for i in range(len(matching_tag)):
        for j in range(len(matching_tag[i])):
            if matching_tag[i][j] == 0:
                illegal_idx.append(str(i) + '_' + str(j))

    if 'f' in args.eva_matrix:

        test_f1, test_pre, test_rec, test_acc = evaluator.calc_score(ner_model, test_dataset, illegal_idx, args.bichar)

        print(jd['checkpoint'] + 'test_f1: %.4f test_rec: %.4f test_pre: %.4f test_acc: %.4f\n' % (test_f1, test_pre, test_rec, test_acc))

    else:

        test_acc = evaluator.calc_score(ner_model, test_dataset, illegal_idx, args.bichar)

        print(jd['checkpoint'] + 'test_acc: %.4f\n' % test_acc)
