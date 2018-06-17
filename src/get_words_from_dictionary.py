# -*- coding:utf-8 -*-
from __future__ import print_function
import datetime
import time
import codecs
import argparse


def get_dict_list(emb_file):

    words = []
    for line in open(emb_file, 'r', encoding='utf-8'):
        line = line.split(' ')
        if len(line) > 2 and len(line[0]) > 1:
        # if len(line) > 2 :
            words.append(line[0])

    return words

def count_dict(emb_file, output_file):
    file_out = codecs.open(output_file, "w+", encoding="utf-8")
    word = 0
    word_1 = 0
    word_2 = 0
    word_3 = 0
    for line in open(emb_file, 'r'):
        line = line.split(' ')
        if len(line) > 2:
            file_out.write("%s\n" % (line[0]))
        word += 1
        if len(line[0]) == 1:
            word_1 += 1
        elif len(line[0]) == 2:
            word_2 += 1
        elif len(line[0]) == 3:
            word_3 += 1
    print(word, word_1, word_2, word_3)

def compare_dict(emb1, emb2):
    word_list1 = get_dict_list(emb1)
    word_list2 = get_dict_list(emb2)
    same_word = set(word_list1)& set(word_list2)
    wl1_1 = [w for w in word_list1 if len(w) == 1]
    wl1_2 = [w for w in word_list1 if len(w) == 2]
    wl1_3 = [w for w in word_list1 if len(w) == 3]

    wl2_1 = [w for w in word_list2 if len(w) == 1]
    wl2_2 = [w for w in word_list2 if len(w) == 2]
    wl2_3 = [w for w in word_list2 if len(w) == 3]
    print(len(same_word))

def get_words(word_list, sentence):

    lexicons = []
    # print(sentence)
    for w in word_list:
        start = 0
        # print(w)
        while sentence.find(w, start) >= 0:
            lexicons.append([w, sentence.find(w, start), sentence.find(w, start) + len(w) - 1])
            start = sentence.find(w, start) + len(w) - 1

    return lexicons


def get_words_list(input_file, output_file, emb_file):

    word_list = get_dict_list(emb_file)
    file_out = codecs.open(output_file, "w", encoding="utf-8")

    with codecs.open(input_file, 'r', 'utf-8') as f:
        print(input_file)
        lines = f.readlines()
        sentence = ''
        for line in lines:
            if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
                line = line.rstrip('\n').split()
                if len(line[0]) > 1:
                    print(line[0])
                    sentence += line[0][0]
                else:
                    sentence += line[0]
            elif len(sentence) > 0:
                words = get_words(word_list, sentence)
                sentence = ''
                if len(words) > 0:
                    for w in words:
                        file_out.write("%s %s %s\n" % (w[0],w[1],w[2]))
                else:
                    file_out.write("<emp-dict>\n")
                file_out.write("\n")

def test_data_token_len(input_file):
    with codecs.open(input_file, 'r', 'utf-8') as f:
        print(input_file)
        lines = f.readlines()
        sentence = ''
        for line in lines:
            if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
                line = line.rstrip('\n')
                split = line.split('')
                if len(split[0]) > 1:
                    print(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning with BLSTM-CRF')
    parser.add_argument('--input_text_file', default='', help='input file')
    parser.add_argument('--out_lexicon_file', default='', help='out lexicon file')
    parser.add_argument('--dict_file', default='', help='each sentence in input file get its own lexicon from this emd file which could be treated as a dict file')
    args = parser.parse_args()
    get_words_list(args.text_file, args.lexicon_file, args.dict_file)
