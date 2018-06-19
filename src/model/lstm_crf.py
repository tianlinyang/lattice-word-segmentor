# -*- coding:utf-8 -*-
"""
.. module:: lstm_crf
    :synopsis: lstm_crf

"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import src.model.crf as crf
import src.model.utils as utils
from src.model.Lattice_LSTMCell import LatticeLSTM


# from model.Lattice_LSTMCell_2 import LatticeLSTM
# from model.Lattice_LSTMCell import LatticeLSTM


class LSTM_CRF(nn.Module):
    """LSTM_CRF model

    args:
        vocab_size: size of word dictionary
        tagset_size: size of label set
        embedding_dim: size of word embedding
        hidden_dim: size of word-level blstm hidden dim
        rnn_layers: number of word-level lstm layers
        dropout_ratio: dropout ratio
        large_CRF: use CRF_L or not, refer model.crf.CRF_L and model.crf.CRF_S for more details
    """

    def __init__(self, char_vocab_size, bichar_vocab_size, word_vocab_size, tagset_size, embedding_dim, hidden_dim,
                 rnn_layers, dropout_ratio, gpu, isbiChar, large_CRF=True):
        super(LSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.char_vocab_size = char_vocab_size
        self.word_vocab_size = word_vocab_size
        self.bichar_vocab_size = bichar_vocab_size

        self.gpu = gpu
        # self.bidirectional = bidirectional

        self.word_embeds = nn.Embedding(word_vocab_size, embedding_dim)
        self.char_embeds = nn.Embedding(char_vocab_size, embedding_dim)
        self.bichar_embeds = nn.Embedding(bichar_vocab_size, embedding_dim)

        self.lstm = LatticeLSTM(embedding_dim, hidden_dim, isbiChar, self.gpu)

        # self.lstm = LatticeLSTM(embedding_dim, hidden_dim, bidirectional=bidirectional)
        self.rnn_layers = rnn_layers

        self.dropout1 = nn.Dropout(p=dropout_ratio)
        self.dropout2 = nn.Dropout(p=dropout_ratio)

        if isbiChar:
            self.dropout3 = nn.Dropout(p=dropout_ratio)

        self.tagset_size = tagset_size
        # if self.bidirectional:
        #     self.crf_in_dim = hidden_dim * 2
        # else:
        self.crf_in_dim = hidden_dim
        if large_CRF:
            self.crf = crf.CRF_L(self.crf_in_dim, tagset_size)
        else:
            self.crf = crf.CRF_S(self.crf_in_dim, tagset_size)

        self.batch_size = 1
        self.seq_length = 1

    def rand_init_hidden(self):
        """
        random initialize hidden variable
        """
        return autograd.Variable(
            torch.randn(2 * self.rnn_layers, self.batch_size, self.hidden_dim // 2)), autograd.Variable(
            torch.randn(2 * self.rnn_layers, self.batch_size, self.hidden_dim // 2))

    def set_batch_size(self, bsize):
        """
        set batch size
        """
        self.batch_size = bsize

    def set_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        tmp = sentence.size()
        self.seq_length = tmp[0]
        self.batch_size = tmp[1]

    def load_pretrained_embedding(self, pre_embeddings):
        """
        load pre-trained word embedding

        args:
            pre_word_embeddings (self.word_size, self.word_dim) : pre-trained embedding
        """

        assert (pre_embeddings[0].size()[1] == self.embedding_dim)
        self.char_embeds.weight = nn.Parameter(pre_embeddings[0])
        if len(pre_embeddings) == 2:
            self.word_embeds.weight = nn.Parameter(pre_embeddings[1])

    def rand_init(self, init_char_embedding=False, init_word_embedding=False, init_bichar_embedding=False):
        """
        random initialization

        args:
            init_embedding: random initialize embedding or not
        """
        if init_char_embedding:
            utils.init_embedding(self.char_embeds.weight)
        if init_bichar_embedding:
            utils.init_embedding(self.bichar_embeds.weight)
        if init_word_embedding:
            utils.init_embedding(self.word_embeds.weight)
        self.crf.rand_init()

    def forward(self, sentence, bi_fea, lexicon, illegal_idx, l_map, is_bichar, hidden=None):
        '''
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''
        self.set_batch_seq_size(sentence)

        if is_bichar:
            bichar_embeds = self.bichar_embeds(bi_fea)
            d_bichar_embeds = self.dropout3(bichar_embeds)

        embeds = self.char_embeds(sentence)
        d_embeds = self.dropout1(embeds)
        lexicon_start_embeds = self.dropout1(self.word_embeds(lexicon.lexicons_features_start))
        lexi = [[lexicon_start_embeds[id], lexicon.lexicons_len_start[id]] for id in
                range(len(lexicon.lexicons_len_start))]
        concat_embeds = d_embeds
        if is_bichar:
            concat_embeds = torch.cat([d_embeds, d_bichar_embeds], 2)
        h, c = self.lstm.forward(concat_embeds, lexi)

        lstm_out = h.view(-1, self.crf_in_dim)

        d_lstm_out = self.dropout2(lstm_out)

        crf_out = self.crf(d_lstm_out)
        crf_out = crf_out.view(self.seq_length, self.batch_size, self.tagset_size, self.tagset_size)

        for each in illegal_idx:
            arr = each.split('_')
            crf_out[:, :, int(arr[0]), int(arr[1])] = -10000
        crf_out[:, :, :, l_map['<start>']] = -10000
        crf_out[:, :, l_map['<pad>'], :] = -10000

        return crf_out, hidden