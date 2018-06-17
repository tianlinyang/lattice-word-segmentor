
import torch
from torch import nn
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn import functional, init
import numpy as np

def orthonormal_initializer(output_size, input_size):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    print(output_size, input_size)
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
                    np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


class WordLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):

        super(WordLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_ih = nn.Linear(in_features=input_size,
                                   out_features=3 * hidden_size)
        self.linear_hh = nn.Linear(in_features=hidden_size,
                                   out_features=3 * hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        W = orthonormal_initializer(self.hidden_size, self.hidden_size + self.input_size)
        W_h, W_x = W[:, :self.hidden_size], W[:, self.hidden_size:]
        self.linear_ih.weight.data.copy_(torch.from_numpy(np.concatenate([W_x] * 3, 0)))
        self.linear_hh.weight.data.copy_(torch.from_numpy(np.concatenate([W_h] * 3, 0)))

        b = np.zeros(3 * self.hidden_size, dtype=np.float32)
        self.linear_ih.bias.data.copy_(torch.from_numpy(b))
        self.linear_hh.bias.data.copy_(torch.from_numpy(b))

    def forward(self, input, hx):

        h, c = hx
        lstm_vector = self.linear_ih(input) + self.linear_hh(h)
        i, f, g = lstm_vector.chunk(chunks=3, dim=1)
        i = self.sigmoid(i)
        f = self.sigmoid(f + 1)
        g = self.tanh(g)
        new_c = f * c + i * g

        return new_c

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)



class MultiInputLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size,is_bichar):
        super(MultiInputLSTMCell, self).__init__()

        if is_bichar:
            input_size = input_size*2
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.weight_ih = nn.Linear(in_features=input_size,
                                   out_features=4 * hidden_size)
        self.weight_hh = nn.Linear(in_features=hidden_size,
                                   out_features=4 * hidden_size)

        self.alpha_weight_ih = nn.Linear(in_features=input_size,
                                           out_features=hidden_size)
        self.alpha_weight_hh = nn.Linear(in_features=hidden_size,
                                           out_features=hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        W = orthonormal_initializer(self.hidden_size, self.hidden_size + self.input_size)
        W_h, W_x = W[:, :self.hidden_size], W[:, self.hidden_size:]
        self.weight_ih.weight.data.copy_(torch.from_numpy(np.concatenate([W_x] * 4, 0)))
        self.weight_hh.weight.data.copy_(torch.from_numpy(np.concatenate([W_h] * 4, 0)))

        W2 = orthonormal_initializer(self.hidden_size, self.hidden_size + self.input_size)
        W_h, W_x = W[:, :self.hidden_size], W2[:, self.hidden_size:]
        self.alpha_weight_ih.weight.data.copy_(torch.from_numpy(np.concatenate([W_x], 0)))
        self.alpha_weight_hh.weight.data.copy_(torch.from_numpy(np.concatenate([W_h], 0)))

        b = np.zeros(4 * self.hidden_size, dtype=np.float32)
        self.weight_ih.bias.data.copy_(torch.from_numpy(b))
        self.weight_hh.bias.data.copy_(torch.from_numpy(b))

        alpha_b = np.zeros(self.hidden_size, dtype=np.float32)
        self.alpha_weight_ih.bias.data.copy_(torch.from_numpy(alpha_b))
        self.alpha_weight_hh.bias.data.copy_(torch.from_numpy(alpha_b))

    def forward(self, input, lexi_input, hx):

        h_0, c_0 = hx
        batch_size = h_0.size(0)
        assert (batch_size == 1)
        lstm_vector = self.weight_ih(input) + self.weight_hh(h_0)
        i, f, o, g = lstm_vector.chunk(chunks=4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f + 1)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_num = len(lexi_input)

        if c_num == 0:
            c_1 = f * c_0 + i * g
            h_1 = o * torch.tanh(c_1)
        else:
            c_input_var = torch.cat(lexi_input, 0)
            alpha = torch.sigmoid(self.alpha_weight_ih(input) + self.alpha_weight_hh(c_input_var))
            alpha = torch.exp(torch.cat([i, alpha], 0))
            alpha_sum = alpha.sum(0)
            alpha = torch.div(alpha, alpha_sum)
            merge_i_c = torch.cat([g, c_input_var], 0)
            c_1 = merge_i_c * alpha
            c_1 = c_1.sum(0).unsqueeze(0)
            h_1 = o * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class LatticeLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, is_bichar, gpu):
        super(LatticeLSTM, self).__init__()
        self.gpu = gpu
        self.hidden_dim = hidden_dim
        self.rnn = MultiInputLSTMCell(input_dim, hidden_dim, is_bichar)
        self.word_rnn = WordLSTMCell(input_dim, hidden_dim)

        if self.gpu >= 0:
            self.rnn = self.rnn.cuda()
            self.word_rnn = self.word_rnn.cuda()

    def forward(self, input, lexicon, hidden=None):

        seq_len = input.size(0)
        batch_size = input.size(1)
        assert (batch_size == 1)
        hidden_out = []
        memory_out = []
        if hidden:
            (hx, cx) = hidden
        else:
            hx = autograd.Variable(torch.zeros(batch_size, self.hidden_dim))
            cx = autograd.Variable(torch.zeros(batch_size, self.hidden_dim))
            if self.gpu >= 0:
                hx = hx.cuda()
                cx = cx.cuda()

        id_list = range(seq_len)
        input_c_list = init_list_of_objects(seq_len)
        for t in id_list:
            (hx, cx) = self.rnn(input[t], input_c_list[t], (hx, cx))
            hidden_out.append(hx)
            memory_out.append(cx)
            if len(lexicon[t][1]) > 0:
                matched_num = len(lexicon[t][1])
                word_embs = lexicon[t][0].chunk(chunks=lexicon[t][0].size(0), dim=0)
                word_emb = torch.cat(word_embs[:matched_num], 0)
                ct = self.word_rnn(word_emb, (hx, cx))
                for idx in range(matched_num):
                    length = lexicon[t][1][idx]
                    input_c_list[t + length - 1].append(ct[idx, :].unsqueeze(0))

        output_hidden, output_memory = torch.cat(hidden_out, 0), torch.cat(memory_out, 0)

        return output_hidden.unsqueeze(0), output_memory.unsqueeze(0)


def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0, size):
        list_of_objects.append(list())
    return list_of_objects


