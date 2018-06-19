"""
.. module:: evaluator
    :synopsis: evaluation method (f1 score and accuracy)
 

"""

import torch
import numpy as np
import itertools

import src.model.utils as utils
from torch.autograd import Variable

from src.model.crf import CRFDecode_vb


class eval_batch:
    """Base class for evaluation, provide method to calculate f1 score and accuracy 

    args: 
        packer: provide method to convert target into original space [TODO: need to improve] 
        l_map: dictionary for labels    
    """

    def __init__(self, packer, l_map):
        self.packer = packer
        self.l_map = l_map
        self.r_l_map = utils.revlut(l_map)

    def reset(self):
        """
        re-set all states
        """
        self.correct_labels = 0
        self.total_labels = 0
        self.gold_count = 0
        self.guess_count = 0
        self.overlap_count = 0

    def calc_f1_batch(self, decoded_data, target_data):
        """
        update statics for f1 score

        args:
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth
        """
        batch_decoded = torch.unbind(decoded_data, 1)
        batch_targets = torch.unbind(target_data, 0)

        for decoded, target in zip(batch_decoded, batch_targets):
            gold = self.packer.convert_for_eval(target)
            # remove padding
            length = utils.find_length_from_labels(gold, self.l_map)
            gold = gold[:length]
            best_path = decoded[:length]
            correct_labels_i, total_labels_i, gold_num, predict_num, right_num = self.eval_instance_mws(best_path.numpy(),gold.numpy())
            self.correct_labels += correct_labels_i
            self.total_labels += total_labels_i
            self.gold_count += gold_num
            self.guess_count += predict_num
            self.overlap_count += right_num

    def calc_acc_batch(self, decoded_data, target_data, task=None):
        """
        update statics for accuracy

        args:
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth
        """
        batch_decoded = torch.unbind(decoded_data, 1)
        batch_targets = torch.unbind(target_data, 0)

        for decoded, target in zip(batch_decoded, batch_targets):
            gold = self.packer.convert_for_eval(target)
            # remove padding
            length = utils.find_length_from_labels(gold, self.l_map)
            gold = gold[:length].numpy()
            best_path = decoded[:length].numpy()

            self.total_labels += length
            self.correct_labels += np.sum(np.equal(best_path, gold))

    def f1_score(self):
        """
        calculate f1 score based on statics
        """
        if self.guess_count == 0:
            return 0.0, 0.0, 0.0, 0.0
        precision = self.overlap_count / float(self.guess_count)
        recall = self.overlap_count / float(self.gold_count)
        if precision == 0.0 or recall == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        f = 2 * (precision * recall) / (precision + recall)
        accuracy = float(self.correct_labels) / self.total_labels
        return f, precision, recall, accuracy

    def acc_score(self):
        """
        calculate accuracy score based on statics
        """
        if 0 == self.total_labels:
            return 0.0
        accuracy = float(self.correct_labels) / self.total_labels
        return accuracy

    def eval_instance_mws(self, best_path, gold):
        total_labels = len(best_path)
        correct_labels = np.sum(np.equal(best_path, gold))

        best_path = list(best_path)
        gold = list(gold)

        best_path = [self.r_l_map[i] for i in best_path]
        gold = [self.r_l_map[i] for i in gold]

        dev_word_list = set()
        ans_word_list = set()
        cnt_char = len(best_path)
        dev_B_bef = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ans_B_bef = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(cnt_char):
            dev_tag = best_path[i].split('_')
            cnt_tag = len(dev_tag)
            for j in range(cnt_tag):
                if dev_tag[j] == 'E':
                    dev_word_list.add(str(dev_B_bef[j]) + '_' + str(i))
                elif dev_tag[j] == 'S':
                    dev_word_list.add(str(i))
                elif dev_tag[j] == 'B':
                    dev_B_bef[j] = i
        for i in range(cnt_char):
            ans_tag = gold[i].split('_')
            cnt_tag = len(ans_tag)
            for j in range(cnt_tag):
                if ans_tag[j] == 'B':
                    ans_B_bef[j] = i
                elif ans_tag[j] == 'S':
                    ans_word_list.add(str(i))
                elif ans_tag[j] == 'E':
                    ans_word_list.add(str(ans_B_bef[j]) + '_' + str(i))
        right_num = len(dev_word_list & ans_word_list)
        predict_num = len(dev_word_list)
        gold_num = len(ans_word_list)

        return correct_labels, total_labels, gold_num, predict_num, right_num


class eval_w(eval_batch):
    """evaluation class for word level model (LSTM-CRF)

    args:
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
        score_type: use f1score with using 'f'

    """

    def __init__(self, packer, l_map, score_type):
        eval_batch.__init__(self, packer, l_map)

        self.decoder = CRFDecode_vb(len(l_map), l_map['<start>'], l_map['<pad>'])

        if 'f' in score_type:
            self.eval_b = self.calc_f1_batch
            self.calc_s = self.f1_score
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def calc_score(self, ner_model, dataset_loader, illegal_idx, is_bichar):
        """
        calculate score for pre-selected metrics

        args:
            ner_model: LSTM-CRF model
            dataset_loader: loader class for test set
        """
        ner_model.eval()
        self.reset()
        for i in range(len(dataset_loader[0])):
            fea_v, tg_v, mask_v, bi_fea_v = self.packer.repack_vb(np.asarray(dataset_loader[0][i]),
                                                        np.asarray(dataset_loader[1][i]),
                                                        np.asarray(dataset_loader[2][i]),
                                                        np.asarray(dataset_loader[4][i]))
            ner_model.zero_grad()
            scores, hidden = ner_model(fea_v, bi_fea_v, dataset_loader[3][i], illegal_idx, self.l_map, is_bichar)
            decoded = self.decoder.decode(scores.data, mask_v.data)
            self.eval_b(decoded, torch.LongTensor(np.asarray(dataset_loader[1][i])).unsqueeze(0))
        return self.calc_s()

    def calc_predict(self, ner_model, dataset_loader, test_features, file_out, file_out_2, f_map):
        """
        calculate score for pre-selected metrics

        args:
            ner_model: LSTM-CRF model
            dataset_loader: loader class for test set
        """
        ner_model.eval()
        self.reset()
        idx2label = {v: k for k, v in self.l_map.items()}
        idx2word = {v: k for k, v in f_map.items()}
        for i in range(len(dataset_loader[0])):
            fea_v, tg_v, mask_v = self.packer.repack_vb(np.asarray(dataset_loader[0][i]),
                                                        np.asarray(dataset_loader[1][i]),
                                                        np.asarray(dataset_loader[2][i]))
            ner_model.zero_grad()
            scores, hidden = ner_model(fea_v, dataset_loader[3][i])
            decoded = self.decoder.decode(scores.data, mask_v.data)
            gold = [d % len(self.l_map) for d in dataset_loader[1][i]]
            # words = [idx2word[w] for w in dataset_loader[0][i]]
            length = utils.find_length_from_labels(gold, self.l_map)
            gold = gold[:length]
            words = test_features[i][:length]
            best_path = decoded.squeeze(1).tolist()[:length]
            gold = [idx2label[g] for g in gold]
            best_path = [idx2label[g] for g in best_path]
            for i in range(length):
                file_out.write("%s %s\n"%(words[i], best_path[i]))
            file_out.write("\n")

            sent = ''
            pos = None
            word = ''
            for i in range(length):
                if best_path[i].startswith('B'):
                    if pos != None:
                        sent += word + '_' + pos + ' '
                        word = ''
                        pos = None
                    word += words[i]
                    pos = best_path[i].split('-')[1]
                else:
                    assert pos != None
                    word += words[i]
            if len(word) > 0:
                sent += word + '_' + pos + ' '
            file_out_2.write("%s\n" % (sent))


class eval_wc(eval_batch):
    """evaluation class for LM-LSTM-CRF

    args: 
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
        score_type: use f1score with using 'f'

    """

    def __init__(self, packer, l_map, score_type):
        eval_batch.__init__(self, packer, l_map)

        self.decoder = CRFDecode_vb(len(l_map), l_map['<start>'], l_map['<pad>'])

        if 'f' in score_type:
            self.eval_b = self.calc_f1_batch
            self.calc_s = self.f1_score
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def calc_score(self, ner_model, dataset_loader):
        """
        calculate score for pre-selected metrics

        args: 
            ner_model: LM-LSTM-CRF model
            dataset_loader: loader class for test set
        """
        ner_model.eval()
        self.reset()

        for f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v in itertools.chain.from_iterable(dataset_loader):
            f_f, f_p, b_f, b_p, w_f, _, mask_v = self.packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v)
            scores = ner_model(f_f, f_p, b_f, b_p, w_f)
            decoded = self.decoder.decode(scores.data, mask_v.data)
            self.eval_b(decoded, tg)

        return self.calc_s()
