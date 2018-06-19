# lattice-word-segmentor

Lattice-LSTM for multi-grained Chinese word segmentation.
Details and Models could be found in the paper Multi-Grained Chinese Word Segmentation.

## Requirements
Python 3.6  
torch 0.3.0
## Data
We use multi-grained chinese segmentation datasets that was built by [Gong and Li et al.（2017）](http://www.aclweb.org/anthology/D/D17/D17-1072.pdf). And you could find details in their paper.
## Input format
Each line includes each character and its label. Sentences are splited with a null line.  

```
戴 B
相 I
龙 E
说 S
中 B
国 E
经 B
济 E
发 B
展 E
为 S
亚 B
洲 E
作 B_S
出 E_S
积 B
极 E
贡 B
献 E  

新 B
华 I
社 E
福 B
冈 E
5 B_B
月 I_E
1 I_B
1 I_I
日 E_E
电 S
( S
记 B
者 E
乐 B
绍 I
延 E
) S
```

## Lexicon file
Before we train the model,we generate each lexicon file for each dataset file with a dictionary file. And we treat a pretrained embedding file as a dictionary file.

```
python get_words_from_dictionary.py --input_text_file ../data/mws_dict/mannual-test-1500.BIES.txt --out_lexicon_file ../data/mws_dict/mws.test.dict.lexicon --dict_file ../data/emb.txt
```

For each sentence in the dataset file,we get its corresponding lexicon.Each line include three parts.One is a word.Another is the index of the first character of the word in the sentence. The other is the index of the last character of the word in the sentence. Lexicons are splited with a null line. The lexicon file format is as follows:

```
中国 4 5
发展 8 9
经济 6 7
积极 15 16
亚洲 11 12
贡献 17 18
作出 13 14
戴相龙 0 2
国经 5 6
说中 3 4
出积 14 15
戴相 0 1
相龙 1 2
济发 7 8

记者 12 13
新华社 0 2
新华 0 1
社福 2 3
福冈 3 4
华社 1 2
乐绍延 14 16
日电 9 10
绍延 15 16
```

## Training
A new model can be trained using the command：  

```
python train_w.py
```
There are many arguments in the python file __train_w.py__. Now we list some relatively main arguments in the following table, and you could see others in train_w.py.

Argument  | Description  |  Default
------------- | -------------  | -------------
--train_file   | train file path  | ../data/mws_dict/multigrain.alltrain.hwc.BIES.all.txt
--dev_file  | dev file path  | ../data/mws_dict/multigrain.alldev.hwc.BIES.txt
--test_file  | test file path  | ../data/mws_dict/mannual-test-1500.BIES.txt
--lexicon\_train_dir | train lexicon file path| ../data/mws_dict/mws.train.dict.lexicon
--lexicon\_dev_dir | dev lexicon file path| ../data/mws_dict/mws.dev.dict.lexicon
--lexicon\_test_dir  | test lexicon file path| ../data/mws_dict/mws.test.dict.lexicon
--checkpoint | path to checkpoint prefix| ../checkpoint/
--embedding_dim |dimension for character/word/bichar embedding | 50
--bichar | add bichar or not| False
--batch_size| batch size| 1
--layers | number of lstm layers| 1
--hidden | hidden dimension| 200
--drop_out | dropout ratio| 0.5
--update | optimizer method| SGD
--lr | initial learning rateo| 0.015
--lr_decay | decay ratio of learning rate| 0.05

For each development evaluation, the F1-score on the development set is computed and compared to the previous best. If the current model is better, the previous model will be deleted and the current model will be saved. 

## Evaluation
A saving model can be evaltated on a test dataset using the command

```
python eval_w.py
```
with the following arguments:

Argument  | Description  |  Default
------------- | -------------  | -------------
--load_arg   | arg json file path  | ../checkpoint/lattice\_word_seg.json
--load\_check_point   | checkpoint path  | ../checkpoint/lattice_word_seg.model
--test_file   | apath to test file, if set to none, would use test_file path in the checkpoint file | ../data/mws_dict/mannual-test-1500.BIES.txt
--lexicon_test_file   | path to test lexiconfile, if set to none, would use lexicon\_test_file path in the checkpoint file  | ../checkpoint/lattice\_word_seg.json

## Performance

model  | dev F  |  test F
------------- | -------------  | -------------
Gong and Li et al.（2017）   | 95.41  | 95.35
Gong and Li et al.（2017）(+bichar) | 96.59  | 95.97
Our model（binary as dict）  | 95.97  | 95.52
Our model（binary as dict）(+bichar)  | 96.96  | 96.18
Our model（now）(epoch 9)  | 96.78  | 96.29
Our model（now)(+pretrain emd）(epoch 7)  | 94.96 | 95.01