# lattice-word-segmentor

Lattice-LSTM for multi-grained Chinese word segmentation.
Details and Models could be found in the paper Multi-Grained Chinese Word Segmentation.

## Requirements
Python 3.6  
torch 0.3.0

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
python get_words_from_dictionary.py --input_text_file ../data/train.txt --out_lexicon_file ../data/lexicon.train.txt --dict_file ../data/emb.txt
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
A new model can be trained using the follow command:

```
python train_w.py --train_file ../data/train.txt --dev_file ../data/dev.txt--test_file ../data/test.txt --lexicon_train_dir ../data/lexicon.train.txt --lexicon_dev_dir ../data/lexicon.dev.txt --lexicon_test_dir ../data/lexicon.test.txt --emb_file ../data/emb.txt --checkpoint ../checkpoint/
```
For each development evaluation, the F1-score on the development set is computed and compared to the previous best. If the current model is better, the previous model will be deleted and the current model will be saved. 

## Evaluation
A saving model can be evaltated on a test dataset using the follow command:

```
python eval_w.py
```
