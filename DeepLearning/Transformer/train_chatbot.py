'''
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-03-25 13:55:59
@LastEditors: Wang Yao
@LastEditTime: 2020-03-26 19:24:32
'''
import os
import re
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from transformer import Transformer, Noam, label_smoothing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dialogs_path = "xiaohuangji50w_fenciA.conv"
dialogs_size = 20000
vocab_size = 3000
max_seq_len = 10
model_dim = 512
batch_size = 256
epochs = 10

def load_data(data_path, dialogs_size, vocab_size=5000, max_len=10):
    with open(data_path, 'r') as f:
        data = f.read()
        relu = re.compile("E\nM (.*?)\nM (.*?)\n")
        match_dialogs = re.findall(relu, data)
        dialogs_size = len(match_dialogs)-1 if dialogs_size > len(match_dialogs) else dialogs_size
        dialogs = match_dialogs[:dialogs_size]
    questions = [dia[0] for dia in dialogs]
    answers = ['<start>/'+dia[1]+'/<stop>' for dia in dialogs]

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(questions + answers)
    questions_seqs = tokenizer.texts_to_sequences(questions)
    questions_seqs = pad_sequences(questions_seqs, maxlen=max_len)
    answers_seqs = tokenizer.texts_to_sequences(answers)
    answers_seqs = pad_sequences(answers_seqs, maxlen=max_len)
    
    decoder_targets = np.zeros((len(answers_seqs), max_len, vocab_size), dtype='float32')
    for i, seq in enumerate(answers_seqs):
        for j, index in enumerate(seq):
            if j > 0: decoder_targets[i, j-1, index-1] = 1
            if index == 0: break
    return questions_seqs, answers_seqs, decoder_targets

print("Data loading and tokenizing ... ")
question_inputs, answer_inputs, decoder_targets = load_data(
    dialogs_path, dialogs_size, vocab_size=vocab_size, max_len=max_seq_len)

print("Label smoothing ... ")
decoder_targets = label_smoothing(decoder_targets)

print("Model Building ... ")
encoder_inputs = Input(shape=(max_seq_len,), name='encoder_inputs')
decoder_inputs = Input(shape=(max_seq_len,), name='decoder_inputs')
outputs = Transformer(vocab_size, model_dim)([encoder_inputs, decoder_inputs])
model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)

model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9), 
    loss='categorical_crossentropy', metrics=['accuracy'])

print("Model Training ... ")
noam = Noam(model_dim)
model.fit([question_inputs, answer_inputs], decoder_targets, 
    batch_size=batch_size, epochs=epochs, validation_split=.2, callbacks=[noam])


    
    
    
