'''
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-04-02 14:13:11
@LastEditors: Wang Yao
@LastEditTime: 2020-04-02 17:24:19
'''
import os
import sys
sys.path.append("../")
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from Embeddings.embeddings import Embedding
from rnn import RNN
from gru import GRU
from recurrent import BiDirectional
from attention import Attention

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

vocab_size = 5000
max_len = 256
model_dim = 64
batch_size = 128
epochs = 10

print("Data downloading and pre-processing ... ")
(x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen=max_len, num_words=vocab_size)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('Model building ... ')
inputs = Input(shape=(max_len,), name="inputs")
embeddings = Embedding(vocab_size, model_dim)(inputs)
outputs = BiDirectional(GRU(model_dim, return_outputs=True))(embeddings)
x = GlobalAveragePooling1D()(outputs)
x = Dropout(0.2)(x)
x = Dense(10, activation='relu')(x)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9), 
    loss='categorical_crossentropy', metrics=['accuracy'])

print("Model Training ... ")
es = EarlyStopping(patience=5)
model.fit(x_train, y_train, 
    batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[es])

test_metrics = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print("loss on Test: %.4f" % test_metrics[0])
print("accu on Test: %.4f" % test_metrics[1])

