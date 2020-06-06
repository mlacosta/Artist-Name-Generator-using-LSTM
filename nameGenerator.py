import numpy as np
import random
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU, TimeDistributed, LSTM, BatchNormalization


def sample(char_to_ix, vocab_size, model):

    model.reset_states()
    x = np.zeros((1, 1, vocab_size))

    indices = []

    newline_character = char_to_ix['\n']

    counter = 0
    idx = -1

    while (idx != newline_character and counter != 20):

        y = model.predict(x)
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())
        indices.append(idx)

        x = np.zeros((1, 1, vocab_size))
        x[:, :, idx] = 1

        counter += 1

    if (counter == 20):
        indices.append(char_to_ix['\n'])

    return indices


def one_hot(X, vocab_size):

    n = len(X)

    x_train = np.empty([1, n, vocab_size], dtype=float)

    for inx in range(n):

        one_hot_vec = np.zeros((1, vocab_size))
        index = X[inx]

        if index is not None:
            one_hot_vec[:, index] = 1

        x_train[0, inx, :] = one_hot_vec

    return x_train


def convert(s):

    # initialization of string to ""
    new = ""

    # traverse in the string
    for x in s:
        new += x

    # return string
    return new


data = open(
    r'names.txt',
    'r').read()
data = data.replace(u'\xa0', u' ')
chars = list(set(data))
print(sorted(chars))
data_size, vocab_size = len(data), len(chars)
print(
    'There are %d total characters and %d unique characters in your data.' %
     (data_size, vocab_size))


char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
print(ix_to_char)


# LSTM model

model = Sequential()

model.add(
    LSTM(
        vocab_size,
        batch_input_shape=(
            1,
            None,
            vocab_size),
        return_sequences=True,
        stateful=True,
        recurrent_dropout=0.3))

model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)


model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'])


# Trainning data

with open(r'names.txt') as f:
    examples = f.readlines()

examples = [x.lower().strip() for x in examples]


num_of_iter = 200

for j in range(num_of_iter):

    np.random.shuffle(examples)
    for inx in range(len(examples)):

        X = [None] + [char_to_ix[ch] for ch in examples[inx]]
        Y = X[1:] + [char_to_ix["\n"]]

        x_train = one_hot(X, vocab_size)
        y_train = one_hot(Y, vocab_size)

        model.fit(x_train, y_train, epochs=1, shuffle=False, verbose=1)
        model.reset_states()


new_words = 300

file1 = open("generated.txt", "a")
file1.write("\nNEW WORDS: \n\n")

for i in range(new_words):
    indices = sample(char_to_ix, vocab_size, model)
    file1.write(convert([ix_to_char[i] for i in indices]))

file1.close()
