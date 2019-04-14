import json
import os
import pickle
import random
from pathlib import Path

import nltk
import numpy as np
import tensorflow as tf
from flask import Response

import keras
from keras.models import load_model
from keras.layers import Dense
from keras.models import Sequential
from keras.models import model_from_json

from nltk.stem.lancaster import LancasterStemmer
from multiprocessing.pool import ThreadPool

pool = ThreadPool(processes=20)

stemmer = LancasterStemmer()
words = {}
classes = {}
train_x_dict = {}
train_y_dict = {}
multiple_models = {}
graph = tf.get_default_graph()


def load_keras_model(model_name):
    global words
    global classes
    global documents

    global train_x
    global train_y

    model_path = "C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_name}/{model_name}.json".format(
        model_name=model_name)
    model_path_h5 = 'C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_dir}/{model_name}.h5'.format(
        model_dir=model_name,
        model_name=model_name)
    if (os.path.isfile("C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_name}/{model_name}.json".format(
            model_name=model_name))):
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        my_file = Path("C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_name}/{name}".format(
            model_name=model_name, name=model_name))
        if my_file.is_file():
            data = pickle.load(open("C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_name}/{name}".format(
                model_name=model_name, name=model_name), "rb"))
            words[model_name] = data['words_{model}'.format(model=model_name)]
            classes[model_name] = data['classes_{model}'.format(model=model_name)]
            train_x_dict[model_name] = data['train_x_{model}'.format(model=model_name)]
            train_y_dict[model_name] = data['train_y_{model}'.format(model=model_name)]
        loaded_model = model_from_json(loaded_model_json)
        multiple_models[model_name] = get_model_keras(model_name)

        print("Loaded model from disk")

    pass


def get_model_keras(model_name):
    train_x = train_x_dict[model_name]
    train_y = train_y_dict[model_name]
    model_nn = Sequential()
    model_nn.add(Dense(12, input_dim=len(train_x[0]), activation='relu'))
    model_nn.add(Dense(8, activation='relu'))
    model_nn.add(Dense(len(train_y[0]), activation='softmax'))
    # file_path = 'C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_dir}/{model_name}.h5'.format(
    #    model_dir=model_name,
    #    model_name=model_name)
    # model_nn.load_weights(file_path)
    # model_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_nn


def train_keras(model_name):
    # tf.reset_default_graph()
    global graph
    output_data = []
    words_list = []
    inputclasses = []
    documents_vocab = []
    train_xinput = []
    train_youtput = []

    f = open("C:/Users/Amar/PycharmProjects/SpawnMLBackend/training_data/data_{model_name}.csv".format(
        model_name=model_name), 'rU')

    for line in f:
        cells = line.split(",")
        output_data.append((cells[0], cells[1]))

    f.close()

    print(output_data)
    print("%s sentences in training data" % len(output_data))

    ignore_words = ['?']

    for pattern in output_data:
        w = nltk.word_tokenize(pattern[1])
        words_list.extend(w)
        documents_vocab.append((w, pattern[0]))
        if pattern[0] not in inputclasses:
            inputclasses.append(pattern[0])

    words_list = [stemmer.stem(w.lower()) for w in words_list if w not in ignore_words]
    words_list = sorted(list(set(words_list)))

    # remove duplicates
    inputclasses = sorted(list(set(inputclasses)))

    print(len(documents_vocab), "documents_vocab")
    print(len(inputclasses), "classes", inputclasses)
    print(len(words_list), "unique stemmed words", words_list)

    training = []
    output_data = []
    output_empty = [0] * len(inputclasses)

    for doc in documents_vocab:
        bag = []
        pattern_words = doc[0]
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        for w in words_list:
            bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty)
        output_row[inputclasses.index(doc[1])] = 1

        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training)
    train_xinput = list(training[:, 0])
    train_youtput = list(training[:, 1])
    with graph.as_default():
        model_nn = Sequential()
        model_nn.add(Dense(12, input_dim=len(train_xinput[0]), activation='relu'))
        model_nn.add(Dense(8, activation='relu'))
        model_nn.add(Dense(len(train_youtput[0]), activation='softmax'))

        model_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_nn.fit(np.array(train_xinput), np.array(train_youtput), epochs=50, batch_size=8)

        model_path = 'C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_dir}/{model_name}.h5'.format(
            model_dir=model_name,
            model_name=model_name)
        model_nn.save(model_path)

    pickle.dump(
        {'words_{model}'.format(model=model_name): words_list, 'classes_{model}'.format(model=model_name): inputclasses,
         'train_x_{model}'.format(model=model_name): train_xinput,
         'train_y_{model}'.format(model=model_name): train_youtput},
        open("C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_name}/{name}".format(
            model_name=model_name, name=model_name), "wb"))

    # if os.path.isfile("C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_name}/{model_name}.json".format(
    #         model_name=model_name)):
    #    os.remove("C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_name}/{model_name}.json".format(
    #        model_name=model_name))
    #   model_json = model_nn.to_json()
    #    with open("C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_name}/{model_name}.json".format(
    #            model_name=model_name), "w") as json_file:
    #        json_file.write(model_json)

    #   model_nn.save_weights('C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_dir}/{model_name}.h5'.format(
    #         model_dir=model_name,
    #        model_name=model_name))
    #    print("Saved model to disk")
    # else:
    #    model_json = model_nn.to_json()
    #    with open("C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_name}/{model_name}.json".format(
    #           model_name=model_name), "w") as json_file:
    #        json_file.write(model_json)

    #    model_nn.save_weights('C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_dir}/{model_name}.h5'.format(
    #        model_dir=model_name,
    #        model_name=model_name))
    #    print("Saved model to disk")

    load_keras_model(model_name)
    return {'message': 'success', 'model_name': model_name}


def train_parallel(model_name):
    my_file = os.path.isdir(
        "C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_name}".format(model_name=model_name))
    if my_file == False:
        os.mkdir("C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_name}".format(model_name=model_name))
    async_train_result = pool.apply_async(train_keras, (model_name,))
    return async_train_result.get()


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)

    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return (bag)


def classifyKeras(sentence, model_name):

    with graph.as_default():

        loaded_model = multiple_models.get(model_name)
        file_path = 'C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_dir}/{model_name}.h5'.format(
            model_dir=model_name,
            model_name=model_name)
        print(file_path)
        loaded_model.load_weights(file_path)

        result = loaded_model.predict(np.array([bow(sentence, words.get(model_name))]))[0]
        result = [[i, r] for i, r in enumerate(result) if r > 0.25]
        result.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        return_intent = []
        intent = []
        return_probability = 0
        for r in result:
            return_list.append((classes.get(model_name)[r[0]], r[1]))

        if (len(return_list) > 0):
            return_intent = return_list[0]
            intent = return_intent[0]
            return_probability = str(return_intent[1])
            js = {
                "text": sentence,
                "entities": {
                    "intent": [{
                        "confidence": return_probability,
                        "value": intent,

                    }],
                }
            }
            return js
        else:
            js = {
                "text": sentence,
                "entities": {
                    "intent": [{
                        "confidence": "",
                        "value": "",

                    }],
                }
            }

        return js
