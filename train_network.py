import os
import json
import sys
from flask import Response
from nltk.stem.porter import PorterStemmer
import nltk
import numpy as np
import tensorflow as tf
import tflearn
import pickle
import random
from pathlib import Path
import cache_response
from multiprocessing.pool import ThreadPool

pool = ThreadPool(processes=20)

stemmer = PorterStemmer()
words = {}
classes = {}
train_x = []
train_y = []
#model = tflearn.DNN


def loadModel(model_name):
    #global model
    global words
    global classes
    global train_x
    global train_y

    my_file = Path(
        "C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_dir}/{model_name}".format(model_dir=model_name,
                                                                                              model_name=model_name))
    if my_file.is_file():
        tf.reset_default_graph()
        data = pickle.load(open(my_file, "rb"))
        words[model_name] = data['words_{model}'.format(model=model_name)]
        classes[model_name] = data['classes_{model}'.format(model=model_name)]
        train_x = data['train_x_{model}'.format(model=model_name)]
        train_y = data['train_y_{model}'.format(model=model_name)]
        net = tflearn.input_data(shape=[None, len(train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
        net = tflearn.regression(net)
        model = tflearn.DNN(net,
                            tensorboard_dir='C:/Users/Amar/PycharmProjects/SpawnMLBackend/training_data/{model_name}/tflearn_logs'.format(
                                model_name=model_name))
        model_path = 'C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_dir}/{model_name}'.format(
            model_dir=model_name,
            model_name=model_name)
        # model.load(model_path)
        model.load(model_path)

    pass

def get_model(model_name):
    global words
    global classes
    global train_x
    global train_y

    my_file = Path(
        "C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_dir}/{model_name}".format(model_dir=model_name,
                                                                                              model_name=model_name))
    if my_file.is_file():
        #tf.reset_default_graph()
        data = pickle.load(open(my_file, "rb"))
        words[model_name] = data['words_{model}'.format(model=model_name)]
        classes[model_name] = data['classes_{model}'.format(model=model_name)]
        train_x = data['train_x_{model}'.format(model=model_name)]
        train_y = data['train_y_{model}'.format(model=model_name)]
        net = tflearn.input_data(shape=[None, len(train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
        net = tflearn.regression(net)
        model = tflearn.DNN(net,
                            tensorboard_dir='C:/Users/Amar/PycharmProjects/SpawnMLBackend/training_data/{model_name}/tflearn_logs'.format(
                                model_name=model_name))
        return model


def train_parallel(model_name):
    my_file = os.path.isdir(
        "C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_name}".format(model_name=model_name))
    if my_file == False:
        os.mkdir("C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_name}".format(model_name=model_name))
    async_train_result = pool.apply_async(train, (model_name,))
    return async_train_result.get()


def train(model_name):
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

    tf.reset_default_graph()
    tflearn.init_graph()
    net = tflearn.input_data(shape=[None, len(train_xinput[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)

    net = tflearn.fully_connected(net, len(train_youtput[0]), activation='softmax')
    net = tflearn.regression(net)

    model_nn = tflearn.DNN(net,
                           tensorboard_dir='C:/Users/Amar/PycharmProjects/SpawnMLBackend/training_data/{model_name}/tflearn_logs'.format(
                               model_name=model_name))
    model_nn.fit(train_xinput, train_youtput, n_epoch=50, batch_size=8, show_metric=True)

    model_path = 'C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_dir}/{model_name}'.format(
        model_dir=model_name,
        model_name=model_name)

    model_nn.save(model_path)

    pickle.dump(
        {'words_{model}'.format(model=model_name): words_list, 'classes_{model}'.format(model=model_name): inputclasses,
         'train_x_{model}'.format(model=model_name): train_xinput,
         'train_y_{model}'.format(model=model_name): train_youtput},
        open("C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_name}/{name}".format(
            model_name=model_name, name=model_name), "wb"))
    return {'message': 'success', 'model_name': model_name}


# probability threshold
ERROR_THRESHOLD = 0.60


def classify(sentence, model_name):
    res = None  # cache_response.get_cahce(sentence)
    if (res is not None):
        return res
    else:
        return process_response(sentence, model_name)


def process_response(sentence, model_name):
    model = get_model(model_name)
    model.load('C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_dir}/{model_name}'.format(
            model_dir=model_name,
            model_name=model_name))
    results = model.predict([bow(sentence, words.get(model_name))])[0]

    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    return_intent = []
    intent = []
    return_probability = 0
    msg_id = random.randint(0, 999)
    for r in results:
        return_list.append((classes.get(model_name)[r[0]], r[1]))

    if (len(return_list) > 0):
        return_intent = return_list[0]
        intent = return_intent[0]
        return_probability = str(return_intent[1])
        js = {
            "msg_id": msg_id,
            "text": sentence,
            "response": {
                "intent": [{
                    "confidence": return_probability,
                    "value": intent,

                }],

            }
        }
        cache_response.update_cache(sentence, js)
        return (js)
    else:
        js = {
            "msg_id": msg_id,
            "text": sentence,
            "entities": {
                "intent": [{
                    "confidence": "",
                    "value": "",

                }],
            }
        }
    cache_response.update_cache(sentence, js)
    return js


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

    return (np.array(bag))
