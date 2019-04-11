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

stemmer = PorterStemmer()
output = []
words = []
classes = []
documents = []
train_x = []
train_y = []
intents = []
model = tflearn.DNN

interpreter = None

def loadModel(model_name):
    global model
    global words
    global classes
    global documents
    global model
    global train_x
    global train_y
    global interpreter

    my_file = Path("C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_name}".format(model_name=model_name))
    if my_file.is_file():
        tf.reset_default_graph()
        data = pickle.load(open(my_file, "rb"))
        words = data['words']
        classes = data['classes']
        train_x = data['train_x']
        train_y = data['train_y']
        net = tflearn.input_data(shape=[None, len(train_x[0])])
        net = tflearn.fully_connected(net, 32)
        net = tflearn.fully_connected(net, 32)
        net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
        net = tflearn.regression(net)
        model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
        model.load('C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_name}'.format(model_name=model_name))

    pass


def train(model_name):
    global interpreter
    global output
    global words
    global classes
    global documents
    global model
    global train_x
    global train_y
    global intents

    f = open("C:/Users/Amar/PycharmProjects/SpawnMLBackend/training_data/data.csv", 'rU')

    for line in f:
        cells = line.split(",")
        output.append((cells[0], cells[1]))

    f.close()

    print(output)
    print("%s sentences in training data" % len(output))

    ignore_words = ['?']

    for pattern in output:
        w = nltk.word_tokenize(pattern[1])
        words.extend(w)
        documents.append((w, pattern[0]))
        if pattern[0] not in classes:
            classes.append(pattern[0])

    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    # remove duplicates
    classes = sorted(list(set(classes)))

    print(len(documents), "documents")
    print(len(classes), "classes", classes)
    print(len(words), "unique stemmed words", words)

    training = []
    output = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training)
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    tf.reset_default_graph()
    tflearn.init_graph()
    net = tflearn.input_data(shape=[None, len(train_x[0])])
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 32)

    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
    net = tflearn.regression(net)

    model = tflearn.DNN(net, tensorboard_dir='C:/Users/Amar/PycharmProjects/SpawnMLBackend/training_data/tflearn_logs')
    model.fit(train_x, train_y, n_epoch=50, batch_size=16, show_metric=True)
    model.save('C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{model_name}'.format(model_name=model_name))

    pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y},
                open("C:/Users/Amar/PycharmProjects/SpawnMLBackend/models/{name}".format(name=model_name), "wb"))
    pass


# probability threshold
ERROR_THRESHOLD = 0.60


def classify(sentence):
    res = cache_response.get_cahce(sentence)
    if (res is not None):
        return res
    else:
        return process_response(sentence)


def process_response(sentence):
    results = model.predict([bow(sentence, words)])[0]

    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    return_intent = []
    intent = []
    return_probability = 0
    msg_id = random.randint(0, 999)
    for r in results:
        return_list.append((classes[r[0]], r[1]))

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
