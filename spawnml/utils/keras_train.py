import json
import os
import pickle
import random
from multiprocessing.pool import ThreadPool
from pathlib import Path

import nltk
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from nltk.stem.lancaster import LancasterStemmer

pool = ThreadPool(processes=20)
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
model_base_path = os.path.abspath(os.path.join(ROOT_PATH, '..', '..', 'models/'))
training_base_path = os.path.abspath(os.path.join(ROOT_PATH, '..', '..', 'training_data/'))

stemmer = LancasterStemmer()
words = {}
classes = {}
train_x_dict = {}
train_y_dict = {}
multiple_models = {}
graph = tf.get_default_graph()
# ignore_words = ['?', 'how', 'are', 'is', 'you', 'a', 'an', 'the', 'who', 'what', 'whats', 'were', 'there', 'their',
#                'she', 'he', 'can']
ignore_words = ['?']


def load_keras_model(model_name):
    global words
    global classes
    global documents
    global train_x
    global train_y
    global multiple_models
    global model_base_path

    multiple_models[model_name] = None

    my_file = Path(model_base_path + "/{model_name}/{name}".format(
        model_name=model_name, name=model_name))
    if my_file.is_file():
        data = pickle.load(open(model_base_path + "/{model_name}/{name}".format(
            model_name=model_name, name=model_name), "rb"))
        words[model_name] = data['words_{model}'.format(model=model_name)]
        classes[model_name] = data['classes_{model}'.format(model=model_name)]
        train_x_dict[model_name] = data['train_x_{model}'.format(
            model=model_name)]
        train_y_dict[model_name] = data['train_y_{model}'.format(
            model=model_name)]
        print("Loaded model from disk")


def get_model_keras(model_name, file_path):
    train_x = train_x_dict[model_name]
    train_y = train_y_dict[model_name]
    model_nn = Sequential()
    model_nn.add(Dense(50, input_dim=len(train_x[0]), activation='relu'))
    model_nn.add(Dense(25, activation='relu'))
    model_nn.add(Dense(len(train_y[0]), activation='softmax'))
    model_nn.load_weights(file_path)
    return model_nn


def train_keras(model_name):
    global graph
    global ignore_words
    output_data = []
    words_list = []
    inputclasses = []
    documents_vocab = []
    train_xinput = []
    train_youtput = []

    tf.reset_default_graph()

    with open(training_base_path + '/training_data_{model}.json'.format(model=model_name), encoding='UTF-8') as f:
        data = json.load(f)

    output_data = list((data.get('rasa_nlu_data').get('common_examples')))
    print("%s sentences in training data" % len(output_data))

    for pattern in output_data:
        w = nltk.word_tokenize(pattern['text'])
        words_list.extend(w)
        documents_vocab.append((w, pattern['intent']))
        if pattern['intent'] not in inputclasses:
            inputclasses.append(pattern['intent'])

    words_list = [stemmer.stem(w.lower())
                  for w in words_list if w not in ignore_words]
    words_list = sorted(list(set(words_list)))

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
        model_nn.add(Dense(50, input_dim=len(
            train_xinput[0]), activation='relu'))
        model_nn.add(Dense(25, activation='relu'))
        model_nn.add(Dense(len(train_youtput[0]), activation='softmax'))

        model_nn.compile(loss='categorical_crossentropy',
                         optimizer='adam', metrics=['accuracy'])
        model_nn.fit(np.array(train_xinput), np.array(
            train_youtput), epochs=100, batch_size=8)

        model_path = model_base_path + '/{model_dir}/{model_name}.h5'.format(
            model_dir=model_name,
            model_name=model_name)
        model_nn.save(model_path)

    pickle.dump(
        {'words_{model}'.format(model=model_name): words_list, 'classes_{model}'.format(model=model_name): inputclasses,
         'train_x_{model}'.format(model=model_name): train_xinput,
         'train_y_{model}'.format(model=model_name): train_youtput},
        open(model_base_path + "/{model_name}/{name}".format(
            model_name=model_name, name=model_name), "wb"))

    load_keras_model(model_name)
    return {'message': 'success', 'model_name': model_name}


def train_parallel(model_name):
    my_file = os.path.isdir(
        model_base_path + "/{model_name}".format(model_name=model_name))
    if my_file == False:
        os.mkdir(
            model_base_path + "/{model_name}".format(model_name=model_name))
    async_train_result = pool.apply_async(train_keras, (model_name,))
    return async_train_result.get()


def clean_up_sentence(sentence):
    global ignore_words
    sentence_words = nltk.word_tokenize(str(sentence))

    sentence_words = [stemmer.stem(word.lower())
                      for word in sentence_words if word not in ignore_words]
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
        file_path = model_base_path + '/{model_dir}/{model_name}.h5'.format(
            model_dir=model_name,
            model_name=model_name)
        loaded_model = multiple_models.get(model_name)
        if loaded_model is None:
            multiple_models[model_name] = get_model_keras(
                model_name, file_path)
            loaded_model = multiple_models.get(model_name)

        result = loaded_model.predict(
            np.array([bow(sentence, words.get(model_name))]))[0]
        class_integer = np.argmax(result)
        intent_class = classes.get(model_name)[class_integer]
        probability = result[class_integer]

        if (probability > 0.55):

            js = {
                "text": sentence,
                "entities": {
                    "intent": [{
                        "confidence": str(probability),
                        "value": intent_class,

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


def classify_parallel(sentence, model_name):
    async_train_result = pool.apply_async(
        classifyKeras, (sentence, model_name))
    return async_train_result.get()
