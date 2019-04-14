from flask import Flask, request, json, Response, jsonify, render_template

import train_network
import sys
from functools import wraps
import _thread
from multiprocessing.pool import ThreadPool

import keras_train

app = Flask(__name__)

models = {'1': 'spawn_wiki', '2': 'spawn', '3': 'spawn_1'}
loading_models = {}

for loaded_model in models.values():
    keras_train.load_keras_model(loaded_model)
pool = ThreadPool(processes=4)


def check_auth(username, password):
    return username == 'username' and password == 'password'


def authenticate():
    message = {'message': 'You are not authorized user to access this url'}
    return Response(json.dumps(message), mimetype='application/json')


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth:
            return authenticate()

        elif not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)

    return decorated


@app.route('/')
def hello_world():
    return "Hello Spawn ML Bot Backend"


@app.route('/api/train', methods=['GET'])
# @requires_auth
def train():
    try:
        model_name = request.args.get('model_name')
        if (model_name is None):
            return (jsonify(
                {'message': 'Model name parameter is not defined / empty.', 'error': 'Model could be trained',
                 'status': 'error'}))

        train_msg = keras_train.train_parallel(model_name)

    except Exception as e:
        print(e)
        return (jsonify({'message': 'Error processing request.', 'error': 'Model could be trained', 'status': 'error'}))

    return jsonify(train_msg)


@app.route('/api/classify', methods=["GET"])
# @requires_auth
def classify():
    sentence = request.args.get('query')
    model_name = request.args.get('model_name')
    if (sentence is not None):
        return_list = keras_train.classifyKeras(sentence, model_name)
    else:
        return {'message': 'query cannot be empty', 'status': 'error'}
    return jsonify(return_list)
