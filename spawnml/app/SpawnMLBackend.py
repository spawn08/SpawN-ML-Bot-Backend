from functools import wraps
from multiprocessing.pool import ThreadPool

from flask import Flask, request, json, Response, jsonify
import spacy
from spawnml.utils import keras_train

nlp = spacy.load("en_core_web_md")

app = Flask(__name__)

models = {'1': 'spawn_wiki', '2': 'spawn', '3': 'spawn_1'}
loading_models = {}
cache = {}
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
@requires_auth
def train():
    try:
        model_name = request.args.get('model_name')
        if (model_name is None):
            return (jsonify(
                {'message': 'Model name parameter is not defined / empty.', 'error': 'Model could not be trained',
                 'status': 'error'}))

        train_msg = keras_train.train_parallel(model_name)

    except Exception as e:
        print(e)
        return (jsonify(
            {'message': 'Error processing request.', 'error': 'Model could not be trained', 'status': 'error',
             'model_name': model_name}))

    return jsonify(train_msg)


@app.route('/api/classify', methods=["GET"])
@requires_auth
def classify():
    sentence = request.args.get('query')
    model_name = request.args.get('model_name')
    if (sentence is not None):
        return_list = keras_train.classifyKeras(sentence, model_name)
    else:
        return {'message': 'query cannot be empty', 'status': 'error', 'model_name': model_name}
    return jsonify(return_list)


@app.route('/entity', methods=['GET'])
@requires_auth
def get_ner():
    global cache
    entities = []
    labels = {}
    query = request.args.get('q')
    if (cache.get(query) is not None):
        return jsonify(cache.get(query))
    if query is not None:
        doc = nlp(query)
        for ent in doc.ents:
            labels['tag'] = ent.label_
            labels['value'] = ent.text
            entities.append(labels)
            labels = {}
            print(ent.text, ent.label_)
        cache[query] = entities
        if (len(entities) == 0):
            return jsonify([{'tag': '', 'value': query}])
    else:
        return jsonify([{'tag': '', 'value': query}])
    return jsonify(entities)
