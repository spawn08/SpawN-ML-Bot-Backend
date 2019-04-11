from flask import Flask, request, json, Response, jsonify, render_template

import train_network
import sys
from functools import wraps

app = Flask(__name__)

train_network.loadModel(model_name='spawn')


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


@app.route('/api/train')
@requires_auth
def train():
    model_name = request.args.get('model_name')
    train_network.train(model_name)
    return jsonify({'message': 'success', 'model_name': model_name})


@app.route('/api/classify', methods=["GET"])
@requires_auth
def classify():
    sentence = request.args.get('query')
    if (sentence is not None):
        return_list = train_network.classify(sentence)
    else:
        return {'message': 'query cannot be empty', 'status': 'error'}
    return jsonify(return_list)


if __name__ == '__main__':
    app.run(host='localhost', port='4567', threaded=True)
