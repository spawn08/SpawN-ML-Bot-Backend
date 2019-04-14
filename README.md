## [SpawN ML Bot Backend V 1.0.0]((https://placehold.it/15/1589F0/000000?text=+) `#1589F0`)

### The high scalable intent classification engine for chatbot. The model built is based on Bag of words approach. This is best suited for in domain classification task.
#### SpawN ML now support loading multiple models for inference for intent classification. 
#### Note:- The application can be run using Python3 only

 
>* **This repo contains the text classification back end wrtten in flask**
>* **This uses Tensorflows TFLearn library for building the neural network model**
>* **The flask app is run using the wsgi wrapper server using Tornado for better concurrency**
>* **This repo can be used for making high scalable backend for building the chatbots**

## 1. Project Structure

### The project has following structure:-
>* **models/ - This directory contains the models built by TFLearn. This directory is used for saving and loding the models. The directory also has a pickle.**
>* **training_data - This folder contains the data file for training the neural network. Also the tflearn logs are stored inside this folder.**
>* **SpawnMLBackend.py - Flask webservice python file. The file has also decorator written for authentication of the webservice. The authentication used is Basic Authentication. This can be changed as per your requirement.**
>* **train_network.py - All the training logic,loading models at startup and classification logic is written in this file.**
>* **tensorflow_async_server.py - Flask server is not suitable for deploying in production. The wsgi wrapper is written using Tornado webserver for high scalability.**
>* **waitress_server.py - Waitress is meant to be a production-quality pure-Python WSGI server with very acceptable performance. You can either server the model using tensorflow_async_server or waitress_server**

#### Note: _Please change the directory inside train_network.py and keras_train.py to your specific directory. In future release, this will be fixed._

#### _In my load test, I found Tornado webserver to be more stable than waitress. Waitress gave me higher throughput than Tornado webserver but Tornado webserver gave me better stability with zero failed request._ 

## 2. Running application
##### To run this application , install the depencencies as:
>* **pip install tensorflow**
>* **pip install tflearn**
>* **pip install flask**
>* **pip install tornado**
>* **pip install nltk - you will also need to download 'punkt' of nltk.**
>* **pip install pathlib**

#### After installing all the dependencies, run the app as:
>* **python3 tensorflow_async_server.py**
>* **For background running task**- **nohup python3 tensorflow_async_server.py &**

##### Note: The default authentication for testing is username=username, pass- password. You can change the authentication as per your requirement.
>* Basic dXNlcm5hbWU6cGFzc3dvcmQ=

## 3. Links

>* [TFLearn](http://tflearn.org/)
>* [Tensorflow](https://www.tensorflow.org/)
>* [Tornado](https://www.tornadoweb.org/en/stable/)
>* [Waitress](https://github.com/Pylons/waitress)


### Future Release: Docker support, Dynamic model building, Web UI for building the models.
