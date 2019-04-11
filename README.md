# SpawN ML Bot Backend
### The high scalable intent classification engine for chatbot. The model built is based on Bag of words approach. This is best suited for in domain classification task.

#### Note:- The application can be run using Python3 only

 
>* This repo contains the text classification back end wrtten in flask
>* This uses Tensorflows TFLearn library for building the neural network model
>* **The flask app is run using the wsgi wrapper server using Tornado for better concurrency**
>* **This repo can be used for making high scalable backend for building the chatbots**

## 1. Project Structure

### The project has following structure:-
>* **models/ - This directory contains the models built by TFLearn. This directory is used for saving and loding the models. The directory also has a pickle.**
>* **training_data - This folder contains the data file for training the neural network. Also the tflearn logs are stored inside this folder.**
>* **SpawnMLBackend.py - Flask webservice python file. The file has also decorator written for authentication of the webservice. The authentication used is Basic Authentication. This can be changed as per your requirement.**
>* **train_network.py - All the training logic,loading models at startup and classification logic is written in this file.**
>* **tensorflow_async_server.py - Flask server is not suitable for deploying in production. The wsgi wrapper is written using Tornado webserver for high scalability.**

## 2. Running application
##### To run this application , install the depencencies as:
>* **pip install tensorflow**
>* **pip install tflearn**
>* **pip install flask**
>* **pip install tornado**
>* **pip install nltk - you will also need to download 'punkt' of nltk.**
>* **pip install pathlib**

####After installing all the dependencies, run the app as:
>* python3 tensorflow_async_server.py
>* For background running task- nohup python3 tensorflow_async_server.py &

## 2. Links

>* [TFLearn](http://tflearn.org/)
>* [Tensorflow](https://www.tensorflow.org/)
