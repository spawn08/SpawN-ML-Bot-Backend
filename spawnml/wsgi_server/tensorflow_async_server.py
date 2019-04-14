from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.wsgi import WSGIContainer

from spawnml.app.SpawnMLBackend import app

http_server = HTTPServer(WSGIContainer(app))
http_server.listen(4789, '0.0.0.0')
IOLoop.instance().start()
