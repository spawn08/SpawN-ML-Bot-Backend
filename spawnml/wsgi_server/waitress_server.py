from waitress import serve

from spawnml.app.SpawnMLBackend import app

serve(app, host='0.0.0.0', port=4789)
