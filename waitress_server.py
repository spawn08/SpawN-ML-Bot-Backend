from waitress import serve
from SpawnMLBackend import app

serve(app, host='0.0.0.0', port=4789)
