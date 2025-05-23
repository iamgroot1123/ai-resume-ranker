from flask import Flask
from .routes import main
import os

def create_app():
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB limit
    app.register_blueprint(main)
    
    if __name__ == '__main__':
        port = int(os.environ.get('PORT', 8000))
        app.run(host='0.0.0.0', port=port)
    
    return app