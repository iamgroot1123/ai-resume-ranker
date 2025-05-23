from flask import Flask
from .routes import main
import os

def create_app():
    app = Flask(__name__)
    
    # Configure app
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.config['PORT'] = int(os.environ.get('PORT', 8000))
    
    # Register blueprints
    app.register_blueprint(main)
    
    return app