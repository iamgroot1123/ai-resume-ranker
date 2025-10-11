from flask import Flask
from .routes import main
from .utils import load_model_once
import os

def create_app():
    app = Flask(__name__)
    
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.config['PORT'] = int(os.environ.get('PORT', 8000))
    
    # Load the SBERT model once when the app starts.
    # This prevents reloading the model on every single request, which is critical for performance.
    with app.app_context():
        app.model = load_model_once()

    app.register_blueprint(main)
    
    return app
