from flask import Flask
from .routes import main

def create_app():
    app = Flask(__name__)
    # app.config["UPLOAD_FOLDER"] = "uploads"
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
    
    app.register_blueprint(main)
    
    return app