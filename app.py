from flask import Flask
from config import Config
from routes.main import main_bp
from routes.auth import auth_bp
import os
import datetime


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.permanent_session_lifetime = datetime.timedelta(days=7)

    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
