from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import logging
import traceback
import os
from dotenv import load_dotenv

# Importing models from models folder
from models import db, User, Sessions, Feedback, PerformanceMetrics, SensorData

# Importing services for the business logic
from services import build_lstm_autoencoder, run_isolation_forest, logical_check, run_analysis

# Importing additional utilities
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sqlalchemy import text
import uuid
from utils.auth import auth, signup, login  # login buraya dikkat!
from utils.auth_decorator import login_required

# Load dotenv file
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app initialization
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'fallback_dev_key')

# App configuration for SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')  # from .env
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db.init_app(app)

# Initialize ML system
logger.info("Initializing anomaly detection system...")

try:
    scaler = MinMaxScaler()
    timesteps = 10
    n_features = 3
    logger.debug("Building LSTM Autoencoder model...")
    lstm_model = build_lstm_autoencoder(timesteps, n_features)
    logger.info("LSTM Autoencoder initialized successfully")

    if os.path.exists('trained_lstm_model.h5'):
        logger.info("Loading trained LSTM model weights...")
        lstm_model.load_weights('trained_lstm_model.h5')
        logger.info("LSTM model weights loaded successfully")
    else:
        logger.warning("No pre-trained LSTM model found. Model is untrained.")

except Exception as e:
    logger.error(f"Initialization error: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# ================== ROUTES ====================

@app.route('/')
def index():
    logger.info("Rendering static index page")
    return render_template('index.html')

@app.route('/signup', methods=['GET'])
def signup_page():
    return render_template('signup.html')

@app.route('/signup', methods=['POST'])
def signup_user():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        role = data.get('role')

        # Firebase kullanıcı oluştur
        user = auth.create_user_with_email_and_password(email, password)

        # PostgreSQL'e kaydet
        new_user = User(
            first_name=first_name,
            last_name=last_name,
            email=email,
            role=role,
            firebase_uid=user['localId']
        )
        db.session.add(new_user)
        db.session.commit()

        return jsonify({'status': 'success', 'message': 'User registered successfully'}), 200

    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        return jsonify({'status': 'error', 'message': f"Error during signup: {str(e)}"}), 500

@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_user():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'status': 'error', 'message': 'Email and password are required'}), 400

        # Doğru login fonksiyonunu çağırıyoruz
        user = login(email, password)

        if "error" in user:
            return jsonify({'status': 'error', 'message': user['error']}), 400

        # ✅ Login başarılı
        session['user_id'] = user['id']
        logger.info(f"User logged in successfully: {user['email']}")
        return jsonify({'status': 'success', 'message': 'Login successful! Redirecting...', 'redirect_url': '/dashboard'}), 200

    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        return jsonify({'status': 'error', 'message': f"Error during login: {str(e)}"}), 500

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/sessions')
@login_required
def sessions():
    logger.info("Rendering sessions page")

    try:
        # Trainer'ları veritabanından çekiyoruz
        trainers = User.query.filter_by(role='trainer').all()

        # İsimleri birleştiriyoruz
        trainer_options = [
            {'id': str(trainer.id), 'full_name': f"{trainer.first_name} {trainer.last_name}"}
            for trainer in trainers
        ]
    
        return render_template('sessions.html', trainers=trainer_options)

    except Exception as e:
        logger.error(f"Error loading trainers: {str(e)}")
        return "An error occurred loading the sessions page", 500


@app.route('/sessions/start', methods=['POST'])
@login_required
def start_session():
    try:
        data = request.get_json()
        lift_type = data.get('lift_type')
        trainer_id = data.get('trainer_id')

        # 🛑 Burada GİRİŞ yapan kullanıcıyı çekiyoruz (Flask session'dan)
        athlete_id = session.get('user_id')

        if not athlete_id:
            logger.warning("Unauthorized attempt to start a session (no athlete ID)")
            return jsonify({'status': 'error', 'message': 'Unauthorized: No athlete info'}), 401

        if not lift_type or not trainer_id:
            logger.warning("Missing required fields for starting session")
            return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

        new_session = Sessions(
            id=uuid.uuid4(),
            lift_type=lift_type,
            trainer=trainer_id,
            athlete=athlete_id,
            started_at=db.func.now(),
            status='ongoing'
        )

        db.session.add(new_session)
        db.session.commit()

        logger.info(f"✅ Session started with ID: {new_session.id}")

        return jsonify({
            'status': 'success',
            'message': f"Session started with ID: {new_session.id}"
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"❌ Error starting session: {str(e)}")
        return jsonify({'status': 'error', 'message': f"Error: {str(e)}"}), 500


@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    try:
        data = request.get_json()
        movement_type = data.get('movement_type')

        # Run analysis
        results = run_analysis()

        return jsonify({'status': 'success', 'results': results})
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask anomaly detection API (file-based with plot)...")
    app.run(debug=True, port=5000)
    logger.info("Flask API started successfully")
    logger.info("Flask API is running on port 5000")
    logger.info("Flask API is ready to accept requests")
