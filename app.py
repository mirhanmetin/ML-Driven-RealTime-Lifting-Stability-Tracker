from flask import Flask, request, jsonify, render_template, redirect, url_for, session as flask_session
import logging
import traceback
import os
from dotenv import load_dotenv
from flask_socketio import SocketIO, emit, join_room

import threading
import time

# Importing models from models folder
from models import db, User, Sessions, Feedback, PerformanceMetrics, SensorData

# Importing services for the business logic
from services import build_lstm_autoencoder, run_isolation_forest, logical_check, run_analysis_realtime

# Importing additional utilities
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
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

# Flask app’ine SocketIO ekle
socketio = SocketIO(app)

# App configuration for SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
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
        lstm_model.load_weights('trained_lstm_model.weights.h5')
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

        user = auth.create_user_with_email_and_password(email, password)

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

        user = login(email, password)

        if "error" in user:
            return jsonify({'status': 'error', 'message': user['error']}), 400

        flask_session['user_id'] = user['id']
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
        trainers = User.query.filter(User.role == 'trainer').all()
        trainer_options = [
            {'id': str(trainer.id), 'full_name': f"{trainer.first_name} {trainer.last_name}"}
            for trainer in trainers
        ]
        return render_template('sessions.html', trainers=trainer_options)
    except Exception as e:
        logger.error(f"Error loading trainers: {str(e)}")
        return "An error occurred loading the sessions page", 500

@app.route('/sessions', methods=['POST'])
@login_required
def start_session():
    try:
        data = request.get_json()
        lift_type = data.get('lift_type')
        trainer_id = data.get('trainer_id')
        athlete_id = flask_session.get('user_id')

        if not athlete_id:
            logger.warning("Unauthorized attempt to start a session (no athlete ID)")
            return jsonify({'status': 'error', 'message': 'Unauthorized: No athlete info'}), 401

        if not lift_type or not trainer_id:
            logger.warning("Missing required fields for starting session")
            return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

        new_session = Sessions(
            id=str(uuid.uuid4()),
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
            'message': f"Session started with ID: {new_session.id}",
            'redirect_url': f"/sessions/{new_session.id}"
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"❌ Error starting session: {str(e)}")
        return jsonify({'status': 'error', 'message': f"Error: {str(e)}"}), 500


@app.route('/sessions/<session_id>')
@login_required
def session_detail(session_id):
    try:
        current_session = Sessions.query.filter(Sessions.id == session_id).first()
        if not current_session:
            return "Session not found", 404

        sensor_data_entry = SensorData.query.filter(SensorData.session == session_id).first()
        data_list = []
        if sensor_data_entry and sensor_data_entry.raw_data:
            data_list = sensor_data_entry.raw_data

        return render_template(
            'session_detail.html',
            session=current_session,
            sensor_data=data_list
        )
    except Exception as e:
        logger.error(f"Error loading session detail: {str(e)}")
        return "An error occurred loading the session detail page.", 500

@app.route('/sessions/<session_id>/sensor_data', methods=['POST'])
@login_required
def upload_sensor_data(session_id):
    try:
        # Check if file is in request
        if 'csv_file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file part in request'}), 400

        file = request.files['csv_file']

        # Check if session exists
        session_obj = Sessions.query.filter_by(id=session_id).first()
        if not session_obj:
            return jsonify({'status': 'error', 'message': 'Session not found'}), 404

        # Read CSV into DataFrame
        df = pd.read_csv(file)
        required_columns = {'left_foot_pressure', 'right_foot_pressure', 'core_stability'}
        if not required_columns.issubset(df.columns):
            return jsonify({'status': 'error', 'message': 'CSV missing required columns'}), 400

        # Save raw sensor data
        sensor_data_list = df.to_dict(orient='records')
        new_sensor_data = SensorData(
            id=str(uuid.uuid4()),
            session=session_obj.id,
            athlete=session_obj.athlete,
            raw_data=sensor_data_list
        )

        db.session.add(new_sensor_data)
        db.session.flush()  # Get generated ID

        # Update session with sensor_data_id
        session_obj.sensor_data_id = new_sensor_data.id
        db.session.commit()

        logger.info(f"✅ Sensor data saved for session {session_id}, starting real-time analysis...")

        # Start real-time analysis in background
        threading.Thread(target=run_analysis_realtime, args=(session_id, df, socketio)).start()

        return jsonify({'status': 'success', 'message': 'Sensor data uploaded and real-time analysis started.'}), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"❌ Error uploading sensor data for session {session_id}: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f"Internal error: {str(e)}"}), 500

@app.route('/sessions/<session_id>/end', methods=['POST'])
@login_required
def end_session(session_id):
    try:
        # Fetch session
        current_session = Sessions.query.filter(Sessions.id == session_id).first()
        if not current_session:
            return jsonify({'status': 'error', 'message': 'Session not found'}), 404

        if current_session.status == 'ended':
            return jsonify({'status': 'error', 'message': 'Session already ended'}), 400

        # Fetch sensor data
        sensor_data_entry = SensorData.query.filter_by(session=session_id).first()
        if not sensor_data_entry or not sensor_data_entry.raw_data:
            return jsonify({'status': 'error', 'message': 'No sensor data found for this session'}), 404

        df = pd.DataFrame(sensor_data_entry.raw_data)
        features = ['left_foot_pressure', 'right_foot_pressure', 'core_stability']
        if not set(features).issubset(df.columns):
            return jsonify({'status': 'error', 'message': 'Sensor data missing required features'}), 400

        # Calculate performance metrics
        avg_left = float(df['left_foot_pressure'].mean())
        avg_right = float(df['right_foot_pressure'].mean())
        avg_core = float(df['core_stability'].mean())
        balance_score = float(1 - abs(avg_left - avg_right))
        stability_score = float(avg_core)
        injury_risk = float(1 - (balance_score * stability_score))

        # Save performance metrics
        perf_metrics_id = str(uuid.uuid4())
        perf_metrics = PerformanceMetrics(
            id=perf_metrics_id,
            session=session_id,
            balance_score=balance_score,
            stability_score=stability_score,
            injury_risk=injury_risk
        )
        db.session.add(perf_metrics)
        db.session.flush()  # ⭐ flush ensures perf_metrics.id exists in DB for foreign key

        # Create feedback text
        alerts = []
        if balance_score < 0.6:
            alerts.append("Denge sorunları tespit edildi.")
        if stability_score < 0.6:
            alerts.append("Çekirdek stabilitesi düşük.")
        if injury_risk > 0.7:
            alerts.append("Yüksek sakatlanma riski.")
        if not alerts:
            alerts.append("Performans mükemmel, hiçbir sorun bulunmadı!")

        # Save feedback
        feedback = Feedback(
            id=str(uuid.uuid4()),
            session=session_id,
            metrics_id=perf_metrics_id,  # connect to metrics
            feedback_text=" ".join(alerts)
        )
        db.session.add(feedback)

        # Mark session as ended
        current_session.status = 'ended'
        current_session.ended_at = db.func.now()

        # Commit all at once
        db.session.commit()

        logger.info(f"✅ Session {session_id} ended. Metrics and feedback saved.")

        return jsonify({
            'status': 'success',
            'message': 'Session ended successfully. Metrics and feedback saved.',
            'redirect_url': f"/sessions/{session_id}/details"
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"❌ Error ending session {session_id}: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f"Internal error: {str(e)}"}), 500

@app.route('/sessions/<session_id>/details', methods=['GET'])
@login_required
def session_details_page(session_id):
    try:
        current_session = Sessions.query.filter(Sessions.id == session_id).first()
        if not current_session:
            return "Session not found", 404

        metrics = PerformanceMetrics.query.filter_by(session=session_id).first()
        feedback = Feedback.query.filter_by(session=session_id).first()

        if not metrics or not feedback:
            return "Metrics or feedback not found for this session", 404

        return render_template(
            'result.html',
            session=current_session,
            metrics=metrics,
            feedback=feedback
        )
    except Exception as e:
        logger.error(f"❌ Error loading session details: {str(e)}", exc_info=True)
        return "An error occurred loading session details page.", 500


@socketio.on('connect')
def handle_connect():
    logger.info("Client connected.")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected.")


@socketio.on('join')
def on_join(room_id):
    join_room(room_id)
    logger.info(f"Client joined room {room_id}")

if __name__ == '__main__':
    logger.info("Starting Flask anomaly detection API...")
    socketio.run(app, debug=True, port=5000)

