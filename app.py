from flask import Flask, request, jsonify, render_template, redirect, url_for
import logging
import traceback
import os
from dotenv import load_dotenv

# Importing models from models folder
from models import db, User, Session, Feedback, PerformanceMetrics, SensorData

# Importing services for the business logic
from services import build_lstm_autoencoder, run_isolation_forest, logical_check, run_analysis

# Importing additional utilities
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sqlalchemy import text
import uuid

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

# App configuration for SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')  # from .env
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db.init_app(app)

# Test database connection and log the result
with app.app_context():
    try:
        # Test database connection
        db.session.execute(text("SELECT 1"))
        db.session.commit()
        logger.info("✅ Successfully connected to the database!")
    except Exception as e:
        logger.error(f"❌ Failed to connect to the database: {e}")
        raise

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

# Start Session Route
@app.route('/start_session', methods=['POST'])
def start_session():
    try:
        data = request.get_json()
        movement_type = data.get('movement_type')

        # Create a new session in the database
        new_session = Session(
            id=uuid.uuid4(),
            trainer=None,  # Set appropriate trainer ID
            athlete=None,  # Set appropriate athlete ID
            lift_type=movement_type,
            sensor_data_id=None,  # Set this as per the sensor data
            performance_metric_id=None,  # Set performance metric ID
            feedback_id=None,  # Set feedback ID
            started_at=db.func.now(),
            ended_at=None,
            created_at=db.func.now(),
            updated_at=db.func.now()
        )

        db.session.add(new_session)
        db.session.commit()

        logger.info(f"✅ Session started with ID: {new_session.id}")

        return jsonify({'status': 'success', 'message': f"Session started with ID: {new_session.id}"}), 200

    except Exception as e:
        logger.error(f"❌ Error starting session: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# End Session Route
@app.route('/end_session', methods=['POST'])
def end_session():
    try:
        # Assuming the session ID is sent with the request to end the session
        data = request.get_json()
        session_id = data.get('session_id')

        session = Session.query.filter_by(id=session_id).first()
        if session:
            session.ended_at = db.func.now()
            db.session.commit()

            # Analyze session after it ends
            results = run_analysis()

            logger.info(f"✅ Session {session_id} ended successfully. Analysis results generated.")

            return jsonify({'status': 'success', 'results': results}), 200
        else:
            return jsonify({'status': 'error', 'message': 'Session not found'}), 404

    except Exception as e:
        logger.error(f"❌ Error ending session: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/')
def index():
    logger.info("Rendering static index page")
    return render_template('index.html')

@app.route('/signup', methods=['GET'])
def signup_page():
    # Render the signup.html page
    return render_template('signup.html')  # Render the signup.html page

@app.route('/signup', methods=['POST'])
def signup_user():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        # Use the signup function from auth.py
        user = signup(email, password)
        
        if "error" in user:
            return jsonify({'status': 'error', 'message': user["error"]}), 400
        
        return jsonify({'status': 'success', 'message': 'User registered successfully', 'user': user}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Error during signup: {str(e)}"}), 500


@app.route('/login', methods=['GET'])
def login_page():
    # Login page rendering
    return render_template('login.html')  # Render the login.html page

@app.route('/login', methods=['POST'])
def login_user():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        # Use the login function from auth.py
        user = login(email, password)

        if "error" in user:
            return jsonify({'status': 'error', 'message': user['error']}), 400
        
        # Login successful, redirect or handle session
        return redirect(url_for('dashboard'))  # Example redirect after login
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Error during login: {str(e)}"}), 500



@app.route('/analyze', methods=['POST'])
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
