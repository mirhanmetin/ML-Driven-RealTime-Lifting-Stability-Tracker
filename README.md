ML-Driven Real-Time Lifting Stability Tracker

A machine learning-powered real-time monitoring system designed to analyze balance and stability during powerlifting exercises (squat, bench press, deadlift).
It uses LSTM autoencoders and hybrid anomaly detection models (Isolation Forest, One-Class SVM) to detect posture instabilities and atypical lifting patterns.
The system provides automated feedback to athletes and trainers based on real-time sensor data, aiming to reduce injury risk and improve lifting technique.


Project Requirements & Setup Guide

Python Environment Requirements

    Python 3.x → Specifically, a version compatible with TensorFlow (we recommend Python 3.10)

Initial Setup (macOS with Homebrew)

    Install Python 3.10:

        $ brew install python@3.10

    Create a virtual environment:

        $ python3.10 -m venv venv310
        $ source venv310/bin/activate

    You should now see your shell like:

        $ (venv310) yourmachine%

Python Library Requirements

    We use several key libraries for scientific computing, data manipulation, visualization, and machine learning:

    - Numpy
        -> One of the most fundamental scientific calculation library
        -> Using for Array and matrix operations
        -> In this project, Numpy using in:
            -- Converting data to 3-dimensional arrays for LSTM
            -- MSE calculation
    - Pandas
        -> One of the most common library for data analysis and manipulation
        -> Processes data tables like Excel files with DataFrame structure
        -> In this project, Pandas using in:
            -- Reading CSV files
            -- Add column to result via converting DataFrame
    - Matplotlib
        -> One of the most popular graph-draw library
        -> Using for visualization like graph, scatter plot, and bar-chart
        -> In this project, Matplotlib using in:
            -- Visualization
    - Scikit-Learn
        -> One of the most basic ML library
        -> Includes hundreds of model
        -> In this project, Scikit-Learn using in:
            -- IsolationForest
            -- OneClassSVM
            -- Data scaling
    - Tensorflow
        -> Huge library for deep learning
        -> Includes models such as LSTM, CNN, and Autoencoder
        -> In this project, Scikit-Learn using in:
            -- Creating LSTM autoencoder model
            -- Training LSTM autoencoder model

Install Required Libraries

    $ pip install numpy pandas matplotlib scikit-learn tensorflow

Run the Project

    $ phyton app.py

Project Workflow

    1.  Load and Clean Data
        Read CSV files
        Drop or handle missing values
    2. Normalization
        It represents compressing data into a range between 0 and 1.
        MinMaxScaler applies to each column separately-> (X[scaled] = X - X[min]) / (X[max] - X[min])
        Example:
            Raw: 20, 50, 80 → Scaled: 0, 0.5, 1
    3. Train Models
        Train LSTM autoencoder on prepared sequences
        Combine with IsolationForest and One-Class SVM for hybrid anomaly detection
    4. Provide Feedback
        Display alerts, recommendations, and visualizations

Database Setup

    Connection String (PostgreSQL on Railway):
        terminal -> PGPASSWORD=EzoBNoVJIZgMoVscYsURFdcNvwRAKjac psql -h interchange.proxy.rlwy.net -U postgres -p 57128 -d railway

    Tables:

    Enum Types:
        1. lift_types: ["squat", "bench_press", "deadlift"]
            query: CREATE TYPE lift_type AS ENUM ('squat', 'bench_press', 'deadlift');

    1. users

        CREATE EXTENSION IF NOT EXISTS "pgcrypto";

        CREATE TABLE users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            first_name VARCHAR(100) NOT NULL,
            last_name VARCHAR(100) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            role VARCHAR(50) CHECK (role IN ('trainer', 'athlete', 'admin')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

    2. feedback

        CREATE TABLE feedback (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session UUID NOT NULL REFERENCES sessions(id),
            metrics_id UUID NOT NULL REFERENCES performance_metrics(id),
            feedback_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

    3. performance_metrics

        CREATE TABLE performance_metrics (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session UUID NOT NULL REFERENCES sessions(id),
            balance_score FLOAT NOT NULL,
            stability_score FLOAT NOT NULL,
            injury_risk FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

    4. sensor_data

        CREATE TABLE sensor_data (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            athlete UUID REFERENCES users(id),
            session UUID REFERENCES sessions(id),
            raw_data JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

    5. sessions

        CREATE TABLE sessions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            trainer UUID REFERENCES users(id),
            athlete UUID REFERENCES users(id),
            lift_type lift_type NOT NULL,
            sensor_data_id UUID REFERENCES sensor_data(id),
            performance_metric_id UUID REFERENCES performance_metrics(id),
            feedback_id UUID REFERENCES feedback(id),
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(50) CHECK (status IN ('active', 'ended')) NOT NULL
        );

Structure

    /app.py → Initiator for web app
    /models → DB models
    /services → business logic
    /templates → html templates
    /models/checkpoints/lstm_autoencoder.h5 → Saved LSTM model weight

APIs:

    GET /signup -> gets sign-up page
    POST /signup → user registration
    GET /login -> gets login page
    POST /login → login user
    GET /dashboard -> gets welcome page
    POST /sessions → creates new session
    GET /sessions/<session_id> -> gets session details
    GET /sessions/<session_id>/sensor_data -> gets sensor data for session
    POST /sessions/<id>/end → end session and start analyze
    GET /profile → user profile and session history
