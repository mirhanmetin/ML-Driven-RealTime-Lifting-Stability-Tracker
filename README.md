\*\* Requirements: \*\*

Phyton and/or Phyton3 -> Phyton version should be one of tensorflow satisfied versions.

--- Should run this commands firstly: ---

    $ brew install python@3.10

--- After that, create a virtual environment on this version: ---

    $ source venv310/bin/activate
        -> Should seem like that:
        $ (venv310) yourmachine%

--- Upgrade your pip to the newest version: ---

    $ pip install --upgrade pip

--- Pre-Requirements: ---

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

--- Instal prerequirements: ---

    $ pip install numpy pandas matplotlib scikit-learn tensorflow

--- Structure

    1. Load data and clean
    2. Normalization
        It represents compressing data into a range between 0 and 1.
        MinMaxScaler applies to each column separately-> (X[scaled] = X - X[min]) / (X[max] - X[min])
        Example for left_foot_pressure:
            Values: 20
                    50
                    80
            Min: 20 Max: 80
            20 -> (20−20)/(80−20)=0
            50 -> (50−20)/(80−20)=30/60=0.5
            80 -> (80−20)/(80−20)=1
