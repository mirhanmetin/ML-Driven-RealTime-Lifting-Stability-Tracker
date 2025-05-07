from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

# LSTM Autoencoder Model
# @model = LSTM Autoencoder model for time series anomaly detection.
# @timesteps = Number of previous time steps to consider for each sample.
# @n_features = Number of features in the input data.
# @Sequential = Keras Sequential model for building the LSTM Autoencoder.
# @LSTM = Long Short-Term Memory layer for processing sequential data.
# @Dense = Fully connected layer for output.
# @RepeatVector = Layer that repeats the input for each time step in the output sequence.
# @TimeDistributed = Layer that applies a layer to each time step of the input.
# @compile = Compiles the model with Adam optimizer and Mean Squared Error loss function.
# @input_shape = Shape of the input data (timesteps, n_features).

# @param timesteps: Number of previous time steps to consider for each sample.
# @param n_features: Number of features in the input data.
# @return Compiled LSTM Autoencoder model.
def build_lstm_autoencoder(timesteps, n_features):
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(timesteps, n_features)), # 64 columns
        LSTM(32, activation='relu', return_sequences=False),
        RepeatVector(timesteps),
        LSTM(32, activation='relu', return_sequences=True),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mse')

    return model
