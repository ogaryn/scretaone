import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# Constants
TRAIN_SIZE = 0.7
WINDOW_SIZE = 30
NUM_EPOCHS = 100
NUM_NEURONS = 500
NUM_CLASSES = 2
BATCH_SIZE = 32
ESN_RHO = 0.6
ESN_SP = 0.9
ESN_SR = 1e-7
ESN_RES_SIZE = 1000

# Define the architecture of the ANN
model = Sequential()
model.add(Dense(32, input_dim=(X.shape[1], 1)))
model.add(Dense(16, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# Compile the ANN
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# Train the ANN
model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

# Evaluate the ANN
score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print("Test accuracy: ", score)

# Make predictions on test data
y_pred = model.predict(X_test)

# Convert predictions to binary format
y_pred = np.where(y_pred > 0.5, 1, 0)

import numpy as np
import pandas as pd
import talib

# Define some parameters
WINDOW = 200  # ESN input window size
N_RESERVOIR = 1000  # Number of ESN reservoir neurons
SPARSITY = 0.1  # Reservoir connection sparsity
SPECTRAL_RADIUS = 0.95  # Maximum absolute eigenvalue of reservoir matrix
ALPHA = 0.5  # Leakage rate of reservoir neurons
N_OUTPUTS = 1  # Number of ESN output neurons
REGRESSION_METHOD = 'ridge'  # Type of regression method to use
REGRESSION_PARAMETERS = {'alpha': 1e-3}  # Parameters for the regression method

# Load some data
data = pd.read_csv('data.csv')
close_prices = data['close'].values

# Compute some technical indicators
rsi = talib.RSI(close_prices, timeperiod=14)
macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)

# Create the ESN
reservoir = np.random.rand(N_RESERVOIR, N_RESERVOIR) - 0.5
reservoir *= 2 * SPECTRAL_RADIUS
mask = np.random.rand(N_RESERVOIR, N_RESERVOIR)
mask[mask < SPARSITY] = 0
reservoir *= mask
x = np.zeros((N_RESERVOIR, 1))
y = np.zeros((N_OUTPUTS, 1))
w_in = np.random.rand(N_RESERVOIR, WINDOW + 1) - 0.5
w_out = np.zeros((N_OUTPUTS, N_RESERVOIR + WINDOW + 1))

state = np.random.rand(N_RESERVOIR, 1) * 2 - 1

for i in range(len(inputs)):
    input_vec = np.vstack([inputs[i], state, np.ones((1, 1))])
    state = np.tanh(np.dot(W_in, input_vec) + np.dot(W_res, state))


output = np.dot(W_out, np.vstack([state, np.ones((1, 1))]))
W_out = np.dot(target.T, np.hstack([state, np.ones((len(inputs), 1))])) \
        .dot(np.linalg.inv(np.dot(np.hstack([state, np.ones((len(inputs), 1))]).T,
                                  np.hstack([state, np.ones((len(inputs), 1))]) + reg * np.eye(N_RESERVOIR + 1))))

# Download data for the trading pairs
data1 = download_data("BTC-USD")
data2 = download_data("ETH-USD")
data3 = download_data("LTC-USD")

def load_data(pair, start_date, end_date):
    data = pd.read_csv(f'{pair}.csv')
    data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    data = data.drop(['Open', 'High', 'Low'], axis=1)
    data.columns = ['price']
    return data

pairs = ['EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD', 'XAUUSD']

start_date = '2010-01-01'
end_date = '2022-03-28'

data = {}
for pair in pairs:
    data[pair] = load_data(pair, start_date, end_date)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Define the trading pairs
pairs = ['EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD', 'EURJPY', 'XAUUSD']

# Load the data for each trading pair
dfs = []
for pair in pairs:
    df = pd.read_csv(f"{pair}.csv")
    df = df.dropna()
    dfs.append(df)

# Combine the data for all trading pairs
df_combined = pd.concat(dfs)

# Split the data into training and testing sets
train_size = int(0.8 * len(df_combined))
train_df = df_combined[:train_size]
test_df = df_combined[train_size:]

# Normalize the data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_df)
test_data = scaler.transform(test_df)

# Split the data into input and output
train_input = train_data[:, :-1]
train_output = train_data[:, -1]
test_input = test_data[:, :-1]
test_output = test_data[:, -1]

#erge data and compute features
df = pd.concat([data1, data2, data3, data['EURUSD'], data['USDJPY'], data['GBPUSD'], data['USDCHF'], data['AUDUSD'], data['USDCAD'], data['NZDUSD'], data['XAUUSD']], axis=1)
df = df.dropna()
df["returns"] = df["price"].pct_change()

for i in range(1, 21):
    df[f"returns_lag_{i}"] = df["returns"].shift(i)
df[f"price_lag_{i}"] = df["price"].shift(i)
df = df.dropna()

df["sma_5"] = df["price"].rolling(5).mean()
df["sma_20"] = df["price"].rolling(20).mean()
df["sma_50"] = df["price"].rolling(50).mean()

df["rsi"] = talib.RSI(df["price"].values, timeperiod=14)
df["macd"], df["signal"], df["hist"] = talib.MACD(df["price"].values, fastperiod=12, slowperiod=26, signalperiod=9)

df["adx"] = talib.ADX(df["High"].values, df["Low"].values, df["Close"].values, timeperiod=14)
df["cci"] = talib.CCI(df["High"].values, df["Low"].values, df["Close"].values, timeperiod=20)
df["williams %r"] = talib.WILLR(df["High"].values, df["Low"].values, df["Close"].values, timeperiod=14)
df["aroon_up"], df["aroon_down"] = talib.AROON(df["High"].values, df["Low"].values, timeperiod=14)

df = df.dropna()

#Scale features
scaler = StandardScaler()
X = scaler.fit_transform(df.drop(["price"], axis=1).values)
y = np.where(df["returns"].values > 0, 1, -1)

#Split data into training and testing sets
train_size = int(len(X) * 0.8)
train_input_data = X[:train_size]
train_output_data = y[:train_size]
test_input_data = X[train_size:]
test_output_data = y[train_size:]

#Define the neural network architecture
model = Sequential()
model.add(Dense(32, input_dim=train_input_data.shape[1], activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

#Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

#Train the model
model.fit(train_input_data, train_output_data, epochs=100, batch_size=32, validation_split=0.2)

#Generate trade signals
predictions = model.predict(test_input_data)
predicted_signals = np.where(predictions > 0.5, 1, -1)

#Compute accuracy
accuracy = (predicted_signals == test_output_data).mean()
print("Accuracy: ", accuracy)

#Generate trades
trades = predicted_signals.copy()
trades[1:][predicted_signals[:-1] == predicted_signals[1:]] = 0

#Visualize trades
df_test = df.iloc[train_size:]
df_test["trades"] = trades

def preprocess_data(data, input_len, output_len, train_size):
    # Split data into input (X) and output (y) data
    X = []
    y = []
    for i in range(input_len, len(data) - output_len):
        X.append(data[i - input_len:i])
        y.append(data[i:i + output_len])
        X = np.array(X)
        y = np.array(y)

        # Normalize the input data
        X_mean = np.mean(X, axis=(0, 1))
        X_std = np.std(X, axis=(0, 1))
        X = (X - X_mean) / X_std
        # Split the data into training and testing sets
        train_size = int(train_size * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        return X_train, X_test, y_train, y_test, X_mean, X_std

#Set the input and output lengths, and the training size
input_len = 30
output_len = 1
train_size = 0.8

#Preprocess the data for each trading pair
preprocessed_data = {}
for pair in pairs:
    # Extract the price data
    price_data = data[pair]['price'].values

# Preprocess the data
X_train, X_test, y_train, y_test, X_mean, X_std = preprocess_data(price_data, input_len, output_len, train_size)

# Store the preprocessed data
preprocessed_data[pair] = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'X_mean': X_mean,
    'X_std': X_std,
}
# Compute features
for pair in pairs:
    data[pair]['log_return'] = np.log(data[pair]['price'] / data[pair]['price'].shift(1))
    data[pair]['rolling_mean'] = data[pair]['log_return'].rolling(window=5).mean()
    data[pair]['rolling_std'] = data[pair]['log_return'].rolling(window=5).std()

# Remove first 5 rows with missing data
for pair in pairs:
    data[pair] = data[pair][5:]

# Combine data for all pairs
combined_data = pd.DataFrame(index=data[pairs[0]].index)
for pair in pairs:
    combined_data[f'{pair}_log_return'] = data[pair]['log_return']
    combined_data[f'{pair}_rolling_mean'] = data[pair]['rolling_mean']
    combined_data[f'{pair}_rolling_std'] = data[pair]['rolling_std']

# Remove rows with missing data
combined_data.dropna(inplace=True)

# Split data into training and testing sets
train_size = int(0.8 * len(combined_data))
train_data = combined_data.iloc[:train_size]
test_data = combined_data.iloc[train_size:]

# Scale the data
scaler = StandardScaler()
train_input_data = scaler.fit_transform(train_data.iloc[:, 1:])
train_output_data = np.where(train_data.iloc[:, 0] > 0, 1, 0)
test_input_data = scaler.transform(test_data.iloc[:, 1:])
test_output_data = np.where(test_data.iloc[:, 0] > 0, 1, 0)

# Train the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(train_input_data.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_input_data, train_output_data, epochs=20, batch_size=32)

# Generate trade signals
predictions = model.predict(test_input_data)
predicted_signals = np.where(predictions > 0.5, 1, -1)

# Compute accuracy
accuracy = (predicted_signals == test_output_data).mean()
print("Accuracy: ", accuracy)

# compute technical indicators
def compute_technical_indicators(data):
    # compute EMA
    ema_short = ta.trend.EMAIndicator(close=data['price'], window=20)
    ema_long = ta.trend.EMAIndicator(close=data['price'], window=50)
    data['ema_short'] = ema_short.ema_indicator()
    data['ema_long'] = ema_long.ema_indicator()

    # compute RSI
    rsi = ta.momentum.RSIIndicator(close=data['price'], window=14)
    data['rsi'] = rsi.rsi()

    # compute MACD
    macd = ta.trend.MACD(close=data['price'], window_slow=26, window_fast=12, window_sign=9)
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()

    # compute Bollinger Bands
    bb = ta.volatility.BollingerBands(close=data['price'], window=20, window_dev=2)
    data['bb_upper'] = bb.bollinger_hband()
    data['bb_lower'] = bb.bollinger_lband()

    return data

for pair in pairs:
    data[pair] = compute_technical_indicators(data[pair])

# Split the data into training and testing sets
train_data = {}
test_data = {}
for pair in pairs:
    train_data[pair] = data[pair].iloc[:-365]
    test_data[pair] = data[pair].iloc[-365:]

# Preprocess the data
train_input_data, train_output_data = preprocess_data(train_data, window_size)
test_input_data, test_output_data = preprocess_data(test_data, window_size)

# Train the model
model = build_model(window_size, num_features)
model.fit(train_input_data, train_output_data, epochs=10, batch_size=32)

# Generate trade signals
predictions = model.predict(test_input_data)
predicted_signals = np.where(predictions > 0.5, 1, -1)

# Compute accuracy
accuracy = (predicted_signals == test_output_data).mean()
print("Accuracy: ", accuracy)

#Split data into training and testing sets
train_data = {}
test_data = {}
for pair in pairs:
    train_data[pair] = data[pair].loc['2010-01-01':'2020-12-31']
test_data[pair] = data[pair].loc['2021-01-01':'2022-03-28']

#Generate feature and label data for each trading pair
train_features = {}
train_labels = {}
test_features = {}
test_labels = {}

for pair in pairs:
    # Drop missing values
    train_data[pair] = train_data[pair].dropna()
    test_data[pair] = test_data[pair].dropna()

# Generate feature and label data for training set
train_data_shifted = train_data[pair].shift(-1)
train_features[pair] = train_data[pair][:-1].values
train_labels[pair] = np.where(train_data_shifted['price'][:-1] > train_data[pair]['price'][:-1], 1, 0)

# Generate feature and label data for testing set
test_data_shifted = test_data[pair].shift(-1)
test_features[pair] = test_data[pair][:-1].values
test_labels[pair] = np.where(test_data_shifted['price'][:-1] > test_data[pair]['price'][:-1], 1, 0)
#Scale feature data using MinMaxScaler
scaler = MinMaxScaler()

for pair in pairs:
    train_features[pair] = scaler.fit_transform(train_features[pair])
test_features[pair] = scaler.transform(test_features[pair])

#Convert feature and label data to tensors
train_input_data = {}
train_output_data = {}
test_input_data = {}
test_output_data = {}

for pair in pairs:
    train_input_data[pair] = tf.convert_to_tensor(train_features[pair], dtype=tf.float32)
train_output_data[pair] = tf.convert_to_tensor(train_labels[pair], dtype=tf.float32)

#Copy code
test_input_data[pair] = tf.convert_to_tensor(test_features[pair], dtype=tf.float32)
test_output_data[pair] = tf.convert_to_tensor(test_labels[pair], dtype=tf.float32)
#Build and train model
model = Sequential([
Dense(32, activation='relu', input_shape=(train_input_data[pairs[0]].shape[1],)),
Dense(16, activation='relu'),
Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_input_data[pairs[0]], train_output_data[pairs[0]], epochs=50, batch_size=32)

#Generate trade signals
predictions = model.predict(test_input_data[pairs[0]])
predicted_signals = np.where(predictions > 0.5, 1, -1)

#Compute accuracy
accuracy = (predicted_signals == test_output_data[pairs[0]]).mean()
print("Accuracy for {}: {}".format(pairs[0], accuracy))

#Generate trade signals and compute accuracy for all trading pairs
accuracies = []

for pair in pairs:
    predictions = model.predict(test_input_data[pair])
predicted_signals = np.where(predictions > 0.5, 1, -1)
accuracy = (predicted_signals == test_output_data[pair]).mean()
accuracies.append(accuracy)
print("Accuracy for {}: {}".format(pair, accuracy))

#Compute overall accuracy
overall_accuracy = np.mean(accuracies)
print("Overall accuracy: ", overall_accuracy)
# Modify the column names to match the input arguments of add_all_ta_features
data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)

# Call add_all_ta_features with the corrected column names
data = add_all_ta_features(data, open='open', high='high', low='low', close='close', volume=None)

# Clean and preprocess data
data = data.dropna()  # Remove any missing data
data.index = pd.to_datetime(data.index)
data = data.resample('D').last()  # Resample to daily frequency and select last observation of each day

df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])

def add_all_ta_features(df, open, high, low, close, volume):
    """
    Add all technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        open (str): Name of 'open' column.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted
        vectorized(bool): if True, use only vectorized functions indicators

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df = add_volume_ta(
        df=df,
        high=high,
        low=low,
        close=close,
        volume=volume,
        fillna=fillna,
        colprefix=colprefix,
        vectorized=vectorized,
    )
    df = add_volatility_ta(
        df=df,
        high=high,
        low=low,
        close=close,
        fillna=fillna,
        colprefix=colprefix,
        vectorized=vectorized,
    )
    df = add_trend_ta(
        df=df,
        high=high,
        low=low,
        close=close,
        fillna=fillna,
        colprefix=colprefix,
        vectorized=vectorized,
    )
    df = add_momentum_ta(
        df=df,
        high=high,
        low=low,
        close=close,
        volume=volume,
        fillna=fillna,
        colprefix=colprefix,
        vectorized=vectorized,
    )
    df = add_others_ta(
        df=df, high=high, low=low,open=open, close=close, fillna=fillna, colprefix=colprefix
    )
    return df
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('EURUSD=X_close', axis=1), data['EURUSD=X_close'],
                                                    test_size=0.2, shuffle=False)
# Read data from CSV file
data = pd.read_csv('data.csv')
df = pd.read_csv('data.csv')

print(df.columns.tolist())
# Check if 'high' column is present
df.get('high', default="high")
# Add technical indicators
def add_all_ta_features(df, open, high, low, close, volume):


# Scale and normalize data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

# Add lagged features
lags = 10
X_train_lagged = pd.DataFrame()
X_test_lagged = pd.DataFrame()

for i in range(lags):
    lagged_features_train = X_train.shift(i + 1)
    lagged_features_test = X_test.shift(i + 1)

    lagged_features_train.columns = [col + '_lag' + str(i + 1) for col in X_train.columns]
    lagged_features_test.columns = [col + '_lag' + str(i + 1) for col in X_test.columns]

    X_train_lagged = pd.concat([X_train_lagged, lagged_features_train], axis=1)
    X_test_lagged = pd.concat([X_test_lagged, lagged_features_test], axis=1)

# Drop rows with missing data due to lagging
X_train_lagged = X_train_lagged.iloc[lags:]
X_test_lagged = X_test_lagged.iloc[lags:]

# Reshape the data for LSTM input
X_train_lagged = X_train_lagged.values.reshape((X_train_lagged.shape[0], lags, X_train.shape[1]))
X_test_lagged = X_test_lagged.values.reshape((X_test_lagged.shape[0], lags, X_test.shape[1]))

# Define the model
num_neurons = 50
model = Sequential()
model.add(LSTM(num_neurons, input_shape=(lags, X_train.shape[1])))
model.add(Dense(1))

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Fit the model
batch_size = 32
epochs = 100

model = Sequential()
model.add(LSTM(num_neurons, input_shape=(lags, num_features)))
model.add(Dense(1, activation='linear'))

#Train the model
history = model.fit(X_train_lagged.values.reshape((-1, lags, num_features)), y_train[lags:], validation_data=(X_test_lagged.values.reshape((-1, lags, num_features)), y_test[lags:]), batch_size=batch_size, epochs=epochs)

#Evaluate the model
train_loss = model.evaluate(X_train_lagged.values.reshape((-1, lags, num_features)), y_train[lags:])
test_loss = model.evaluate(X_test_lagged.values.reshape((-1, lags, num_features)), y_test[lags:])
print(f"Train loss: {train_loss:.4f}")
print(f"Test loss: {test_loss:.4f}")

#Make predictions on test data
y_pred = model.predict(X_test_lagged.values.reshape((-1, lags, num_features)))

#Inverse transform predictions and actual values to original scale
y_pred = scaler.inverse_transform(y_pred)
y_test_lagged_inv = scaler.inverse_transform(y_test[lags:].values.reshape((-1, 1)))

#Plot predicted vs actual values
import matplotlib.pyplot as plt
plt.plot(y_test_lagged_inv, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()

#Save the model
model.save('my_model.h5')

#Use the model
#Define the trading environment


import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

class EchoStateNetwork:
    def __init__(self, n_input, n_reservoir, n_output, spectral_radius=1.0, sparsity=0.0):
        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.n_output = n_output
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_weights = None
        self.reservoir_weights = None
        self.output_weights = None

    def train_esn(self, x_train, y_train):
        # Train the ANN to get the input weights
        ann = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, tol=1e-6)
        ann.fit(x_train, y_train)
        self.input_weights = ann.coefs_[0].T

        # Generate random sparse weights for the reservoir
        self.reservoir_weights = csr_matrix((np.random.rand(self.n_reservoir, self.n_reservoir) < self.sparsity).astype(float))
        self.reservoir_weights.data -= 0.5

        # Scale the spectral radius of the reservoir weights
        max_eigenvalue = np.max(np.abs(np.linalg.eigvals(self.reservoir_weights.todense())))
        self.reservoir_weights *= self.spectral_radius / max_eigenvalue

        # Compute the activations of the reservoir nodes
        x_reservoir = np.zeros((x_train.shape[0], self.n_reservoir))
        for i in range(x_train.shape[0]):
            if i == 0:
                x_reservoir[i] = np.tanh(self.input_weights @ x_train[i])
            else:
                x_reservoir[i] = np.tanh(self.input_weights @ x_train[i] + self.reservoir_weights @ x_reservoir[i-1])

        # Train the output weights using ridge regression
        self.output_weights = Ridge(alpha=1e-6).fit(x_reservoir, y_train).coef_

    def predict(self, x_test):
        # Compute the activations of the reservoir nodes for the test data
        x_reservoir = np.zeros((x_test.shape[0], self.n_reservoir))
        for i in range(x_test.shape[0]):
            if i == 0:
                x_reservoir[i] = np.tanh(self.input_weights @ x_test[i])
            else:
                x_reservoir[i] = np.tanh(self.input_weights @ x_test[i] + self.reservoir_weights @ x_reservoir[i-1])

        # Predict the output using the trained output weights
        y_pred = x_reservoir @ self.output_weights

        return y_pred

def _request(self, method, url, headers=None, **kwargs):
        """
        Send a request to the OANDA API.

        Args:
        - method (str): The HTTP method to use for the request (e.g. "GET", "POST", "PUT", "DELETE").
        - url (str): The URL to send the request to.
        - headers (dict): Additional headers to include in the request.
        - **kwargs: Additional arguments to pass to the requests library (e.g. "params", "data", "json", etc.).

        Returns:
        - dict: The response from the API, parsed as JSON.

        Raises:
        - V20Error: If the request fails or the response indicates an error.
        """
        # Add any custom headers to the request
        if headers is None:
            headers = {}
        headers.update(self.client.headers)

        # Send the request to the API
        response = self.client.request(method, url, headers=headers, **kwargs)

        # Parse the response
        if response.status_code == requests.codes.ok:
            return response.json()
        elif response.status_code == requests.codes.unauthorized:
            raise V20Error("Authentication error: access token is invalid or expired.")
        else:
            try:
                response_body = response.json()
            except ValueError:
                response_body = response.content
            error_msg = response_body["errorMessage"] if "errorMessage" in response_body else "Unknown error"
            raise V20Error(f"Error {response.status_code}: {error_msg}")

def request(self, method, endpoint, params=None, data=None):
        """
        Send a request to the OANDA API.

        Args:
        - method (str): The HTTP method to use for the request (e.g. "GET", "POST", "PUT", "DELETE").
        - endpoint (str): The endpoint to send the request to.
        - params (dict): Query parameters to include in the request.
        - data (dict): Data to include in the request body.

        Returns:
        - dict: The response from the API, parsed as JSON.

        Raises:
        - V20Error: If the request fails or the response indicates an error.
        """
        url = self.environment + endpoint

        # Add any query parameters to the URL
        if params is not None:
            url += "?" + urllib.parse.urlencode(params)

        # Send the request to the API
        if data is not None:
            response = self._request(method, url, json=data)
        else:
            response = self._request(method, url)

        return response
TRADING_ENVIRONMENTS = {
    "live": {
        "hostname": "api-fxtrade.oanda.com",
        "streaming_hostname": "stream-fxtrade.oanda.com",
        "ssl": True,
        "port": 443,
        "streaming_port": 443,
        "streaming_ssl": True
    },
    "practice": {
        "hostname": "api-fxpractice.oanda.com",
        "streaming_hostname": "stream-fxpractice.oanda.com",
        "ssl": True,
        "port": 443,
        "streaming_port": 443,
        "streaming_ssl": True
    },
    "sandbox": {
        "hostname": "api-fxtrade-sandbox.oanda.com",
        "streaming_hostname": "stream-fxtrade-sandbox.oanda.com",
        "ssl": True,
        "port": 443,
        "streaming_port": 443,
        "streaming_ssl": True
    },
    "my_environment": {
        "hostname": "api-my-environment.oanda.com",
        "streaming_hostname": "stream-my-environment.oanda.com",
        "ssl": True,
        "port": 443,
        "streaming_port": 443,
        "streaming_ssl": True
    }
}


def get_instruments(self, account_id):
        """
        Get a list of tradeable instruments for a given account.

        Args:
        - account_id (str): The ID of the account to get the instruments for.

        Returns:
        - list of dict: The list of tradeable instruments, with each instrument represented as a dict.
        """
        endpoint = f"/v3/accounts/{account_id}/instruments"
        response = self.request("GET", endpoint)
        return response["instruments"]

def get_candles(self, instrument, granularity, count=None, from_time=None, to_time=None):
    # Get historical candlestick data for an instrument.

        # set up streaming
   class OandaClient:
    def __init__(self, access_token, account_id, environment="practice"):
        # Set up client object
        self.client = oandapyV20.API(access_token=access_token, environment=environment)

        # Set up account ID
        self.account_id = account_id

        # Set up request count
        self._request_count = 0

    def request(self, method, endpoint, params=None, body=None, headers=None):
        # API request method

        url = f"{self.client.api_url}{endpoint}"


class LSTMModel:
    def __init__(self, num_features: int, num_units: int = 50, dropout_rate: float = 0.2, learning_rate: float = 0.001):
        self.model = Sequential()
        self.model.add(LSTM(units=num_units, input_shape=(None, num_features)))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(units=1))
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def fit(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, epochs: int = 100, validation_split: float = 0.2):
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class EchoStateNetwork:
    def __init__(self, n_reservoir: int, spectral_radius: float, sparsity: float, random_state: int):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.random_state = random_state
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def _initialize_reservoir(self, X: np.ndarray) -> np.ndarray:
        np.random.seed(self.random_state)
        n_features = X.shape[1]
        self.W_in = np.random.rand(self.n_reservoir, n_features) - 0.5
        self.W_res = np.random.rand(self.n_reservoir, self.n_reservoir) - 0.5
        self.W_res[self.W_res > self.sparsity] = 0
        self.W_res *= self.spectral_radius / np.max(np.abs(np.linalg.eigvals(self.W_res)))
        self.W_out = None

    def _transform_input(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(np.dot(X, self.W_in.T))

    def _iterate_reservoir(self, X: np.ndarray, initial_state: np.ndarray) -> np.ndarray:
        state = initial_state
        states = [state]
        for t in range(1, X.shape[0]):
            state = np.tanh(np.dot(state, self.W_res.T) + np.dot(X[t], self.W_in.T))
            states.append(state)
        return np.array(states)

    def fit(self, X: np.ndarray, y: np.ndarray, train_fraction: float = 0.8, initial_state_scale: float = 0.01):
        train_size = int(train_fraction * X.shape[0])
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        self._initialize_reservoir(X_train)
        initial_state = np.random.rand(self.n_reservoir) * initial_state_scale

        X_train_res = self._iterate_reservoir(self._transform_input(X_train), initial_state)
        self.scaler.fit(X_train_res)
        X_train_res_scaled = self.scaler.transform(X_train_res)
        self.W_out = np.dot(np.linalg.pinv(X_train_res_scaled), y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_res = self._iterate_reservoir(self._transform_input(X), np.random.rand(self.n_reservoir))
        X_res_scaled = self.scaler.transform(X_res)
        return np.dot(X_res_scaled, self.W_out)


def prepare_data(data: pd.DataFrame, lookback: int, test_size: float) -> Tuple[np.ndarray, np.ndarray]:
    add_all_ta_features(data, open='open', high='high', low='low', close='close', volume='volume', fillna=True)
    data = data.dropna()
    data = data.resample('D').last()

    features = data.drop('close', axis=1).columns
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data[features] = scaler.fit_transform(data[features])


def _request(self, method, url, headers=None, **kwargs):
        """
        Send a request to the OANDA API.

        Args:
        - method (str): The HTTP method to use for the request (e.g. "GET", "POST", "PUT", "DELETE").
        - url (str): The URL to send the request to.
        - headers (dict): Additional headers to include in the request.
        - **kwargs: Additional arguments to pass to the requests library (e.g. "params", "data", "json", etc.).

        Returns:
        - dict: The response from the API, parsed as JSON.

        Raises:
        - V20Error: If the request fails or the response indicates an error.
        """
        # Add any custom headers to the request
        if headers is None:
            headers = {}
        headers.update(self.client.headers)

        # Send the request to the API
        response = self.client.request(method, url, headers=headers, **kwargs)

        # Parse the response
        if response.status_code == requests.codes.ok:
            return response.json()
        elif response.status_code == requests.codes.unauthorized:
            raise V20Error("Authentication error: access token is invalid or expired.")
        else:
            try:
                response_body = response.json()
            except ValueError:
                response_body = response.content
            error_msg = response_body["errorMessage"] if "errorMessage" in response_body else "Unknown error"
            raise V20Error(f"Error {response.status_code}: {error_msg}")

def request(self, method, endpoint, params=None, data=None):
        """
        Send a request to the OANDA API.

        Args:
        - method (str): The HTTP method to use for the request (e.g. "GET", "POST", "PUT", "DELETE").
        - endpoint (str): The endpoint to send the request to.
        - params (dict): Query parameters to include in the request.
        - data (dict): Data to include in the request body.

        Returns:
        - dict: The response from the API, parsed as JSON.

        Raises:
        - V20Error: If the request fails or the response indicates an error.
        """
        url = self.environment + endpoint

        # Add any query parameters to the URL
        if params is not None:
            url += "?" + urllib.parse.urlencode(params)

        # Send the request to the API
        if data is not None:
            response = self._request(method, url, json=data)
        else:
            response = self._request(method, url)

        return response
TRADING_ENVIRONMENTS = {
    "live": {
        "hostname": "api-fxtrade.oanda.com",
        "streaming_hostname": "stream-fxtrade.oanda.com",
        "ssl": True,
        "port": 443,
        "streaming_port": 443,
        "streaming_ssl": True
    },
    "practice": {
        "hostname": "api-fxpractice.oanda.com",
        "streaming_hostname": "stream-fxpractice.oanda.com",
        "ssl": True,
        "port": 443,
        "streaming_port": 443,
        "streaming_ssl": True
    },
    "sandbox": {
        "hostname": "api-fxtrade-sandbox.oanda.com",
        "streaming_hostname": "stream-fxtrade-sandbox.oanda.com",
        "ssl": True,
        "port": 443,
        "streaming_port": 443,
        "streaming_ssl": True
    },
    "my_environment": {
        "hostname": "api-my-environment.oanda.com",
        "streaming_hostname": "stream-my-environment.oanda.com",
        "ssl": True,
        "port": 443,
        "streaming_port": 443,
        "streaming_ssl": True
    }
}


def get_instruments(self, account_id):
        """
        Get a list of tradeable instruments for a given account.

        Args:
        - account_id (str): The ID of the account to get the instruments for.

        Returns:
        - list of dict: The list of tradeable instruments, with each instrument represented as a dict.
        """
        endpoint = f"/v3/accounts/{account_id}/instruments"
        response = self.request("GET", endpoint)
        return response["instruments"]

def get_candles(self, instrument, granularity, count=None, from_time=None, to_time=None):
    # Get historical candlestick data for an instrument.

        # set up streaming
   class OandaClient:
    def __init__(self, access_token, account_id, environment="practice"):
        # Set up client object
        self.client = oandapyV20.API(access_token=access_token, environment=environment)

        # Set up account ID
        self.account_id = account_id

        # Set up request count
        self._request_count = 0

    def request(self, method, endpoint, params=None, body=None, headers=None):
        # API request method

        url = f"{self.client.api_url}{endpoint}"

        # Set up headers
        if headers is None:
            headers = {}

        # Set up query parameters
        if params is None:
            params = {}

        # Set up request body
        if body is not None:
            body = json.dumps(body)

        # Make request
        response = self.client.request(method, url, params=params, data=body, headers=headers)

        # Increment request count
        self._request_count += 1

        # Check for errors
        if response.status_code >= 400:
            raise V20Error(response)

        # Parse response
        return response.json()

        headers = {}
        headers.update({"Content-Type": "application/json"})

        # Set up query parameters
        if params is not None:
            url += "?" + urllib.parse.urlencode(params)

        # Set up request body
        if body is not None:
            if isinstance(body, str):
                data = body
            else:
                data = json.dumps(body)
        else:
            data = None

        # Send the request
        response = self.client.request(method, url, headers=headers, data=data)

        # Check for errors
        if response.status_code >= 400:
            raise ConnectionError(f"Request failed with status {response.status_code}:\n{response.text}")

        return response.json()

        # update request count
        self._request_count += 1

        # set up request parameters
        request_params = self.request_params.copy()

        # update request parameters with headers
        if headers is not None:
            request_params.update(headers)

        # set up request body
        if isinstance(body, dict):
            body = json.dumps(body)

        # set up request URL
        url = self._get_url(endpoint)

        # make request
        try:
            response = self.client.request(method, url, params=params, data=body, **request_params)
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            if response.status_code == 404:
                logger.error("HTTP Error: 404 Not Found for endpoint %s", endpoint)
                raise ValueError("HTTP Error: 404 Not Found for endpoint {}".format(endpoint))
            elif response.status_code == 429:
                logger.error("HTTP Error: 429 Too Many Requests for endpoint %s", endpoint)
                raise ConnectionError("HTTP Error: 429 Too Many Requests for endpoint {}".format(endpoint))
            else:
                logger.error("HTTP Error: %s for endpoint %s", response.status_code, endpoint)
                raise ConnectionError("HTTP Error: {} for endpoint {}".format(response.status_code, endpoint))
        except requests.exceptions.ConnectionError as err:
            logger.error("Connection Error for endpoint %s", endpoint)
            raise ConnectionError("Connection Error for endpoint {}".format(endpoint))

        # log response
        logger.debug("API Response: %s", response.content.decode())

        # return response
        return response.json()

def build_model(sequence_length, features):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, features)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Define function to predict price using LSTM  model
def predict_price(model, data):
    predictions = model.predict(data)
    return predictions

# Define function to extract text from image
def extract_text(image_path):
    text = pytesseract.image_to_string(image_path)
    return text
def api (api,environment):

#api = '032a5480c5d3d98801eeb60a4cb20ca-dca0f9afff4d3355d42fc29d6a4084c3',environment='fxTrade Practice
# Define list of instruments to trade
   instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD']

# Define dictionary to store candle data for each instrument
candle_data = {}

# Define granularity of candles to retrieve
granularity = 'M15'

# Define number of candles to retrieve
count = 500

# Define list of instruments to retrieve data for
instruments = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'USD_CHF', 'AUD_USD']




# Define sequence length for LSTM
sequence_length = 60

training_data = {}
test_data = {}
for instrument in instruments:
    # Select data for current instrument
    instrument_data = data[data['instrument'] == instrument].copy()

    # Normalize the data
    scaler = MinMaxScaler()
    instrument_data[['o', 'h', 'l', 'c', 'v']] = scaler.fit_transform(instrument_data[['o', 'h', 'l', 'c', 'v']])

    # Split data into training and test sets
    X = instrument_data[['o', 'h', 'l', 'v']]
    y = instrument_data['c']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Add data to dictionaries
    training_data[instrument] = {'X': X_train, 'y': y_train}
    test_data[instrument] = {'X': X_test, 'y': y_test}

#Section 3: Build and train the Artificial Neural Network
def build_model(input_shape):
# Define the model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# Train the model on each instrument
models = {}
for instrument in instruments:
    # Build the model
    input_shape = (4,)
    model = build_model(input_shape)

    # Train the model
    X_train = training_data[instrument]['X']
    y_train = training_data[instrument]['y']
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    # Add the model to the dictionary
    models[instrument] = model

#Section 4: Implement the Echo State Network
class EchoStateNetwork:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius):
        # Initialize parameters
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius

        # Generate the reservoir matrix
        np.random.seed(42)
        self.W_reservoir = np.random.rand(reservoir_size, reservoir_size) - 0.5
        self.W_reservoir *= 1.0 / spectral_radius
        self.X = np.zeros((1, reservoir_size))

        # Generate the input-to-reservoir weights
    self.W_input = np.random.rand(reservoir_size, input_size) - 0.5
    self.W_input *= 1.0 / np.sqrt(input_size)
    
    # Generate the reservoir-to-output weights
    self.W_output = np.random.rand(output_size, reservoir_size) - 0.5

def fit(self, X_train, y_train, X_test, y_test):
    # Compute the reservoir states for the training data
    X_train_reservoir = np.zeros((X_train.shape[0], self.reservoir_size))
    for i in range(X_train.shape[0]):
        u = X_train[i].reshape((1, self.input_size))
        x = self.X
        for j in range(100):
            x = np.tanh(np.dot(u, self.W_input.T) + np.dot(x, self.W_reservoir))
        X_train_reservoir[i]
        
# Create and train the ANN model for each instrument
ann_models = {}
for instrument in instruments:
    # Define the model
    model = Sequential()
    model.add(Dense(units=50, activation='relu', input_dim=NUM_FEATURES))
    model.add(Dense(units=25, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train[instrument], y_train[instrument], 
                        validation_data=(X_test[instrument], y_test[instrument]),
                        epochs=50, batch_size=32, verbose=0)

    # Save the trained model
    ann_models[instrument] = model
    
    # Build the model
    model = Sequential()

    # Add LSTM layers
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    # Add output layer
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)

print(f'Training loss: {train_loss}')
print(f'Test loss: {test_loss}')

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)

print(f'Training loss: {train_loss}')
print(f'Test loss: {test_loss}')

# Evaluate the model on the test set
test_loss = model.evaluate(x_test, y_test, verbose=2)

# Print the test loss
print(f'Test loss: {test_loss}')

# Make predictions on the test set
y_pred = model.predict(x_test)

# Convert the predictions back to their original scale
y_pred_orig = scaler.inverse_transform(y_pred)
y_test_orig = scaler.inverse_transform(y_test)

# Calculate the root mean squared error (RMSE) of the predictions
rmse = np.sqrt(np.mean(np.square(y_pred_orig - y_test_orig)))
print(f'RMSE: {rmse}')

# Plot the predictions vs actual values for the test set
plt.figure(figsize=(12,6))
plt.plot(y_test_orig, label='Actual')
plt.plot(y_pred_orig, label='Predicted')
plt.legend()
plt.title(f'{instrument} Closing Price Prediction')
plt.show()

# Visualize the predictions
plt.figure(figsize=(10,6))
plt.plot(test_y, color='blue', label='Actual price')
plt.plot(predicted_stock_price, color='red', label='Predicted price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Calculate and print the root mean squared error (RMSE)
rmse = math.sqrt(mean_squared_error(test_y, predicted_stock_price))
print(f'Root Mean Squared Error: {rmse}')

import numpy as np

class ESN:
    def __init__(self, n_inputs, n_outputs, n_reservoir, spectral_radius=0.99, sparsity=0.5, noise=0.001):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        
        # Initialize the ESN weights
        self.W_in = np.random.uniform(-0.5, 0.5, size=(n_reservoir, n_inputs))
        self.W = np.random.rand(n_reservoir, n_reservoir) - 0.5
        self.W[self.W < -sparsity] = 0
        self.W[self.W > sparsity] = 0
        self.W *= spectral_radius / np.max(np.abs(np.linalg.eigvals(self.W)))
        self.W += np.random.randn(n_reservoir, n_reservoir) * noise
        
        self.W_out = None


        
    def train(self, X, Y, washout=100, ridge_alpha=1e-6):
        # Add bias to the input
        global X_states
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        
        # Initialize the reservoir state
        state = np.zeros((self.n_reservoir, 1))
        
        # Run the input through the reservoir to generate the states
        for t in range(X.shape[0]):
            state = np.tanh(np.dot(self.W_in, X[t]) + np.dot(self.W, state))
            
            if t > washout:
                if t == washout + 1:
                    X_states = state.T
                else:
                    X_states = np.vstack((X_states, state.T))
        
        # Compute the output weights using Ridge regression
        self.W_out = np.dot(np.linalg.inv(np.dot(X_states.T, X_states) + ridge_alpha * np.eye(self.n_reservoir)),
                            np.dot(X_states.T, Y[washout:]))
        
    def predict(self, X):
        # Add bias to the input
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        
        # Initialize the reservoir state
        state = np.zeros((self.n_reservoir, 1))
        
        # Run the input through the reservoir to generate the states
        for t in range(X.shape[0]):
            state = np.tanh(np.dot(self.W_in, X[t]) + np.dot(self.W, state))
            
        # Compute the output using the trained output weights
        Y_pred = np.dot(state.T, self.W_out)
        
        return Y_pred
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
from scipy.linalg import pinv2
import random

# Define ESN class
class EchoStateNetwork:
    def __init__(self, n_input, n_reservoir, n_output, spectral_radius=1.0,
                 sparsity=0.0, random_seed=None):
        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.n_output = n_output
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.random_seed = random_seed

        # Initialize weights
        self.Win = (np.random.rand(n_reservoir, n_input + 1) - 0.5) * 2.0
        self.W = np.random.rand(n_reservoir, n_reservoir) - 0.5
        self.W[self.W > sparsity] = 0

        # Scale spectral radius of W
        rhoW = np.max(np.abs(np.linalg.eigvals(self.W)))
        self.W *= spectral_radius / rhoW

        self.Wout = np.zeros((n_output, n_reservoir + n_input + 1))

    def fit(self, X, y, n_train):
        # Initialize state
        self.x = np.zeros((self.n_reservoir, 1))

        # Allocate memory for reservoir states
        Xr = np.zeros((n_train, self.n_reservoir + self.n_input + 1))

        # Run reservoir
        for t in range(n_train):
            u = np.hstack([X[t], [1.0]])
            self.x = np.tanh(np.dot(self.Win, u) + np.dot(self.W, self.x))
            Xr[t] = np.hstack([self.x.flatten(), u])

        # Train output layer
        Xr_pinv = pinv2(Xr)
        self.Wout = np.dot(y[:n_train].T, Xr_pinv).T

        # Predict using trained ESN
        y_pred = np.zeros(y.shape)
        for t in range(n_train, len(X)):
            u = np.hstack([X[t], [1.0]])
            self.x = np.tanh(np.dot(self.Win, u) + np.dot(self.W, self.x))
            y_pred[t] = np.dot(self.Wout, np.hstack([self.x.flatten(), u]))

        return y_pred

# Set random seed for reproducibility
np.random.seed(42)

# Load data
data = pd.read_csv('data.csv')
data = data.dropna()

# Preprocess data
scaler = MinMaxScaler()
data['price'] = scaler.fit_transform(data[['price']])
data['volume'] = scaler.fit_transform(data[['volume']])

# Set input and output dimensions
n_input = 4
n_output = 1

# Set reservoir size and spectral radius
n_reservoir = 100
spectral_radius = 0.9

# Create ESN and LSTM-based ANN
esn = EchoStateNetwork(n_input, n_reservoir, n_output, spectral_radius=spectral_radius)
ann = Sequential()
ann.add(LSTM(units=50, return_sequences=True, input_shape=(n_input, 1)))
ann.add(Dropout(0.2))
ann.add(LSTM(units=50, return_sequences=True))
ann.add(Dropout(0.2))
ann.add(LSTM(units=50, return_sequences=False))
ann.add(Dropout(0.2))
ann.add(Dense(units=1))

# Compile the ANN
ann.compile(optimizer='adam', loss='mean_squared_error')

# Create training and test data sets
X = []
y = []
for i in range(n_input, len(data)):
    X.append(data.iloc[i-n_input:i, 1:].values)
    y.append(data.iloc[i, 1])
X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train ESN
y_pred_esn = esn.fit(X_train[:, :, 0], y_train, len(X_train))

# Train ANN
ann.fit(X_train, y_train, epochs=100, batch_size=32)

# Predict using trained models
y_pred_esn = scaler.inverse_transform(y_pred_esn.reshape(-1, 1))
y_pred_ann = scaler.inverse_transform(ann.predict(X_test))

# Compute test error
test_error_esn = mean_squared_error(y_test, y_pred_esn)
test_error_ann = mean_squared_error(y_test, y_pred_ann)
print(f'Test error (ESN): {test_error_esn:.6f}')
print(f'Test error (ANN): {test_error_ann:.6f}')

# Set random seed for reproducibility
np.random.seed(42)

# Load data
data = pd.read_csv('data.csv')
data = data.dropna()

# Preprocess data
scaler = MinMaxScaler()
data['price'] = scaler.fit_transform(data[['price']])
data['volume'] = scaler.fit_transform(data[['volume']])

# Set input and output dimensions
n_input = 4
n_output = 1

# Set reservoir size and spectral radius
n_reservoir = 100
spectral_radius = 0.9

# Create ESN and LSTM-based ANN
esn = EchoStateNetwork(n_input, n_reservoir, n_output, spectral_radius=spectral_radius)

ann = Sequential()
ann.add(LSTM(units=50, return_sequences=True, input_shape=(n_input, 1)))
ann.add(Dropout(0.2))
ann.add(LSTM(units=50, return_sequences=True))
ann.add(Dropout(0.2))
ann.add(LSTM(units=50, return_sequences=True))
ann.add(Dropout(0.2))
ann.add(LSTM(units=50))
ann.add(Dropout(0.2))
ann.add(Dense(units=n_output))

# Compile model
ann.compile(optimizer='adam', loss='mean_squared_error')

# Create training and test data sets
X = data[['price', 'volume', 'bid', 'ask']].values
y = data['price'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Fit ESN and predict training and test sets
y_train_esn = esn.fit(X_train, y_train, n_train=len(X_train))
y_test_esn = esn.fit(X_test, y_test, n_train=len(X_test))

# Reshape input data for LSTM
X_train_lstm = X_train.reshape((X_train.shape[0], n_input, 1))
X_test_lstm = X_test.reshape((X_test.shape[0], n_input, 1))

# Fit LSTM model and predict training and test sets
ann.fit(X_train_lstm, y_train, epochs=100, batch_size=32)
y_train_ann = ann.predict(X_train_lstm)
y_test_ann = ann.predict(X_test_lstm)

# Inverse transform predicted data
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)
y_train_esn = scaler.inverse_transform(y_train_esn)
y_test_esn = scaler.inverse_transform(y_test_esn)
y_train_ann = scaler.inverse_transform(y_train_ann)
y_test_ann = scaler.inverse_transform(y_test_ann)

# Calculate RMSE for each model
train_rmse_esn = np.sqrt(mean_squared_error(y_train, y_train_esn))
test_rmse_esn = np.sqrt(mean_squared_error(y_test, y_test_esn))
train_rmse_ann = np.sqrt(mean_squared_error(y_train, y_train_ann))
test_rmse_ann = np.sqrt(mean_squared_error(y_test, y_test_ann))

print(f'Train RMSE ESN: {train_rmse_esn:.2f}')
print(f'Test RMSE ESN: {test_rmse_esn:.2f}')
print(f'Train RMSE ANN: {train_rmse_ann:.2f}')
print(f'Test RMSE ANN: {test_rmse_ann:.2f}')

ann.add(Dense(units=1))
ann.compile(optimizer='adam', loss='mean_squared_error')
# Train the LSTM-based ANN
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
ann.fit(X_train_lstm, y_train, epochs=50, batch_size=32)

# Make predictions on test data
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
y_pred_lstm = ann.predict(X_test_lstm)

# Calculate MSE and RMSE of LSTM-based ANN predictions
mse_lstm = mean_squared_error(y_test, y_pred_lstm)
rmse_lstm = np.sqrt(mse_lstm)
print(f'LSTM-based ANN test MSE: {mse_lstm:.6f}')
print(f'LSTM-based ANN test RMSE: {rmse_lstm:.6f}')

ann.compile(optimizer='adam', loss='mean_squared_error')
history = ann.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Predict using ESN
y_pred_esn = esn.fit(X_test, y_test, n_train=len(X_train))

# Predict using LSTM-based ANN
y_pred_ann = ann.predict(X_test)
y_pred_ann = scaler.inverse_transform(y_pred_ann.reshape(-1, 1))

# Calculate RMSE for each model
rmse_esn = np.sqrt(mean_squared_error(y_test, y_pred_esn))
rmse_ann = np.sqrt(mean_squared_error(y_test, y_pred_ann))
print(f'RMSE (ESN): {rmse_esn:.5f}')
print(f'RMSE (LSTM-based ANN): {rmse_ann:.5f}')

# Compile model
ann.compile(optimizer='adam', loss='mean_squared_error')

# Train model
history = ann.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Evaluate model
train_loss = ann.evaluate(X_train, y_train)
test_loss = ann.evaluate(X_test, y_test)

# Make predictions
y_train_pred = ann.predict(X_train)
y_test_pred = ann.predict(X_test)

import matplotlib.pyplot as plt

# Plot loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
from scipy.linalg import pinv2
import random

# Define ESN class
class EchoStateNetwork:
    def __init__(self, n_input, n_reservoir, n_output, spectral_radius=1.0,
                 sparsity=0.0, random_seed=None):
        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.n_output = n_output
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.random_seed = random_seed

        # Initialize weights
        self.Win = (np.random.rand(n_reservoir, n_input + 1) - 0.5) * 2.0
        self.W = np.random.rand(n_reservoir, n_reservoir) - 0.5
        self.W[self.W > sparsity] = 0

        # Scale spectral radius of W
        rhoW = np.max(np.abs(np.linalg.eigvals(self.W)))
        self.W *= spectral_radius / rhoW

        self.Wout = np.zeros((n_output, n_reservoir + n_input + 1))

    def fit(self, X, y, n_train):
        # Initialize state
        self.x = np.zeros((self.n_reservoir, 1))

        # Allocate memory for reservoir states
        Xr = np.zeros((n_train, self.n_reservoir + self.n_input + 1))

        # Run reservoir
        for t in range(n_train):
            u = np.hstack([X[t], [1.0]])
            self.x = np.tanh(np.dot(self.Win, u) + np.dot(self.W, self.x))
            Xr[t] = np.hstack([self.x.flatten(), u])

        # Train output layer
        Xr_pinv = pinv2(Xr)
        self.Wout = np.dot(y[:n_train].T, Xr_pinv).T

        # Predict using trained ESN
        y_pred = np.zeros(y.shape)
        for t in range(n_train, len(X)):
            u = np.hstack([X[t], [1.0]])
            self.x = np.tanh(np.dot(self.Win, u) + np.dot(self.W, self.x))
            y_pred[t] = np.dot(self.Wout, np.hstack([self.x.flatten(), u]))

        return y_pred

# Set random seed for reproducibility
np.random.seed(42)

# Load data
data = pd.read_csv('data.csv')
data = data.dropna()

# Preprocess data
scaler = MinMaxScaler()
data['price'] = scaler.fit_transform(data[['price']])
data['volume'] = scaler.fit_transform(data[['volume']])

# Set input and output dimensions
n_input = 4
n_output = 1

# Set reservoir size and spectral radius
n_reservoir = 100
spectral_radius = 0.9

# Create ESN and LSTM-based ANN
esn = EchoStateNetwork(n_input, n_reservoir, n_output, spectral_radius=spectral_radius)
ann = Sequential()
ann.add(LSTM(units=50, return_sequences=True, input_shape=(n_input, 1)))
ann.add(Dropout(0.2))
ann.add(LSTM(units=50, return_sequences=True))
ann.add(Dropout(0.2))
ann.add(Dense(units=n_output))
ann.compile

# Make predictions on test data
test_predictions = model.predict(X_test)

# Convert predictions back to original scale
test_predictions = scaler.inverse_transform(test_predictions)

# Plot predictions vs actual values
plt.plot(y_test, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.title('Test Data Predictions')
plt.ylabel('Price')
plt.xlabel('Time')
plt.legend()
plt.show()

# Get last n_input days of data
last_n_days = data.tail(n_input)

# Reshape data for input into model
input_data = last_n_days.values.reshape((1, n_input, 1))

# Make prediction using LSTM model
predicted_price = model.predict(input_data)[0][0]

# Inverse transform predicted price to get original scale
predicted_price = scaler.inverse_transform([[predicted_price]])[0][0]

# Print predicted price
print(f"Predicted closing price: {predicted_price:.2f}")

# Load data
data = pd.read_csv('data.csv')
data = data.dropna()

# Preprocess data
scaler = MinMaxScaler()
data['price'] = scaler.fit_transform(data[['price']])
data['volume'] = scaler.fit_transform(data[['volume']])

# Set input and output dimensions
n_input = 4
n_output = 1

# Split data into input and output variables
X = []
y = []
for i in range(n_input, len(data)):
    X.append(data.iloc[i-n_input:i].values)
    y.append(data.iloc[i]['price'])
X = np.array(X)
y = np.array(y)

# Create LSTM-based ANN
ann = Sequential()
ann.add(LSTM(units=50, return_sequences=True, input_shape=(n_input, 1)))
ann.add(Dropout(0.2))
ann.add(LSTM(units=50, return_sequences=True))
ann.add(Dropout(0.2))
ann.add(Dense(units=n_output))

# Compile model
ann.compile(optimizer='adam', loss='mean_squared_error')

# Train model
history = ann.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, shuffle=False)

# Plot loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Retrain model on entire dataset
history = ann.fit(X, y, epochs=100, batch_size=32, shuffle=False)

# Make predictions
y_pred = ann.predict(X)

# Inverse transform predictions
y_pred = scaler.inverse_transform(y_pred)

# Plot predictions
plt.plot(data['price'].values, label='actual')
plt.plot(np.arange(n_input, len(y_pred)+n_input), y_pred, label='predicted')
plt.title('Stock price prediction')
plt.ylabel('Price')
plt.xlabel('Time')
plt.legend()
plt.show()

# Plot predicted vs actual prices
plt.figure(figsize=(10, 6))
plt.plot(test_y, label='Actual')
plt.plot(test_predict, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Evaluate model performance
mse = mean_squared_error(test_y, test_predict)
mae = mean_absolute_error(test_y, test_predict)
print('MSE:', mse)
print('MAE:', mae)

# Define the stop loss and take profit thresholds
stop_loss = 0.02  # Stop loss of 2%
take_profit = 0.05  # Take profit of 5%

# Calculate the potential stop loss and take profit prices
stop_loss_price = test_y[0] * (1 - stop_loss)
take_profit_price = test_y[0] * (1 + take_profit)

# Initialize the trading state
in_position = False
position_price = None

# Loop through each prediction and actual price
for i in range(len(test_predict)):
    # Check if we're currently in a position
    if in_position:
        # Check if the current price has hit our stop loss or take profit thresholds
        if test_y[i] <= stop_loss_price or test_y[i] >= take_profit_price:
            # Calculate the profit or loss
            profit_loss = (test_y[i] - position_price) / position_price
            print(f"Position closed at {test_y[i]:.2f}, profit/loss of {profit_loss:.2%}")
            # Update the trading state
            in_position = False
            position_price = None
    else:
        # Check if the predicted price has crossed the stop loss or take profit thresholds
        if test_predict[i] <= stop_loss_price:
            print(f"Stop loss triggered at {test_y[i]:.2f}")
        elif test_predict[i] >= take_profit_price:
            print(f"Take profit triggered at {test_y[i]:.2f}")
        else:
            # No trading opportunity, continue
            continue

        # Enter the position at the current price
        in_position = True
        position_price = test_y[i]

# Define the ESN
esn = EchoStateNetwork(n_inputs=len(features), n_outputs=1, n_reservoir=500)

# Set the spectral radius
esn.set_spectral_radius(0.5)

# Define the learning methods
learning_methods = ['ridge_regression', 'least_squares', 'pseudo_inverse', 'pinv_tikhonov']

# Train the ESN with all types of learning methods
for method in learning_methods:
    # Set the learning method
    esn.set_learning_method(method)

    # Train the ESN
    train_error = esn.fit(train_features, train_targets)
# Predict the test set using the ESN
test_predict = esn.predict(test_x)

# Calculate the actual and predicted returns
actual_returns = test_y[1:] - test_y[:-1]
predicted_returns = test_predict[1:] - test_y[:-1]

# Define the stop loss and take profit thresholds
stop_loss = -0.01
take_profit = 0.02

# Initialize the portfolio value and positions
portfolio_value = 100000
position = 0

# Loop through the predicted returns and adjust the position and portfolio value based on the stop loss and take profit thresholds
for i in range(len(predicted_returns)):
    # Calculate the return on the current position
    position_return = position * actual_returns[i]
    
    # Check if we should sell due to stop loss
    if position_return < stop_loss * portfolio_value:
        position = 0
        portfolio_value += position_return
        continue
    
    # Check if we should sell due to take profit
    if position_return > take_profit * portfolio_value:
        position = 0
        portfolio_value += position_return
        continue
    
    # If we haven't sold, determine whether we should buy or hold
    if predicted_returns[i] > 0:
        position = portfolio_value / test_y[i]
        portfolio_value = 0
    else:
        position = 0
        
# Calculate the final portfolio value
final_portfolio_value = portfolio_value + position * test_y[-1]

# Preprocess the data
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Split the data into input and output variables
train_x, train_y = train_data[:, :-1], train_data[:, -1]
test_x, test_y = test_data[:, :-1], test_data[:, -1]

# Create the ANN model
model = Sequential()
model.add(Dense(50, input_dim=train_x.shape[1], activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Fit the model to the training data
model.fit(train_x, train_y, epochs=100, batch_size=32, verbose=0)

# Make predictions on the test set
ann_predictions = model.predict(test_x)

# Train the ESN with the ANN predictions
train_error = esn.fit(train_x, ann_predictions)

# Predict the test set using the ESN
test_predict = esn.predict(test_x)

# Define the Echo State Network
esn = EchoStateNetwork(n_inputs=4, n_outputs=1, n_reservoir=1000, spectral_radius=0.8)

# Train the ESN
train_error = esn.fit(train_inputs, train_outputs)

# Predict the test set using the ESN
test_predictions = esn.predict(test_inputs)

# Define the stop loss and take profit thresholds
stop_loss = -0.01
take_profit = 0.02

# Define the risk levels
risk_levels = [0.5, 0.75, 1.0]

# Initialize the portfolio value and positions
portfolio_values = {}
positions = {}
for pair in pairs:
    portfolio_values[pair] = 100000
    positions[pair] = 0

# Initialize the drawdown and max portfolio value
drawdowns = {}
max_portfolio_values = {}
for pair in pairs:
    drawdowns[pair] = 0
    max_portfolio_values[pair] = portfolio_values[pair]

# Define the maximum drawdown threshold
max_drawdown = 0.13

# Loop through the predicted returns and adjust the position and portfolio value based on the stop loss and take profit thresholds
for i in range(len(predicted_returns)):
    # Calculate the return on the current position for each pair
    position_returns = {}
    for pair in pairs:
        position_returns[pair] = positions[pair] * actual_returns[pair][i]
        
    # Check if we should sell due to stop loss or take profit for each pair
    for pair in pairs:
        if position_returns[pair] < stop_losses[pair] * portfolio_values[pair]:
            positions[pair] = 0
            portfolio_values[pair] += position_returns[pair]
            continue
        if position_returns[pair] > take_profits[pair] * portfolio_values[pair]:
            positions[pair] = 0
            portfolio_values[pair] += position_returns[pair]
            continue
        
        # If we haven't sold, determine whether we should buy, sell, or hold for each pair
        if predicted_returns[pair][i] > 0:
            # Calculate the maximum position size based on the drawdown and equity value
            max_position_size = min(portfolio_values[pair] / test_targets[pair][i], (1 - drawdowns[pair]) * max_portfolio_values[pair] / test_targets[pair][i])
            position_size = min(max_position_size, risk_level * portfolio_values[pair] / test_targets[pair][i])
            positions[pair] = position_size
            portfolio_values[pair] -= position_size * test_targets[pair][i]
        elif predicted_returns[pair][i] < 0:
            positions[pair] = 0
        
        # Update the drawdown and max portfolio value for each pair
        max_portfolio_values[pair] = max(max_portfolio_values[pair], portfolio_values[pair])
        drawdowns[pair] = (max_portfolio_values[pair] - portfolio_values[pair]) / max_portfolio_values[pair]
        
# Calculate the final portfolio value for each pair
final_portfolio_values = {}
for pair in pairs:
    final_portfolio_values[pair] = portfolio_values[pair] + positions[pair] * test_targets[pair][-1]

# Define a function to provide the ESN with necessary information to suggest trading strategy
def analyze_market(pair):
    # Retrieve relevant data for the pair
    historical_prices = get_historical_prices(pair)
    test_inputs, test_targets = preprocess_data(historical_prices)

    # Retrieve predicted returns for the pair
    predicted_returns = esn.predict(test_inputs)

    # Determine whether to enter or exit the market based on predicted returns
    if predicted_returns[-1] > take_profit_threshold:
        enter_market(pair, 'buy')
    elif predicted_returns[-1] < stop_loss_threshold:
        enter_market(pair, 'sell')
    else:
        exit_market(pair)
        
        # Define a function to monitor market conditions and modify open tickets if necessary
def monitor_market():
    # Loop through all open tickets
    for ticket in open_tickets:
        # Retrieve relevant data for the ticket's pair
        pair = ticket['pair']
        historical_prices = get_historical_prices(pair)
        test_inputs, test_targets = preprocess_data(historical_prices)

        # Retrieve predicted returns for the pair
        predicted_returns = esn.predict(test_inputs)

        # Determine whether to modify the ticket based on predicted returns and drawdown
        if predicted_returns[-1] > ticket['take_profit']:
            modify_ticket(ticket, 'take_profit', predicted_returns[-1])
        elif predicted_returns[-1] < ticket['stop_loss'] or ticket['drawdown'] > max_drawdown:
            modify_ticket(ticket, 'stop_loss', predicted_returns[-1])
            update_drawdown(ticket)

# Define a function to prioritize safety of equity and allow user to choose level of risk
def run_machine(risk_level):
    # Set stop loss and take profit thresholds based on risk level
    if risk_level == 'low':
        stop_loss_threshold = -0.005
        take_profit_threshold = 0.01
    elif risk_level == 'medium':
        stop_loss_threshold = -0.01
        take_profit_threshold = 0.02
    else:
        stop_loss_threshold = -0.015
        take_profit_threshold = 0.03

    # Loop through all pairs and analyze market and modify open tickets if necessary
    for pair in pairs:
        analyze_market(pair)
    monitor_market()

# Prioritize safety of equity and provide option to choose level of risk
if equity < max_drawdown * initial_equity:
    print("Drawdown exceeded maximum allowable limit. Trading stopped.")
else:
    while True:
        risk_level = input("Enter the level of risk (1-5) for this run: ")
        if risk_level not in ["1", "2", "3", "4", "5"]:
            print("Invalid input. Please enter a number between 1 and 5.")
            continue
        else:
            max_loss = initial_equity * (int(risk_level) / 10)
            print(f"Maximum allowable loss per trade is: {max_loss}")
            break

# Prioritize safety of equity by limiting maximum loss per trade
if portfolio_value * max_loss_percent / 100 < abs(positions[pair] * test_targets[pair][-1]):
    print("Loss exceeds maximum allowable loss. Closing position...")
    portfolio_values[pair] += positions[pair] * test_targets[pair][-1] - commission_cost
    positions[pair] = 0

# Allow user to choose level of risk before running the machine
if risk_level == "high":
    max_drawdown = 0.13
elif risk_level == "medium":
    max_drawdown = 0.10
elif risk_level == "low":
    max_drawdown = 0.07


# Keep drawdown under the specified limit
if portfolio_value - max_portfolio_value > max_drawdown * max_portfolio_value:
    print("Maximum allowable drawdown reached. Closing all positions...")
    for pair in pairs:
        portfolio_values[pair] += positions[pair] * test_targets[pair][-1] - commission_cost
        positions[pair] = 0
        open_tickets[pair] = []

import csv

# Read in the currency pairs from a CSV file
with open('currency_pairs.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    pairs = [row[0] for row in reader]

# Initialize dictionaries for the portfolio values and positions
portfolio_values = {pair: initial_capital for pair in pairs}
positions = {pair: 0 for pair in pairs}

# Loop over the historical data for each currency pair
for pair in pairs:
    # Load the historical data for the current currency pair
    data = load_data(pair)

    # Split the data into training and testing sets
    train_data, test_data = split_data(data, training_ratio)

    # Train the ESN using all types of learning methods
    esn = train_esn(train_data, spectral_radius, learning_method='all')

    # Use the trained ESN to predict returns for the test data
    test_inputs = create_inputs(test_data)
    test_targets = create_targets(test_data)
    test_predictions = predict_returns(test_inputs, esn)

    # Adjust the position and portfolio value based on the stop loss and take profit thresholds
    for i in range(len(test_data)):
        if test_predictions[i] > take_profit:
            positions[pair] += position_size
        elif test_predictions[i] < -stop_loss:
            positions[pair] -= position_size

        portfolio_values[pair] = portfolio_values[pair] + positions[pair] * test_targets[i]

    # Modify open tickets for the current pair if necessary based on the market conditions
    market_conditions = analyze_market(pair)
    if market_conditions == 'volatile':
        modify_tickets(pair, positions[pair], portfolio_values[pair], drawdown_limit)


# Load the list of currency pairs from a file
with open('currency_pairs.txt', 'r') as f:
    currency_pairs = f.read().splitlines()

# Loop through each currency pair and trade
for pair in currency_pairs:
    # Get the historical price data for the currency pair
    price_data = get_price_data(pair)

    # Train the ESN using all four types of learning methods
    esn = train_esn(price_data)

    # Loop through each trading period and make trades
    for i in range(len(price_data) - trading_period):
        # Get the current market conditions
        market_conditions = get_market_conditions(price_data[i:i+trading_period])

        # Make a prediction using the ESN
        predicted_return = predict_return(esn, market_conditions)

        # Adjust the position and portfolio value based on the stop loss and take profit thresholds
        positions[pair], portfolio_values[pair] = adjust_position_and_portfolio_value(predicted_return,
                                                                                       positions[pair],
                                                                                       portfolio_values[pair],
                                                                                       stop_loss,
                                                                                       take_profit)

        # Modify open tickets for the current pair if necessary based on the market conditions
        market_conditions = get_market_conditions(price_data[i+trading_period:i+2*trading_period])
        positions[pair], portfolio_values[pair] = modify_open_tickets(market_conditions,
                                                                       positions[pair],
                                                                       portfolio_values[pair],
                                                                       stop_loss,
                                                                       take_profit)

    # Print the final portfolio value for the currency pair
    print(f"Final portfolio value for {pair}: {portfolio_values[pair]}")

def read_currency_pairs():
    # Code to read currency pairs from file or database goes here
    # Returns a list of currency pairs
    pass

currency_pairs = read_currency_pairs()

import pandas as pd

# Read in the list of currency pairs from a file
pairs_df = pd.read_csv('currency_pairs.csv')

# Convert the pairs to a list
pairs = pairs_df['pair'].tolist()

# Initialize positions and portfolio values for each pair
positions = {pair: 0 for pair in pairs}
portfolio_values = {pair: 10000 for pair in pairs}

# Define function to adjust position and portfolio value based on stop loss and take profit thresholds
def adjust_position_and_portfolio_value(predicted_return, position, portfolio_value, max_loss, max_profit):
    if predicted_return < -max_loss:
        # Sell to close position
        portfolio_value += abs(position * predicted_return)
        position = 0
    elif predicted_return > max_profit:
        # Sell to close position
        portfolio_value += abs(position * predicted_return)
        position = 0
    return position, portfolio_value

# Define function to simulate trading for a single currency pair
def simulate_trading_for_pair(pair):
    # Load data for the currency pair
    data = load_data(pair)
    
    # Initialize ESN for the currency pair
    esn = initialize_esn(data)
    
    # Split data into training and testing sets
    train_data, test_data = split_data(data)
    
    # Train ESN using all four types of learning methods
    esn = train_esn(esn, train_data)
    
    # Test ESN using testing data and calculate accuracy
    accuracy = test_esn(esn, test_data)
    
    # Print accuracy for the currency pair
    print(f"Accuracy for {pair}: {accuracy}")
    
    # Set maximum allowable loss and profit thresholds
    max_loss = 0.01
    max_profit = 0.02
    
    # Loop through each time step in the data
    for i in range(len(data)):
        # Get the predicted return for the next time step
        predicted_return = predict_return(esn, data[i])
        
        # Adjust the position and portfolio value based on the stop loss and take profit thresholds
        positions[pair], portfolio_values[pair] = adjust_position_and_portfolio_value(predicted_return, positions[pair], portfolio_values[pair], max_loss, max_profit)
        
        # Modify open tickets for the current pair if necessary based on the market conditions
        market_conditions = get_market_conditions(data, i)
        modify_open_tickets_for_pair(pair, market_conditions)
        
    # Print final portfolio value for the currency pair
    print(f"Final portfolio value for {pair}: {portfolio_values[pair]}")
    
# Loop through each currency pair and simulate trading for each pair
for pair in pairs:
    simulate_trading_for_pair(pair)

# Define function to train the ESN using all four types of learning methods
def train_esn_all_methods(X, y, n_reservoir=100, spectral_radius=0.9, sparsity=0.2, alpha=0.3, n_transient=100, learning_method='inv'):
  
    esn = ESN(n_inputs=X.shape[1], n_outputs=y.shape[1], n_reservoir=n_reservoir, spectral_radius=spectral_radius, sparsity=sparsity, alpha=alpha)
    for method in ['inv', 'ridge', 'pseudo', 'lasso']:
        esn.fit(X, y, n_transient=n_transient, learning_method=method)
    return esn

# Define function to prompt the user for their preferred risk level
def get_risk_level():
   

    risk_level = input("Enter your preferred risk level (0-1): ")
    while not (risk_level.isdigit() and 0 <= float(risk_level) <= 1):
        risk_level = input("Please enter a number between 0 and 1: ")
    return float(risk_level)

# Define main function to run the program
def main():
    # Read in list of currency pairs
    with open('currency_pairs.txt') as f:
        currency_pairs = f.read().splitlines()

    # Get user's preferred risk level
    risk_level = get_risk_level()

    # Define dictionary to hold current positions and portfolio values for each currency pair
    positions = {pair: 0 for pair in currency_pairs}
    portfolio_values = {pair: 10000 for pair in currency_pairs}  # Starting portfolio value is $10,000 for each currency pair

    # Train ESN using all four types of learning methods
    X, y = get_input_data(currency_pairs)
    esn = train_esn_all_methods(X, y)

    # Define variables to keep track of total profit and drawdown
    total_profit = 0
    max_portfolio_values = {pair: 10000 for pair in currency_pairs}  # Starting maximum portfolio value is $10,000 for each currency pair
    drawdowns = {pair: 0 for pair in currency_pairs}
# Start trading loop
for i in range(LOOKBACK_PERIOD, len(X)):
    # Get predicted returns for all currency pairs
    predicted_returns = esn.predict(X[i].reshape(1, -1))[0]

    # Adjust the position and portfolio value based on the stop loss and take profit thresholds
    for j, pair in enumerate(currency_pairs):
        predicted_return = predicted_returns[j]
        stop_loss = get_stop_loss(predicted_return, risk_level)
        take_profit = get_take_profit(predicted_return, risk_level)
        positions[pair], portfolio_values[pair] = adjust_position_and_portfolio_value(predicted_return, positions[pair], portfolio_values[pair], stop_loss, take_profit)
        if portfolio_values[pair] > max_portfolio_values[pair]:
            max_portfolio_values[pair] = portfolio_values[pair]
        drawdowns[pair] = (max_portfolio_values[pair] - portfolio_values[pair]) / max_portfolio_values[pair]

        # Modify open tickets for the current pair if necessary based on the market conditions
        market_conditions = get_market_conditions(X[:i],currency_pairs)
        if market_conditions[pair] == 'Trending':
            if positions[pair] == 0:
                # Open a long or short position depending on the direction of the trend
                if predicted_return > 0:
                    positions[pair] = 1
                else:
                    positions[pair] = -1
            elif positions[pair] > 0:
                # Close the long position if the trend is down
                if predicted_return < 0:
                    positions[pair] = 0
                    portfolio_values[pair] = adjust_portfolio_value(positions[pair], portfolio_values[pair], predicted_return)
                    max_portfolio_values[pair] = portfolio_values[pair]
            elif positions[pair] < 0:
                # Close the short position if the trend is up
                if predicted_return > 0:
                    positions[pair] = 0
                    portfolio_values[pair] = adjust_portfolio_value(positions[pair], portfolio_values[pair], predicted_return)
                    max_portfolio_values[pair] = portfolio_values[pair]

        elif market_conditions[pair] == 'Ranging':
            if positions[pair] != 0:
                # Close the position if the market is ranging
                positions[pair] = 0
                portfolio_values[pair] = adjust_portfolio_value(positions[pair], portfolio_values[pair], predicted_return)
                max_portfolio_values[pair] = portfolio_values[pair]

# Continuously monitor market conditions and adjust open tickets in real-time based on changes in market conditions
while True:
    # Get the latest market data
    X, Y = get_latest_data(currency_pairs, window_size, num_features, data_dir)

    # Compute predicted returns and confidence scores for each currency pair
    predicted_returns = {}
    confidence_scores = {}
    for pair in currency_pairs:
        if len(X[pair]) > 0:
            # Use the ESN to predict the next return for this currency pair
            predicted_return = esn.predict(X[pair][-1])
            predicted_returns[pair] = predicted_return

            # Compute the confidence score for this prediction
            confidence_score = np.abs(predicted_return - np.mean(Y[pair])) / np.std(Y[pair])
            confidence_scores[pair] = confidence_score

        # Determine the market conditions based on the latest data
    market_conditions = get_market_conditions(X, currency_pairs)

    # Adjust the position and portfolio value based on the stop loss and take profit thresholds
    positions, portfolio_values = adjust_position_and_portfolio_value(predicted_returns, positions, portfolio_values,
                                                                      stop_loss, take_profit, max_loss)

    # Update the drawdown for each currency pair
    for pair in currency_pairs:
        drawdowns[pair] = compute_drawdown(positions[pair], portfolio_values[pair])

# Check if any open tickets need to be modified or closed based on changes in market conditions
for pair in currency_pairs:
    if positions[pair] != 0:
        if market_conditions[pair] == "bear":
            # Close the long position if the market is bearish
            positions[pair] = 0
            portfolio_values[pair] -= positions[pair] * X[pair][-1][-1]
        elif market_conditions[pair] == "bull":
            if drawdowns[pair] >= max_drawdown:
                # Close the long position if the drawdown exceeds the maximum allowable drawdown
                positions[pair] = 0
                portfolio_values[pair] -= positions[pair] * X[pair][-1][-1]
            elif confidence_scores[pair] >= confidence_threshold:
                # Increase the position size if the confidence score exceeds the threshold
                new_position_size = min(position_size * (1 + risk_level), max_position_size)
                additional_positions = np.floor(
                    (portfolio_values[pair] * (1 + risk_level)) / (X[pair][-1][-1] * new_position_size)) - 1
                positions[pair] += additional_positions
                portfolio_values[pair] -= additional_positions * X[pair][-1][-1] * new_position_size

# Wait for the specified number of seconds before checking the market again
time.sleep(interval)


def backtest_strategy(X, y, currency_pairs, position_size, max_position_size, stop_loss, take_profit, max_loss, max_drawdown, confidence_threshold, risk_level):
    # Initialize the positions and portfolio values for each currency pair
    positions = {pair: 0 for pair in currency_pairs}
    portfolio_values = {pair: 0 for pair in currency_pairs}

    # Initialize the drawdowns for each currency pair
    drawdowns = {pair: 0 for pair in currency_pairs}

    # Initialize the total profit and loss
    total_pnl = 0

    # Loop through the historical data
    for i in range(len(X)):
        # Get the latest data
        X_latest = {pair: X[pair][:i+1] for pair in currency_pairs}
        y_latest = {pair: y[pair][:i+1] for pair in currency_pairs}

        # Predict the returns for each currency pair
        predicted_returns = predict_returns(X_latest, y_latest)

        # Determine the market conditions based on the latest data
        market_conditions = get_market_conditions(X_latest, currency_pairs)

        # Adjust the position and portfolio value based on the stop loss and take profit thresholds
        positions, portfolio_values = adjust_position_and_portfolio_value(predicted_returns, positions, portfolio_values, stop_loss, take_profit, max_loss)

        # Update the drawdown for each currency pair
        for pair in currency_pairs:
            drawdowns[pair] = compute_drawdown(positions[pair], portfolio_values[pair])

        # Check if any open tickets need to be modified or closed based on changes in market conditions
        for pair in currency_pairs:
            if positions[pair] != 0:
                if market_conditions[pair] == "bear":
                    # Close the long position if the market is bearish
                    pnl = positions[pair] * (X_latest[pair][-1][-1] - X_latest[pair][-2][-1])
                    total_pnl += pnl
                    positions[pair] = 0
                    portfolio_values[pair] -= positions[pair] * X_latest[pair][-1][-1]
                elif market_conditions[pair] == "bull":
                    if drawdowns[pair] >= max_drawdown:
                        # Close the long position if the drawdown exceeds the maximum allowable drawdown
                        pnl = positions[pair] * (X_latest[pair][-1][-1] - X_latest[pair][-2][-1])
                        total_pnl += pnl
                        positions[pair] = 0
                        portfolio_values[pair] -= positions[pair] * X_latest[pair][-1][-1]
                    elif confidence_scores[pair] >= confidence_threshold:
                        # Increase the position size if the confidence score exceeds the threshold
                        new_position_size = min(position_size * (1 + risk_level), max_position_size)
                        additional_positions = np.floor((portfolio_values[pair] * (1 + risk_level)) / (X_latest[pair][-1][-1] * new_position_size)) - 1
                        positions[pair] += additional_positions
                        portfolio_values[pair] -= additional_positions * X_latest[pair][-1][-1] * new_position_size

    return total_pnl
