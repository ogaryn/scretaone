//+------------------------------------------------------------------+
//|                                                     paulilla.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#include <ANN/ANN.mqh>

#include <ANN/ANN.mqh>

// Define the size of the training set
const int TRAINING_SET_SIZE = 80;

// Define the number of neurons in the input, hidden, and output layers
const int INPUT_LAYER_SIZE = 1;
const int HIDDEN_LAYER_SIZE = 10;
const int OUTPUT_LAYER_SIZE = 1;

// Define the trading parameters
const double LOTS = 0.1;
const double STOP_LOSS = 100;
const double TAKE_PROFIT = 200;

// Define the ANN model
ANN ann;

// Define the function to normalize the data
double Normalize(double value, double min_value, double max_value) {
   return (value - min_value) / (max_value - min_value);
}

// Define the function to denormalize the output
double Denormalize(double value, double min_value, double max_value) {
   return value * (max_value - min_value) + min_value;
}

// Define the function to enter a trade
void EnterTrade(double price) {
   double stop_loss = price - STOP_LOSS * _Point;
   double take_profit = price + TAKE_PROFIT * _Point;
   int ticket = OrderSend(_Symbol, OP_BUY, LOTS, price, 0, stop_loss, take_profit, "ANN", 0, 0, Green);
   if(ticket > 0) {
      Print("Buy order placed at price ", price);
   }
   else {
      Print("Error placing buy order: ", GetLastError());
   }
}

// Define the function to exit a trade
void ExitTrade(double price) {
   int ticket = OrderClose(_Ticket, LOTS, price, 0, Red);
   if(ticket) {
      Print("Sell order placed at price ", price);
   }
   else {
      Print("Error placing sell order: ", GetLastError());
   }
}

// Define the OnTick() function to check for trading signals
void OnTick() {
   // Get the current price and normalize it
   double current_price = Normalize(_Close, _Lowest(D'2022.01.01', D'2022.03.20', MODE_LOW), _Highest(D'2022.01.01', D'2022.03.20', MODE_HIGH));

   // Feed the current price into the ANN model to get the predicted price
   double predicted_price = Denormalize(ann.FeedForward(current_price), _Lowest(D'2022.01.01', D'2022.03.20', MODE_LOW), _Highest(D'2022.01.01', D'2022.03.20', MODE_HIGH));

   // Check if the predicted price is above the current price and we don't already have a buy order open
   if(predicted_price > _Close && OrderType() == -1) {
      EnterTrade(predicted_price);
   }

   // Check if the predicted price is below the current price and we have a buy order open
   if(predicted_price < _Close && OrderType() == OP_BUY) {
      ExitTrade(predicted_price);
   }
}

// Define the OnInit() function to initialize the ANN model and historical data
int OnInit() {
   // Collect historical data for XAUUSD
   datetime start_date = D'2021.01.01';
   datetime end_date = D'2023.03.20';
   int data_count = TRAINING_SET_SIZE + 20; // add 20 for testing set
   MqlRates rates[];

   if(CopyRates(_Symbol, _Period, start_date, data_count,rates) != data_count) {
Print("Error: Could not copy historical rates data!");
return INIT_FAILED;
}

// Normalize the data
double normalized_data[data_count];
double min_value = ArrayMinimum(rates, WHOLE_ARRAY, 1, MODE_LOW);
double max_value = ArrayMaximum(rates, WHOLE_ARRAY, 1, MODE_HIGH);

for(int i = 0; i < data_count; i++) {
normalized_data[i] = (rates[i].close - min_value) / (max_value - min_value);
}

// Split the data into training and testing sets
double training_set[TRAINING_SET_SIZE];
double testing_set[data_count - TRAINING_SET_SIZE];

ArrayCopy(normalized_data, 0, training_set, 0, TRAINING_SET_SIZE);
ArrayCopy(normalized_data, TRAINING_SET_SIZE, testing_set, 0, data_count - TRAINING_SET_SIZE);

// Build the ANN model
ANN ann;
ann.Create(1, 10, 1); // input size = 1, hidden layer size = 10, output size = 1
ann.Activate();

// Train the model using the training set
double learning_rate = 0.1;
int max_epochs = 1000;

for(int i = 0; i < max_epochs; i++) {
for(int j = 0; j < TRAINING_SET_SIZE; j++) {
double inputs[] = {training_set[j]};
double targets[] = {training_set[j+1]};
ann.Train(inputs, targets, learning_rate);
}
}

// Test the model's performance using the testing set
double mse = 0;

for(int i = 0; i < data_count - TRAINING_SET_SIZE - 1; i++) {
double input = testing_set[i];
double target = testing_set[i+1];
double output = ann.FeedForward(input);

scss
Copy code
  mse += MathPow(target - output, 2);
}

mse /= (data_count - TRAINING_SET_SIZE - 1);
Print("Mean squared error: ", mse);

return INIT_SUCCEEDED;
}

// Define the OnTick() function to make trading decisions based on the ANN model
void OnTick() {
double current_price = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID), _Digits);
double predicted_price = ann.FeedForward((current_price - min_value) / (max_value - min_value));

if(predicted_price > current_price && !IsTradeOpen()) {
OpenBuyTrade();
} else if(predicted_price < current_price && !IsTradeOpen()) {
OpenSellTrade();
} else if(predicted_price < current_price && IsLongPosition()) {
CloseLongTrade();
} else if(predicted_price > current_price && IsShortPosition()) {
CloseShortTrade();
}
}

// Define the IsTradeOpen() function to check if there are any open trades
bool IsTradeOpen() {
int total = PositionsTotal();

for(int i = 0; i < total; i++) {
if(PositionGetSymbol(i) == _Symbol && PositionGetInteger(i, POSITION_TYPE) != POSITION_TYPE_HISTORY) {
return true;
}
}

return false;
}

// Define the OpenBuyTrade() function to open a buy trade
void OpenBuyTrade() {
double lot_size = NormalizeDouble(AccountFreeMargin() / 1000, 2);
int ticket = OrderSend(_Symbol, OP_BUY, lot_size, SymbolInfoDouble(_Symbol, SYMBOL_ASK), 5,
double normalized_data[data_count];
double min_value = ArrayMinimum(rates, WHOLE_ARRAY, 1, MODE_LOW);
double max_value = ArrayMaximum(rates, WHOLE_ARRAY, 1, MODE_HIGH);

for(int i = 0; i < data_count; i++) {
normalized_data[i] = (rates[i].close - min_value) / (max_value - min_value);
}

// Split the data into training and testing sets
double training_set[TRAINING_SET_SIZE];
double testing_set[data_count - TRAINING_SET_SIZE];

ArrayCopy(normalized_data, 0, training_set, 0, TRAINING_SET_SIZE);
ArrayCopy(normalized_data, TRAINING_SET_SIZE, testing_set, 0, data_count - TRAINING_SET_SIZE);

// Build the ANN model
ANN ann;
ann.Create(1, 10, 1); // input size = 1, hidden layer size = 10, output size = 1
ann.Activate();

// Train the model using the training set
double learning_rate = 0.1;
int max_epochs = 1000;

for(int i = 0; i < max_epochs; i++) {
for(int j = 0; j < TRAINING_SET_SIZE; j++) {
double inputs[] = {training_set[j]};
double targets[] = {training_set[j+1]};
ann.Train(inputs, targets, learning_rate);
}
}

// Test the model's performance using the testing set
double mse = 0;

for(int i = 0; i < data_count - TRAINING_SET_SIZE - 1; i++) {
double input = testing_set[i];
double target = testing_set[i+1];
double output = ann.FeedForward(input);

scss
Copy code
  mse += MathPow(target - output, 2);
}

mse /= (data_count - TRAINING_SET_SIZE - 1);
Print("Mean squared error: ", mse);

// Use the model to predict future values
double last_input = testing_set[data_count - TRAINING_SET_SIZE - 1];
double predicted_values[20];

for(int i = 0; i < 20; i++) {
double predicted_value = ann.FeedForward(last_input);
predicted_values[i] = predicted_value;

scss
Copy code
  // Shift the inputs by one to make room for the predicted value
  for(int j = TRAINING_SET_SIZE - 1; j >= 1; j--) {
     training_set[j] = training_set[j-1];
  }

  // Add the predicted value to the inputs for the next iteration
  training_set[0] = predicted_value;
  last_input = predicted_value;
}

// Calculate the current signal using the predicted values
double current_input = testing_set[data_count - TRAINING_SET_SIZE - 1];
double current_output = ann.FeedForward(current_input);
double current_signal = 0;

if(current_output > current_input) {
current_signal = 1;
} else if(current_output < current_input) {
current_signal = -1;
}

// Check if the current signal is different from the previous signal
double prev_signal = 0;

if(CopyBuffer(SignalBuffer, 0, 0, 1, prev_signal) > 0) {
if(current_signal != prev_signal) {
// If the current signal is different from the previous signal, enter a trade
if(current_signal == 1) {
// Enter a long position
Trade.Open(_Symbol, TRADE_TYPE_BUY, Lots, 0

