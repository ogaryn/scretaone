// Include necessary libraries
#include <ANN/ANN.h>
#include <math.h>

// Define the ANN model
CNeuralNetwork ann;
const int input_size = 21;
const int hidden_size = 10;
const int output_size = 1;
double normalized_data[input_size];

// Define the size of the training set
const int TRAINING_SET_SIZE = 80;

// Declare global variables to store min_value, max_value, and the ANN model
double min_value, max_value;

// Define the OnInit() function to initialize the ANN model
int OnInit() {
   // Load the trained weights
   if(!ann.Load("ann_weights.bin")) {
      Print("Error: Could not load ANN weights!");
      return INIT_FAILED;
   }

   // Normalize the input data
   for(int i = 0; i < input_size; i++) {
      normalized_data[i] = (data[i] - min_value) / (max_value - min_value);
   }

   return INIT_SUCCEEDED;
}

// Define the OnDeinit() function to free memory used by the ANN model
void OnDeinit(const int reason) {
   ann.Free();
}

// Define the OnTick() function to make trading decisions based on the ANN model's predictions
void OnTick() {
   // Get the current market price
   double current_price = MarketInfo(_Symbol, MODE_BID);

   // Get the previous 20 prices
   MqlRates rates[20];
   ArraySetAsSeries(rates, true);

   if(CopyRates(_Symbol, _Period, TimeCurrent(), 20, rates) != 20) {
      Print("Error: Could not copy historical rates data!");
      return;
   }

   // Normalize the data
   double normalized_data[21];
   for(int i = 0; i < 20; i++) {
      normalized_data[i] = (rates[i].close - min_value) / (max_value - min_value);
   }
   normalized_data[20] = (current_price - min_value) / (max_value - min_value);

   // Use the model to predict the next price
   double input[input_size];
   for(int i = 0; i < input_size; i++) {
      input[i] = normalized_data[i];
   }
   double predicted_price = ann.FeedForward(input) * (max_value - min_value) + min_value;

   // Enter a long trade if the predicted price is higher than the current price
   if(predicted_price > current_price && !PositionSelect(_Symbol)) {
      int ticket = OrderSend(_Symbol, OP_BUY, 0.01, current_price, 0, 0, 0, "ANN Trading System", 0, 0, Green);
      if(ticket > 0) {
         Print("Long trade opened at ", current_price, " with predicted price of ", predicted_price);
      } else {
         Print("Error opening long trade: ", GetLastError());
      }
   }

   // Exit the long trade if the predicted price is lower than the current price
   if(predicted_price < current_price && PositionSelect(_Symbol) && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {
      int ticket = OrderClose(PositionGetTicket(), 0.01, current_price, 0, Green);
      if(ticket > 0) {
         Print("Long trade closed at ", current_price, " with predicted price of

// Exit the long trade if the predicted price is lower than the current price
if(predicted_price < current_price && PositionSelect(_Symbol) && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {
   int ticket = OrderClose(PositionGetTicket(), 0.01, current_price, 0, Green);
   if(ticket > 0) {
      Print("Long trade closed at ", current_price, " with predicted price of ", predicted_price);
   } else {
      Print("Error closing long trade: ", GetLastError());
   }
}

// Enter a short trade if the predicted price is lower than the current price
if(predicted_price < current_price && !PositionSelect(_Symbol)) {
   int ticket = OrderSend(_Symbol, OP_SELL, 0.01, current_price, 0, 0, 0, "ANN Trading System", 0, 0, Red);
   if(ticket > 0) {
      Print("Short trade opened at ", current_price, " with predicted price of ", predicted_price);
   } else {
      Print("Error opening short trade: ", GetLastError());
   }
}

// Exit the short trade if the predicted price is higher than the current price
if(predicted_price > current_price && PositionSelect(_Symbol) && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) {
   int ticket = OrderClose(PositionGetTicket(), 0.01, current_price, 0, Red);
   if(ticket > 0) {
      Print("Short trade closed at ", current_price, " with predicted price of ", predicted_price);
   } else {
      Print("Error closing short trade: ", GetLastError());
   }
}

// Define the OnDeinit() function to clean up the ANN model
void OnDeinit(const int reason) {
   ann.Destroy();
}

//+------------------------------------------------------------------+







