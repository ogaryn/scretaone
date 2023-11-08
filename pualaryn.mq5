
// Define neural networks
CNet "StudyNet";
CNet "TargetNet";

// Declare variables
double StudyNet[10], TargetNet[10];
double Alpha, Gamma, Theta, Delta, Momentum;
double WeightsIH[10][10], WeightsHO[10][10];
double Hidden[10], Output[10];
double DeltaHidden[10], DeltaOutput[10];

// Define constants
const double Beta = 1.0;

// Define input array
double input[10];

// Define function to activate neural network
double Activate(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Train neural network
void Train() {
    // Loop through training data
    for (int i = 0; i < 100; i++) {
        // Set input values
        input[0] = 1.0;
        input[1] = 2.0;
        input[2] = 3.0;

        // Calculate hidden layer values
        for (int j = 0; j < 10; j++) {
            double sum = 0.0;
            for (int k = 0; k < 10; k++) {
                sum += input[k] * WeightsIH[k][j];
            }
            Hidden[j] = Activate(sum);
        }

        // Calculate output layer values
        for (int j = 0; j < 10; j++) {
            double sum = 0.0;
            for (int k = 0; k < 10; k++) {
                sum += Hidden[k] * WeightsHO[k][j];
            }
            Output[j] = Activate(sum);
        }

        // Calculate delta values for output layer
        for (int j = 0; j < 10; j++) {
            DeltaOutput[j] = Beta * Output[j] * (1.0 - Output[j]) * (TargetNet[j] - Output[j]);
        }

        // Calculate delta values for hidden layer
        for (int j = 0; j < 10; j++) {
            double sum = 0.0;
            for (int k = 0; k < 10; k++) {
                sum += DeltaOutput[k] * WeightsHO[j][k];
            }
            DeltaHidden[j] = Beta * Hidden[j] * (1.0 - Hidden[j]) * sum;

// Initialize function
int OnInit()
{
    // Initialize StudyNet
    StudyNet.Create(eCount, 50, 1);
    StudyNet.SetActivationFunction(SIGMOID);

    // Initialize TargetNet
    TargetNet.Create(eCount, 50, 1);
    TargetNet.SetActivationFunction(SIGMOID);

    return(INIT_SUCCEEDED);
}

// Deinitialize function
void OnDeinit(const int reason)
{
    // Deinitialize StudyNet
    StudyNet.Delete();

    // Deinitialize TargetNet
    TargetNet.Delete();
}

// Start function
void OnTick()
{
    // Get data from the last HistoryBars bars
    for(int bar = 0; bar < HistoryBars; bar++)
    {
        MqlRates Rates[1];
        int copied = CopyRates(_Symbol, PERIOD_CURRENT, bar, 1, Rates);
        if(copied != 1)
        {
            Print("Failed to copy rates");
            return;
        }

        datetime Time = Rates[0].time;
        double Close = Rates[0].close;
        double High = Rates[0].high;
        double Low = Rates[0].low;
        double Volume = Rates[0].tick_volume;

        // Save data to TempData
        TempData[eClose][bar] = Close;
        TempData[eHigh][bar] = High;
        TempData[eLow][bar] = Low;
        TempData[eVolume][bar] = Volume;
    }
}

// Update function
void OnTimer()
{
    // Create input vector
    double InputVector[eCount * HistoryBars];
    for(int i = 0; i < eCount * HistoryBars; i += eCount)
    {
        int bar = i / eCount;
        InputVector[i + eClose] = TempData[eClose][bar];
        InputVector[i + eHigh] = TempData[eHigh][bar];
        InputVector[i + eLow] = TempData[eLow][bar];
        InputVector[i + eVolume] = TempData[eVolume][bar];
    }

    // Feed input vector to StudyNet and get output
    double StudyOutput[1];
    StudyNet.FeedForward(InputVector, StudyOutput);

    // Feed input vector to TargetNet and get output
    double TargetOutput[1];
    TargetNet.FeedForward(InputVector, TargetOutput);

    // Update StudyNet and TargetNet
    if(StudyOutput[0] > TargetOutput[0])
    {
        TargetNet.CopyFrom(StudyNet);
    }
    else if(TargetOutput[0] > StudyOutput[0])
    {
        StudyNet.CopyFrom(TargetNet);
    }
}

constexpr int TRAINING_SET_SIZE = 80;
class ANN {
public:
    void Create(int input_size, int hidden_layer_size, int output_size);
    void Activate();
    void Train(const double* inputs, const double* targets, double learning_rate);
    double FeedForward(double input);

private:
    int m_input_size;
    int m_hidden_layer_size;
    int m_output_size;
    double* m_input_layer;
    double* m_hidden_layer;
    double* m_output_layer;
    double** m_input_weights;
    double**

void ANN::Train(const double* inputs, const double* targets, double learning_rate) {
// Set input layer
for(int i = 0; i < m_input_size; i++) {
m_input_layer[i] = inputs[i];
}
// Activate network to get output
Activate();

// Backpropagate error and adjust weights
for(int i = 0; i < m_output_size; i++) {
    double error = targets[i] - m_output_layer[i];
    for(int j = 0; j < m_hidden_layer_size; j++) {
        double delta = learning_rate * error * m_hidden_layer[j];
        m_hidden_weights[j][i] += delta;
    }
}

for(int i = 0; i < m_hidden_layer_size; i++) {
    double error = 0;
    for(int j = 0; j < m_output_size; j++) {
        error += (targets[j] - m_output_layer[j]) * m_hidden_weights[i][j];
    }
    for(int k = 0; k < m_input_size; k++) {
        double delta = learning_rate * error * m_input_layer[k];
        m_input_weights[k][i] += delta;
    }
}


}

double ANN::FeedForward(double input) {
m_input_layer[0] = input;
Activate();
return m_output_layer[0];
}

// Train function
void Train()
{
// Create neural network
ANN neural_network;
neural_network.Create(1, 4, 1);
// Train neural network
for(int i = 0; i < TRAINING_SET_SIZE; i++) {
    double input = (double)i / TRAINING_SET_SIZE;
    double target = sin(input * 2.0 * M_PI);
    neural_network.Train(&input, &target, 0.1);
}

// Test neural network
for(int i = 0; i < TRAINING_SET_SIZE; i++) {
    double input = (double)i / TRAINING_SET_SIZE;
    double output = neural_network.FeedForward(input);
    double target = sin(input * 2.0 * M_PI);
    Print("input = ", input, ", output = ", output, ", target = ", target);
}



#include <ANN/ANNCreator.mqh>
#include <ANN/ANNModel.mqh>

class CExpertAdvisor : public CExpertAdvisorBase {
private:
    ANNModel g_model;
public:
    void OnTick() override {
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        if (balance <= 0)
            return;
        if (OrderGetInteger(ORDER_TYPE) != -1)
            return;
        if (ANNPredict(g_model, { Ask, Bid })[0] > 0.5)
            OrderSend(_Symbol, OP_BUY, 0.1, Ask, 10, Bid - 30 * _Point, Ask + 40 * _Point);
    }
};

input int n_input = 5;
input int n_hidden_layers = 1;
input int n_neurons_in_hidden_layers[] = { 3 };
input int n_output = 1;
input double learning_rate = 0.01;
input double momentum = 0.9;
input double regularization_factor = 0.01;
input int n_train_data = 1000;
input int n_test_data = 100;

void OnStart() {
    double input[n_input][n_train_data + n_test_data];
    double output[n_output][n_train_data + n_test_data];
    for (int i = 0; i < n_input; i++) {
        ArrayInitialize(input[i], 0.0);
    }
    ArrayInitialize(output[0], 0.0);

    ANNCreator creator;
    ANNModel model = creator.Create(n_input, n_output, n_hidden_layers, n_neurons_in_hidden_layers, ANN_FUNC_SIGMOID);

    while (true) {
        // get latest bid and ask prices
        double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

        // fill input array
        input[0][0] = AccountInfoDouble(ACCOUNT_BALANCE);
        input[1][0] = NormalizeDouble(bid, _Digits);
        input[2][0] = NormalizeDouble(ask, _Digits);
        input[3][0] = NormalizeDouble(bid - ask, Digits);

input[3][1] = NormalizeDouble((bid - ask) / ask * 100, 2);

input[3][2] = NormalizeDouble((bid - ask) / Point, 1);

input[4][0] = iHigh(NULL, PERIOD_H1, 1) - iLow(NULL, PERIOD_H1, 1);
input[4][1] = iHigh(NULL, PERIOD_H1, 1) - bid;
input[4][2] = ask - iLow(NULL, PERIOD_H1, 1);

input[5][0] = iATR(NULL, PERIOD_H1, 14, 1);
input[5][1] = NormalizeDouble((ask - bid) / input[5][0], 2);
void OnTick()
{
    double ask = MarketInfo(Symbol(), MODE_ASK);
    double bid = MarketInfo(Symbol(), MODE_BID);
    double spread = ask - bid;
    
    double input[4][1];
    input[0][0] = iMA(Symbol(), PERIOD_H1, 10, 0, MODE_SMA, PRICE_CLOSE, 0);
    input[1][0] = iMA(Symbol(), PERIOD_H1, 20, 0, MODE_SMA, PRICE_CLOSE, 0);
    input[2][0] = iMA(Symbol(), PERIOD_H1, 50, 0, MODE_SMA, PRICE_CLOSE, 0);
    input[3][0] = NormalizeDouble(bid - ask, Digits);
    
    double output;
    int result = mlp_classify(network, input, 4, &output);
    
    if (result == 1) // Buy signal
    {
        double lotSize = NormalizeDouble(AccountBalance() * riskRatio / (100 * spread), 2);
        double stopLoss = ask - stopLossPips * Point;
        double takeProfit = bid + takeProfitPips * Point;
        
        int ticket = OrderSend(Symbol(), OP_BUY, lotSize, ask, 3, stopLoss, takeProfit, "MLP buy", 0, 0, Green);
        
        if (ticket > 0)
        {
            Print("Buy order opened successfully");
        }
        else
        {
            Print("Error opening buy order: ", GetLastError());
        }
    }
    else if (result == -1) // Sell signal
    {
        double lotSize = NormalizeDouble(AccountBalance() * riskRatio / (100 * spread), 2);
        double stopLoss = bid + stopLossPips * Point;
        double takeProfit = ask - takeProfitPips * Point;
        
        int ticket = OrderSend(Symbol(), OP_SELL, lotSize, bid, 3, stopLoss, takeProfit, "MLP sell", 0, 0, Red);
        
        if (ticket > 0)
        {
            Print("Sell order opened successfully");
        }
        else
        {
            Print("Error opening sell order: ", GetLastError());
        }
    }
}
