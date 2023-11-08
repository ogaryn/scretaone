#include <Trade\Trade.mqh>
CTrade trade;

#include <PositionInfo.mqh>
CPositionInfo m_position;

enum ENUM_ST
{
    Average = 0, // Average
    PartClose = 1 // Part Close
};

// Variable Declarations
double iStartLots = 0.01;          // Initial lot size
double iMinLotSize = 0.01;         // Minimum lot size
double iMaxLotSize = 10.00;        // Maximum lot size
double iMaxStopLossDistance = 30.0; // Maximum stop loss distance in pips
int iMagicNumber = 5050;
int iSlippage = 5;          // Magic number for identifying positions
double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
double riskPercentage = 0.02;  // Example risk percentage of 2%
double maxRiskPerTrade = accountBalance * riskPercentage;

int OnInit()
{
    // Enable dynamic lot size
    Comment(DoubleToString(accountBalance));
    trade.LogLevel(LOG_LEVEL_ERRORS);
    trade.SetExpertMagicNumber(iMagicNumber);
    trade.SetDeviationInPoints(iSlippage);
    trade.SetMarginMode();
    trade.SetTypeFillingBySymbol(Symbol());

    return INIT_SUCCEEDED;
}

// Lot Size Calculation based on Market Volatility and Account Balance
double CalculateLotSize()
{
    // Calculate lot size based on desired risk per trade
    double riskAmount = accountBalance * maxRiskPerTrade;

    // Calculate stop loss distance in points
    double stopLossDistance = iMaxStopLossDistance * SymbolInfoDouble(_Symbol, SYMBOL_POINT);

    // Calculate lot size
    double lotSize = (riskAmount / stopLossDistance) / 10000.0;

    // Limit lot size within the defined range
    lotSize = MathMax(iMinLotSize, MathMin(iMaxLotSize, lotSize));

    return NormalizeDouble(lotSize, 2);
}

double CalculatePositionSize()
{
    double maxRiskPerTrade = accountBalance * riskPercentage;
    double stopLossDistance = iMaxStopLossDistance * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    double positionSize = MathMin(maxRiskPerTrade / stopLossDistance, iMaxLotSize);
    return positionSize;
}

// Adjust Stop Loss based on Market Volatility
double AdjustStopLoss(double stopLoss)
{
    double BuyPriceMax = 0, BuyPriceMin = 0, BuyPriceMaxLot = 0, BuyPriceMinLot = 0;
    double SelPriceMin = 0, SelPriceMax = 0, SelPriceMinLot = 0, SelPriceMaxLot = 0;
    ulong BuyPriceMaxTic = 0, BuyPriceMinTic = 0, SelPriceMaxTic = 0, SelPriceMinTic = 0;
    double op = 0, lt = 0;
    ulong tk = 0;
    int b = 0, s = 0;

    int total = PositionsTotal();
    for (int k = total - 1; k >= 0; k--)
    {
        if (m_position.SelectByIndex(k))
        {
            if (m_position.Symbol() == Symbol())
            {
                if (m_position.Magic() == iMagicNumber)
                {
                    if (m_position.Type() == POSITION_TYPE_BUY || m_position.Type() == POSITION_TYPE_SELL)
                    {
                        op = NormalizeDouble(m_position.PriceOpen(), Digits());
                        lt = NormalizeDouble(m_position.Volume(), 2);
                        tk = m_position.Ticket();

                        if (m_position.Type() == POSITION_TYPE_BUY)
                        {
                            if (BuyPriceMax == 0 || BuyPriceMax < op)
                            {
                                BuyPriceMax = op;
                                BuyPriceMaxLot = lt;
                                BuyPriceMaxTic = tk;
                            }

                            if (BuyPriceMin == 0 || BuyPriceMin > op)
                            {
                                BuyPriceMin = op;
                                BuyPriceMinLot = lt;
                                BuyPriceMinTic = tk;
                            }

                            b++;
                        }
                        else
                        {
                            if (SelPriceMax == 0 || SelPriceMax < op)
                            {
                                SelPriceMax = op;
                                SelPriceMaxLot = lt;
                                SelPriceMaxTic = tk;
                            }

                            if (SelPriceMin == 0 || SelPriceMin > op)
                            {
                                SelPriceMin = op;
                                SelPriceMinLot = lt;
                                SelPriceMinTic = tk;
                            }

                            s++;
                        }
                    }
                }
            }
        }
    }

    double stopLossDistance = iMaxStopLossDistance * SymbolInfoDouble(_Symbol, SYMBOL_POINT);

    // Calculate the average price
    double avgPrice = 0;
    double totalLots = 0;
    double totalAmount = 0;

    if (b > 0)
    {
        avgPrice += BuyPriceMax * BuyPriceMaxLot;
        totalLots += BuyPriceMaxLot;
        totalAmount += BuyPriceMaxLot * BuyPriceMax;
    }

    if (s > 0)
    {
        avgPrice += SelPriceMax * SelPriceMaxLot;
        totalLots += SelPriceMaxLot;
        totalAmount += SelPriceMaxLot * SelPriceMax;
    }

    if (totalLots > 0)
    {
        avgPrice /= totalLots;
        totalAmount /= totalLots;
    }

    double newStopLoss = stopLossDistance;

    if (b > 0 && s == 0)
    {
        newStopLoss += (avgPrice - stopLossDistance);
    }
    else if (b == 0 && s > 0)
    {
        newStopLoss -= (avgPrice - stopLossDistance);
    }
    else if (b > 0 && s > 0)
    {
        newStopLoss += (avgPrice - stopLossDistance) * totalAmount;
    }

    return newStopLoss;
}

void OnTick()
{
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double stopLoss = AdjustStopLoss(iMaxStopLossDistance * SymbolInfoDouble(_Symbol, SYMBOL_POINT));

    // Check if there are any open positions
    int total = PositionsTotal();
    if (total > 0)
    {
        for (int i = total - 1; i >= 0; i--)
        {
            if (m_position.SelectByIndex(i))
            {
                if (m_position.Symbol() == Symbol())
                {
                    if (m_position.Magic() == iMagicNumber)
                    {
                        if (m_position.Type() == POSITION_TYPE_BUY || m_position.Type() == POSITION_TYPE_SELL)
                        {
                            double openPrice = m_position.PriceOpen();
                            double lotSize = m_position.Volume();
                            ulong ticket = m_position.Ticket();

                            // Check if stop loss needs to be adjusted
                            if (stopLoss != m_position.PriceStopLoss())
                            {
                                trade.PositionModify(ticket, openPrice, stopLoss, m_position.PriceTakeProfit(), m_position.Expiration());
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        // No open positions, check if it's time to open a new position
        double positionSize = CalculatePositionSize();

        if (positionSize > 0)
        {
            if (currentPrice > iStartPrice + iStepSize)
            {
                trade.PositionOpen(Symbol(), POSITION_TYPE_BUY, positionSize, currentPrice, stopLoss, 0, "Open", iMagicNumber);
                iStartPrice = currentPrice;
            }
            else if (currentPrice < iStartPrice - iStepSize)
            {
                trade.PositionOpen(Symbol(), POSITION_TYPE_SELL, positionSize, currentPrice, stopLoss, 0, "Open", iMagicNumber);
                iStartPrice = currentPrice;
            }
        }
    }
}

void OnDeinit(const int reason)
{
    Comment("Expert terminated with reason:", reason);
}
