#include <PositionInfo.mqh>
CPositionInfo m_position;
#include <Trade.mqh>
CTrade trade;

#include <SmoothAlgorithms.mqh>
#include <RSI.mqh>
#include <BollingerBands.mqh>

enum ENUM_ST
{
   Average = 0, // Average
   PartClose = 1 // Part Close
};

//  Variable Declarations
double iStartLots = 0.01;  // Initial lot size
double iMaxRiskPerTrade = 0.03;  // Maximum risk per trade (1% of account balance)
double iMinLotSize = 0.01;  //  lot size
double iMaxLotSize = 10.00;  //Minimum Maximum lot size
double iMaxStopLossDistance = 30.0;  // Maximum stop loss distance in pips

//  Lot Size Calculation based on Market Volatility and Account Balance
double CalculateLotSize()
{
    // Calculate lot size based on desired risk per trade
    double accountBalance = AccountBalance();
    double riskAmount = accountBalance * iMaxRiskPerTrade;

    // Calculate stop loss distance in points
    double stopLossDistance = iMaxStopLossDistance * Point();

    // Calculate lot size
    double lotSize = (riskAmount / stopLossDistance) / 10000.0;

    // Limit lot size within the defined range
    lotSize = MathMax(iMinLotSize, MathMin(iMaxLotSize, lotSize));

    return NormalizeDouble(lotSize, 2);
}

// Adjust Stop Loss based on Market Volatility
double AdjustStopLoss(double stopLoss)
{
double
BuyPriceMax = 0, BuyPriceMin = 0, BuyPriceMaxLot = 0, BuyPriceMinLot = 0;
double
SelPriceMin = 0, SelPriceMax = 0, SelPriceMinLot = 0, SelPriceMaxLot = 0;
ulong
BuyPriceMaxTic = 0, BuyPriceMinTic = 0, SelPriceMaxTic = 0, SelPriceMinTic = 0;
double
op = 0, lt = 0, tp = 0;
ulong
tk = 0;
int
b = 0, s = 0;

int
total = PositionsTotal();
for (int k = total - 1; k >= 0; k--)
    {
    if (m_position.SelectByIndex(k))
    {
    if (m_position.Symbol() == Symbol())
    {
    if (m_position.Magic() == iMagicNumber)
    {
    if (m_position.PositionType() == POSITION_TYPE_BUY | | m_position.PositionType() == POSITION_TYPE_SELL)
    {
        op = NormalizeDouble(m_position.PriceOpen(), Digits());
    lt = NormalizeDouble(m_position.Volume(), 2);
    tk = m_position.Ticket();

    if (m_position.PositionType() == POSITION_TYPE_BUY)
    {
    b++;
    if (op > BuyPriceMax | | BuyPriceMax == 0)
    {
    BuyPriceMax = op;
    BuyPriceMaxLot = lt;
    BuyPriceMaxTic = tk;
    }
    if (op < BuyPriceMin | | BuyPriceMin == 0)
    {
    BuyPriceMin = op;
    BuyPriceMinLot = lt;
    BuyPriceMinTic = tk;
    }
    }
    // == =
    if (m_position.PositionType() == POSITION_TYPE_SELL)
    {
    s++;
    if (op > SelPriceMax | | SelPriceMax == 0)
    {
    SelPriceMax = op;
    SelPriceMaxLot = lt;
    SelPriceMaxTic = tk;
    }
    if (op < SelPriceMin | | SelPriceMin == 0)
    {
    SelPriceMin = op;
    SelPriceMinLot = lt;
    SelPriceMinTic = tk;
    }
    }
    }
    }
    }
    }
    }

    return stopLoss;
}

double
BuyLot = 0, SelLot = 0;
if (BuyPriceMinLot == 0)
    BuyLot = iStartLots;
else
    BuyLot = BuyPriceMinLot * 2;
if (SelPriceMaxLot == 0)
    SelLot = iStartLots;
else
    SelLot = SelPriceMaxLot * 2;

if (iMaximalLots > 0)
    {
    if (BuyLot > iMaximalLots)
    BuyLot = iMaximalLots;
    if (SelLot > iMaximalLots)
    SelLot = iMaximalLots;
    }
if (!CheckVolumeValue(BuyLot) | | !CheckVolumeValue(SelLot))
return;

MqlRates
rates[];
CopyRates(Symbol(), PERIOD_CURRENT, 0, 2, rates);

MqlTick
tick;
if (!SymbolInfoTick(Symbol(), tick))
Print("SymbolInfoTick() failed, error = ", GetLastError());

    if (rates[1].close > rates[1].open)
    {
        if ((b == 0) || (b > 0 && (BuyPriceMin - tick.ask) > (iPointOrderStep * Point())))
        {
            double adjustedStopLoss = AdjustStopLoss(BuyPriceMin - iStopLoss * Point());

            if (!trade.Buy(NormalizeDouble(BuyLot, 2), 0, 0, 0, 0, adjustedStopLoss))
                Print("OrderSend error #", GetLastError());
        }
    }

    if (rates[1].close < rates[1].open)
    {
        if ((s == 0) || (s > 0 && (tick.bid - SelPriceMax) > (iPointOrderStep * Point())))
        {
            double adjustedStopLoss = AdjustStopLoss(SelPriceMax + iStopLoss * Point());

            if (!trade.Sell(NormalizeDouble(SelLot, 2), 0, 0, 0, 0, adjustedStopLoss))
                Print("OrderSend error #", GetLastError());
        }
    }

        for (int k = total - 1; k >= 0; k--)
            {
            if (m_position.SelectByIndex(k))
            {
            if (m_position.Symbol() == Symbol())
            {
            if (m_position.Magic() == iMagicNumber)
            {
            if (m_position.PositionType() == POSITION_TYPE_BUY | | m_position.PositionType() == POSITION_TYPE_SELL)
            {
            op = NormalizeDouble(m_position.PriceOpen(), Digits());
            tp = NormalizeDouble(m_position.TakeProfit(), Digits());
            lt = NormalizeDouble(m_position.Volume(), 2);
            tk = m_position.Ticket();

            if (m_position.PositionType() == POSITION_TYPE_BUY & & b == 1 & & tp == 0)
            {
            if (!trade.PositionModify(tk, m_position.StopLoss(), NormalizeDouble(tick.ask + iTakeProfit * Point(), Digits())))
            Print("OrderModify error #", GetLastError());
            }

            if (m_position.PositionType() == POSITION_TYPE_SELL & & s == 1 & & tp == 0)
            {
            if (!trade.PositionModify(tk, m_position.StopLoss(), NormalizeDouble(tick.bid - iTakeProfit * Point(), Digits())))
            Print("OrderModify error #", GetLastError());
            }

            if (iCloseOrder == Average)
            {
            if (m_position.PositionType() == POSITION_TYPE_BUY & & b >= 2)
            {
            if (tk == BuyPriceMaxTic | | tk == BuyPriceMinTic)
            {
            if (tick.bid < AwerageBuyPrice & & tp != AwerageBuyPrice)
            {
            if (!trade.PositionModify(tk, m_position.StopLoss(), AwerageBuyPrice))
            Print("OrderModify error #", GetLastError());
            }
            }

            if (tk != BuyPriceMaxTic & & tk != BuyPriceMinTic & & tp != 0)
            {
            if (!trade.PositionModify(tk, 0, 0))
            Print("OrderModify error #", GetLastError());
            }
            }

            if (m_position.PositionType() == POSITION_TYPE_SELL & & s >= 2)
            {
            if (tk == SelPriceMaxTic | | tk == SelPriceMinTic)
            {
            if (tick.ask > AwerageSelPrice & & tp != AwerageSelPrice)
            {
            if (!trade.PositionModify(tk, m_position.StopLoss(), AwerageSelPrice))
            Print("OrderModify error #", GetLastError());
            }
            }

            if (tk != SelPriceMaxTic & & tk != SelPriceMinTic & & tp != 0)
            {
            if (!trade.PositionModify(tk, 0, 0))
            Print("OrderModify error #", GetLastError());
            }
            }
            }
            }

            if (iCloseOrder == PartClose)
            {
            if (b >= 2)
            {
            if (AwerageBuyPrice > 0 & & tick.bid >= AwerageBuyPrice)
            {
            if (!trade.PositionClosePartial(BuyPriceMaxTic, iStartLots, iSlippage))
            Print("OrderClose Error ", GetLastError());
            if (!trade.PositionClose(BuyPriceMinTic, iSlippage))
            Print("OrderClose Error ", GetLastError());
            }
            }
            if (s >= 2)
            {
            if (AwerageSelPrice > 0 & & tick.ask <= AwerageSelPrice)
            {
            if (!trade.PositionClosePartial(SelPriceMinTic, iStartLots, iSlippage))
            Print("OrderClose Error ", GetLastError());
            if (!trade.PositionClose(SelPriceMaxTic, iSlippage))
            Print("OrderClose Error ", GetLastError());
            }
            }
            }
            }
            }
            }
            }
 }

        // ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** /
        // * * /
        // ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** /
        void
        OnDeinit(const
        int
        reason)
        {

        }
        // ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** /
        // * * /
        // ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** /
        bool
        CheckVolumeValue(double
        volume)
        {
        // --- минимально
        допустимый
        объем
        для
        торговых
        операций
        double
        min_volume = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
        if (volume < min_volume)
            return (false);

        // --- максимально
        допустимый
        объем
        для
        торговых
        операций
        double
        max_volume = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
        if (volume > max_volume)
            return (false);

        // --- получим
        минимальную
        градацию
        объема
        double
        volume_step = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);

        int
        ratio = (int)
        MathRound(volume / volume_step);
        if (MathAbs(ratio * volume_step - volume) > 0.0000001)
            return (false);

        return (true);
        }

    * ** ** ** ** ** ** ** ** **     // *** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** /