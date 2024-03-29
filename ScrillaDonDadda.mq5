#include <Trade\PositionInfo.mqh>
#include <Trade\Trade.mqh>

CPositionInfo m_position;
CTrade trade;

enum ENUM_ST
{
    Average = 0, // Average
    PartClose = 1 // Part Close
};

double AccountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
input double iRiskPerTrade = 0.04;
input double iStartLots = 0.01;
input double iMaximalLots = 2.5;
input double iMinimalProfit = 70.0;
input double iTakeProfit = 140;
enum ENUM_CLOSE_ORDER
{
    CLOSE_ORDER_NONE = 0, // None
    CLOSE_ORDER_AVERAGE = 1, // Average
    CLOSE_ORDER_PARTIAL = 2 // Partial Close
};

int iCloseOrder = 0;
input double iStopLossDistance = 30.0;
int iMagicNumber = 5050;
int iSlippage = 15;

int OnInit()
{
    Comment("");
    trade.LogLevel(LOG_LEVEL_ERRORS);
    trade.SetExpertMagicNumber(iMagicNumber);
    trade.SetDeviationInPoints(iSlippage);
    trade.SetMarginMode();
    trade.SetTypeFillingBySymbol(Symbol());

    return INIT_SUCCEEDED;
}

double CalculatePositionSize()
{
    double accountEquity = AccountInfoDouble(ACCOUNT_EQUITY);
    double maxRiskPerTrade = accountEquity * iRiskPerTrade;
    double stopLossDistance = iStopLossDistance * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    double positionSize = MathMin(maxRiskPerTrade / stopLossDistance, iMaximalLots);
    return positionSize;
}

double CalculateTakeProfit(double openPrice)
{
    double takeProfit = openPrice + iTakeProfit * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    return takeProfit;
}

bool CheckVolumeValue(double volume)
{
    double minVolume = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
    if (volume < minVolume)
        return false;

    double maxVolume = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
    if (volume > maxVolume)
        return false;

    double volumeStep = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
    int ratio = (int)MathRound(volume / volumeStep);
    if (MathAbs(ratio * volumeStep - volume) > 0.0000001)
        return false;

    return true;
}

void OnTick()
{
    double iPointOrderStep = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    // Check margin level
    double marginLevel = AccountInfoDouble(ACCOUNT_MARGIN_LEVEL);

    if (marginLevel >= 20)
    {
        // Margin level reached, do not place new trades
        return;
    }

    double entryPrice = SymbolInfoDouble(Symbol(), SYMBOL_BID); // Stop loss and take profit functionality
    double stopLoss = 20; // Set stop loss value
    double takeProfit = CalculateTakeProfit(entryPrice);

    double buyPriceMax = 0, buyPriceMin = 0, buyPriceMaxLot = 0, buyPriceMinLot = 0;
    double selPriceMin = 0, selPriceMax = 0, selPriceMinLot = 0, selPriceMaxLot = 0;
    ulong buyPriceMaxTic = 0, buyPriceMinTic = 0, selPriceMaxTic = 0, selPriceMinTic = 0;

    double op = 0, lt = 0, tp = 0;
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
                    if (m_position.PositionType() == POSITION_TYPE_BUY || m_position.PositionType() == POSITION_TYPE_SELL)
                    {
                        op = NormalizeDouble(m_position.PriceOpen(), Digits());
                        lt = NormalizeDouble(m_position.Volume(), 2);
                        tk = m_position.Ticket();

                        if (m_position.PositionType() == POSITION_TYPE_BUY)
                        {
                            b++;
                            if (op > buyPriceMax || buyPriceMax == 0)
                            {
                                buyPriceMax = op;
                                buyPriceMaxLot = lt;
                                buyPriceMaxTic = tk;
                            }
                            if (op < buyPriceMin || buyPriceMin == 0)
                            {
                                buyPriceMin = op;
                                buyPriceMinLot = lt;
                                buyPriceMinTic = tk;
                            }
                        }

                        if (m_position.PositionType() == POSITION_TYPE_SELL)
                        {
                            s++;
                            if (op > selPriceMax || selPriceMax == 0)
                            {
                                selPriceMax = op;
                                selPriceMaxLot = lt;
                                selPriceMaxTic = tk;
                            }
                            if (op < selPriceMin || selPriceMin == 0)
                            {
                                selPriceMin = op;
                                selPriceMinLot = lt;
                                selPriceMinTic = tk;
                            }
                        }
                    }
                }
            }
        }
    }

    double buyLot = 0, selLot = 0;
    if (buyPriceMinLot == 0)
        buyLot = iStartLots;
    else
        buyLot = buyPriceMinLot * 2;

    if (selPriceMaxLot == 0)
        selLot = iStartLots;
    else
        selLot = selPriceMaxLot * 2;

    if (iMaximalLots > 0)
    {
        if (buyLot > iMaximalLots)
            buyLot = iMaximalLots;
        if (selLot > iMaximalLots)
            selLot = iMaximalLots;
    }

    if (!CheckVolumeValue(buyLot) || !CheckVolumeValue(selLot))
        return;

    MqlRates rates[];
    if (CopyRates(_Symbol, PERIOD_CURRENT, 0, 2, rates) != 2)
    {
        Print("CopyRates failed, error =", GetLastError());
        return;
    }

    MqlTick tick;
    if (!SymbolInfoTick(_Symbol, tick))
    {
        Print("SymbolInfoTick() failed, error =", GetLastError());
        return;
    }

    if (rates[1].close > rates[1].open)
    {
        if ((b == 0) || (b > 0 && (buyPriceMin - tick.ask) > (iPointOrderStep * Point())))
        {
            if (!trade.Buy(NormalizeDouble(buyLot, 2)))
                Print("OrderSend error #", GetLastError());
        }
    }

    if (rates[1].close < rates[1].open)
    {
        if ((s == 0) || (s > 0 && (tick.bid - selPriceMax) > (iPointOrderStep * Point())))
        {
            if (!trade.Sell(NormalizeDouble(selLot, 2)))
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
                    if (m_position.PositionType() == POSITION_TYPE_BUY || m_position.PositionType() == POSITION_TYPE_SELL)
                    {
                        op = NormalizeDouble(m_position.PriceOpen(), Digits());
                        tp = NormalizeDouble(m_position.TakeProfit(), Digits());
                        lt = NormalizeDouble(m_position.Volume(), 2);
                        tk = m_position.Ticket();

                        if (m_position.PositionType() == POSITION_TYPE_BUY && b == 1 && tp == 0)
                        {
                            if (!trade.PositionModify(tk, m_position.StopLoss(), takeProfit))
                                Print("OrderModify error #", GetLastError());
                        }

                        if (m_position.PositionType() == POSITION_TYPE_SELL && s == 1 && tp == 0)
                        {
                            if (!trade.PositionModify(tk, m_position.StopLoss(), takeProfit))
                                Print("OrderModify error #", GetLastError());
                        }
                    }
                }
            }
        }
    }

    double averageBuyPrice = 0, averageSelPrice = 0;

    if (iCloseOrder == Average)
    {
        if (b >= 2)
            averageBuyPrice = NormalizeDouble((buyPriceMax * buyPriceMaxLot + buyPriceMin * buyPriceMinLot) / (buyPriceMaxLot + buyPriceMinLot) + iMinimalProfit * Point(), Digits());
        if (s >= 2)
            averageSelPrice = NormalizeDouble((selPriceMax * selPriceMaxLot + selPriceMin * selPriceMinLot) / (selPriceMaxLot + selPriceMinLot) - iMinimalProfit * Point(), Digits());
    }
    if (iCloseOrder == PartClose)
    {
        if (b >= 2)
            averageBuyPrice = NormalizeDouble((buyPriceMax * iStartLots + buyPriceMin * buyPriceMinLot) / (iStartLots + buyPriceMinLot) + iMinimalProfit * Point(), Digits());
        if (s >= 2)
            averageSelPrice = NormalizeDouble((selPriceMax * selPriceMaxLot + selPriceMin * iStartLots) / (selPriceMaxLot + iStartLots) - iMinimalProfit * Point(), Digits());
    }

    if (iCloseOrder == PartClose)
    {
        if (b >= 2)
        {
            if (averageBuyPrice > 0 && tick.bid >= averageBuyPrice)
            {
                if (!trade.PositionClosePartial(buyPriceMaxTic, iStartLots, iSlippage))
                    Print("OrderClose Error ", GetLastError());
                if (!trade.PositionClose(buyPriceMinTic, iSlippage))
                    Print("OrderClose Error ", GetLastError());
            }
        }
        if (s >= 2)
        {
            if (averageSelPrice > 0 && tick.ask <= averageSelPrice)
            {
                if (!trade.PositionClosePartial(selPriceMinTic, iStartLots, iSlippage))
                    Print("OrderClose Error ", GetLastError());
                if (!trade.PositionClose(selPriceMaxTic, iSlippage))
                    Print("OrderClose Error ", GetLastError());
            }
        }
    }
}