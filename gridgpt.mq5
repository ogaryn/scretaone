#include <Trade\PositionInfo.mqh>
CPositionInfo m_position;
#include <Trade\Trade.mqh>
CTrade trade;

enum ENUM_ST
{
   Average = 0, // Average
   PartClose = 1 // Part Close
};

double AccountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
input double iRiskPerTrade = 0.05;
input double iStartLots = 0.01;
input double iMaximalLots = 50.00;
input double iMinimalProfit = 70.0;
double iPointOrderStep = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
input double iTakeProfit = 100;

enum ENUM_CLOSE_ORDER
{
   CLOSE_ORDER_NONE = 0, // None
   CLOSE_ORDER_AVERAGE = 1, // Average
   CLOSE_ORDER_PARTIAL = 2 // Partial Close
};

int iCloseOrder = CLOSE_ORDER_NONE;
input double iStopLossDistance = 30.0;
int iMagicNumber = 5050;
int iSlippage = 5;

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

double CalculateTakeProfit(double openPrice, double riskPerTrade)
{
    double takeProfit = openPrice + (openPrice * riskPerTrade);
    return takeProfit;
}

void OnTick()
{
    double entryPrice = SymbolInfoDouble(Symbol(), SYMBOL_BID);
    double StopLoss = 20;
    double TakeProfit = CalculateTakeProfit(entryPrice, iRiskPerTrade);
    double BuyPriceMax = 0, BuyPriceMin = 0, BuyPriceMaxLot = 0, BuyPriceMinLot = 0;
    double SelPriceMin = 0, SelPriceMax = 0, SelPriceMinLot = 0, SelPriceMaxLot = 0;
    ulong BuyPriceMaxTic = 0, BuyPriceMinTic = 0, SelPriceMaxTic = 0, SelPriceMinTic = 0;
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
                            if (op > BuyPriceMax || BuyPriceMax == 0)
                            {
                                BuyPriceMax = op;
                                BuyPriceMaxLot = lt;
                                BuyPriceMaxTic = tk;
                            }
                            if (op < BuyPriceMin || BuyPriceMin == 0)
                            {
                                BuyPriceMin = op;
                                BuyPriceMinLot = lt;
                                BuyPriceMinTic = tk;
                            }
                        }
                        else if (m_position.PositionType() == POSITION_TYPE_SELL)
                        {
                            s++;
                            if (op < SelPriceMin || SelPriceMin == 0)
                            {
                                SelPriceMin = op;
                                SelPriceMinLot = lt;
                                SelPriceMinTic = tk;
                            }
                            if (op > SelPriceMax || SelPriceMax == 0)
                            {
                                SelPriceMax = op;
                                SelPriceMaxLot = lt;
                                SelPriceMaxTic = tk;
                            }
                        }
                    }
                }
            }
        }
    }

    if (b > 0)
    {
        if (entryPrice < BuyPriceMin - iPointOrderStep * 5)
        {
            iCloseOrder = CLOSE_ORDER_AVERAGE;
            Comment("Buy Close Order Average");
            for (int i = total - 1; i >= 0; i--)
            {
                if (m_position.SelectByIndex(i))
                {
                    if (m_position.Symbol() == Symbol())
                    {
                        if (m_position.Magic() == iMagicNumber)
                        {
                            if (m_position.PositionType() == POSITION_TYPE_BUY)
                            {
                                if (m_position.Ticket() != BuyPriceMinTic)
                                {
                                   if (!trade.PositionClose(BuyPriceMinTic, (ulong)lt))

                                    {
                                        Print("Close Error ", GetLastError());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (entryPrice > BuyPriceMax + iPointOrderStep * 5)
        {
            iCloseOrder = CLOSE_ORDER_AVERAGE;
            Comment("Buy Close Order Average");
            for (int i = total - 1; i >= 0; i--)
            {
                if (m_position.SelectByIndex(i))
                {
                    if (m_position.Symbol() == Symbol())
                    {
                        if (m_position.Magic() == iMagicNumber)
                        {
                            if (m_position.PositionType() == POSITION_TYPE_BUY)
                            {
                                if (m_position.Ticket() != BuyPriceMaxTic)
                                {
                                    if (!trade.PositionClose(BuyPriceMaxTic, (ulong)lt))
                                    {
                                        Print("Close Error ", GetLastError());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (s > 0)
    {
        if (entryPrice > SelPriceMax + iPointOrderStep * 5)
        {
            iCloseOrder = CLOSE_ORDER_AVERAGE;
            Comment("Sell Close Order Average");
            for (int i = total - 1; i >= 0; i--)
            {
                if (m_position.SelectByIndex(i))
                {
                    if (m_position.Symbol() == Symbol())
                    {
                        if (m_position.Magic() == iMagicNumber)
                        {
                            if (m_position.PositionType() == POSITION_TYPE_SELL)
                            {
                                if (m_position.Ticket() != SelPriceMaxTic)
                                {
                                   if (!trade.PositionClose(SelPriceMaxTic, (ulong)lt))
                                    {
                                        Print("Close Error ", GetLastError());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (entryPrice < SelPriceMin - iPointOrderStep * 5)
        {
            iCloseOrder = CLOSE_ORDER_AVERAGE;
            Comment("Sell Close Order Average");
            for (int i = total - 1; i >= 0; i--)
            {
                if (m_position.SelectByIndex(i))
                {
                    if (m_position.Symbol() == Symbol())
                    {
                        if (m_position.Magic() == iMagicNumber)
                        {
                            if (m_position.PositionType() == POSITION_TYPE_SELL)
                            {
                                if (m_position.Ticket() != SelPriceMinTic)
                                {
                                    if (!trade.PositionClose(SelPriceMinTic, (ulong)lt))
                                    {
                                        Print("Close Error ", GetLastError());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    double positionSize = CalculatePositionSize();

    if (total == 0 && AccountBalance >= iMinimalProfit)
    {
        if (entryPrice > SelPriceMin + iPointOrderStep * 5 && b == 0)
        {
            trade.Buy(positionSize, NULL, entryPrice, iStopLossDistance, iTakeProfit, "");
        }
        else if (entryPrice < BuyPriceMax - iPointOrderStep * 5 && s == 0)
        {
            trade.Sell(positionSize, NULL, entryPrice, iStopLossDistance, iTakeProfit, "");
        }
    }

    if (iCloseOrder == CLOSE_ORDER_PARTIAL)
    {
        if (total > 0 && entryPrice < BuyPriceMax - iPointOrderStep * 5 && s == 0)
        {
            trade.Sell(positionSize, NULL, entryPrice, iStopLossDistance, iTakeProfit, "");
        }
        else if (total > 0 && entryPrice > SelPriceMin + iPointOrderStep * 5 && b == 0)
        {
            trade.Buy(positionSize, NULL, entryPrice, iStopLossDistance, iTakeProfit, "");
        }
    }
}

void OnDeinit(const int reason)
{
   return;
}
