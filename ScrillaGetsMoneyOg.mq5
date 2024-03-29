#include <Trade\PositionInfo.mqh>
#include <Trade\Trade.mqh>

class ExpertAdvisor
{
private:
   CPositionInfo m_position;
   CTrade trade;
   double iRiskPerTrade;
   double iStartLots;
   double iMaximalLots;
   double iMinimalProfit;
   double iPointOrderStep;
   double iTakeProfit;
   double iStopLossDistance;
   int iMagicNumber;
   int iSlippage;

public:
   ExpertAdvisor()
   {
      iRiskPerTrade = 0.04;
      iStartLots = 0.01;
      iMaximalLots = 1.0;
      iMinimalProfit = 70.0;
      iPointOrderStep = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      iTakeProfit = 200.0;
      iStopLossDistance = 30.0;
      iMagicNumber = 5050;
      iSlippage = 5;
   }

   bool OnInit()
   {
      Comment("");
      trade.LogLevel(LOG_LEVEL_ERRORS);
      trade.SetExpertMagicNumber(iMagicNumber);
      trade.SetDeviationInPoints(iSlippage);
      trade.SetMarginMode();
      trade.SetTypeFillingBySymbol(Symbol());

      return (true);
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
      ulong iCloseOrder = 0;
      double entryPrice = SymbolInfoDouble(Symbol(), SYMBOL_BID);
      double StopLoss = 50;
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
                        if (op > BuyPriceMax)
                        {
                           BuyPriceMax = op;
                           BuyPriceMaxTic = tk;
                           BuyPriceMaxLot = lt;
                        }

                        if (op < BuyPriceMin || BuyPriceMin == 0)
                        {
                           BuyPriceMin = op;
                           BuyPriceMinTic = tk;
                           BuyPriceMinLot = lt;
                        }
                     }
                     else if (m_position.PositionType() == POSITION_TYPE_SELL)
                     {
                        if (op < SelPriceMin || SelPriceMin == 0)
                        {
                           SelPriceMin = op;
                           SelPriceMinTic = tk;
                           SelPriceMinLot = lt;
                        }

                        if (op > SelPriceMax)
                        {
                           SelPriceMax = op;
                           SelPriceMaxTic = tk;
                           SelPriceMaxLot = lt;
                        }
                     }
                  }
               }
            }
         }
      }

      if (total > 0)
      {
         if (BuyPriceMax > 0)
         {
            if (entryPrice - BuyPriceMax > iPointOrderStep)
            {
               if (entryPrice - BuyPriceMax > iStopLossDistance * SymbolInfoDouble(_Symbol, SYMBOL_POINT))
               {
                  iCloseOrder = BuyPriceMaxTic;
                  b++;
               }
            }
         }

         if (BuyPriceMin > 0)
         {
            if (entryPrice - BuyPriceMin < -iPointOrderStep)
            {
               if (entryPrice - BuyPriceMin < -iStopLossDistance * SymbolInfoDouble(_Symbol, SYMBOL_POINT))
               {
                  iCloseOrder = BuyPriceMinTic;
                  b++;
               }
            }
         }

         if (SelPriceMin > 0)
         {
            if (entryPrice - SelPriceMin < -iPointOrderStep)
            {
               if (entryPrice - SelPriceMin < -iStopLossDistance * SymbolInfoDouble(_Symbol, SYMBOL_POINT))
               {
                  iCloseOrder = SelPriceMinTic;
                  s++;
               }
            }
         }

         if (SelPriceMax > 0)
         {
            if (entryPrice - SelPriceMax > iPointOrderStep)
            {
               if (entryPrice - SelPriceMax > iStopLossDistance * SymbolInfoDouble(_Symbol, SYMBOL_POINT))
               {
                  iCloseOrder = SelPriceMaxTic;
                  s++;
               }
            }
         }

         if (iCloseOrder != 0)
         {
            trade.PositionClose(iCloseOrder);
         }
      }

      if (b == 0)
      {
         if (iCloseOrder == 0)
         {
            if (entryPrice - BuyPriceMax > iPointOrderStep)
            {
               double lots = CalculatePositionSize();
               trade.Buy(lots);
            }
         }
      }

      if (s == 0)
      {
         if (iCloseOrder == 0)
         {
            if (entryPrice - SelPriceMax < -iPointOrderStep)
            {
               double lots = CalculatePositionSize();
               trade.Sell(lots);
            }
         }
      }
   }
};

ExpertAdvisor g_expert;

int OnInit()
{
   return g_expert.OnInit() ? INIT_SUCCEEDED : INIT_FAILED;
}

void OnTick()
{
   g_expert.OnTick();
}
