//+------------------------------------------------------------------+
//|                                                Generic A-TLP.mq4 |
//|                                                 Trade Like A Pro |
//|                                          http://tradelikeapro.ru |
//|                                                      open source |
//|   ������ 9.5  - filters have been added to the CCI                         |
//|   ������ 10.0 - held small code optimization             |
//|   ������ 11.1 - ��������� ��������� ������, ������ ��������,     |
//|               �������������� �������� ������ � ���� �������      |
//|               � ������� ��������� ���������� �����               |
//|   ������ 11.2 - ��� ������������� ��������� � ���� ��������      |
//|               � ����� ����� ����������� ��� ������;              |
//|               - �������� ������� ������� StDev ������� �� �������|
//|               ����������                                         |
//|   ������ 11.3 - ���������� ������, ��������� � CCI               |
//|   ������ 11.4 - ���������� ������ ������-������;                 |
//|               - ��������� ����������� �������� �� ������ ����;   |
//|               - �������� �� ����������� ������������� ������     |
//|               �������������� Time to ����� � ���� �������� �������   |
//|   ������ 11.6 - �������� �������� ����                           |
//|   ������ 11.7 - ���������� ������                                |
//|   ������ 11.7.31 - ������ ������ ������� �� ��15                 |
//|   ������ 11.9 - ��������� ����� ����� ��������� ������,          |
//|               - ������ �� ����. ����� �����,                     |
//|               - ���������� ���� �� �������, � �� �� ������,      |
//|               - ����������� ������� ������ � ��������,           |
//|               - Time to�� � ������� � ������� ������� Time to �������    |
//|               ������                                             |
//|   ������ 11.9.1 - ������� ������� ������������ (����� �����      |
//|               ������ � ����.�����)                               |
//|               - �������� ��� ������ ������ ������� lastloss (��� |
//|               ������� � ��� ��������)                            |
//|               - ������������� ����������� (���������� LogMode)   |
//|               - ��������� ����������� ����                       |
//|   ������ 11.9.2 - ����������� ��������������                     |
//|   ������ 11.9.3 - ���������� ������                              |
//|   ������ 11.9.4 - ��������� ���������� ����                      |
//|   ������ 11.10.6 - ������� ��� ������ ������� �������� ���       |
//|                    ����������� ��� �����������.                  |
//|                  - ������ ������(�����������) ������� ���������. |
//|                    ������ ������� ����������� Time to �������� 0.    |
//|                  - ���������� ������� ����������                 | 
//|   ������ 11.10.7 - ��������� ������ Auto_Risk                    |  
//|                  - �������� ������ ����������                    |
//|   ������ 11.10.8 - ������ ������                                 | 
//|   ������ 11.10.9 - ������ ������ ����������                      |
//|                  - ����������������� ��������� ������� �� �����  |
//|   ������ 12.01.14 -����������� ����, ��������� ��� �����������   |
//|                  - �������������� ������, ���������              |
//|                     every_tick ������ ��������� ���� 1 ���       |
//|                     �� ������ � ������ �������� ������ ���� �� �1|
//|   ������ 12.01.15 - ������� MaxDailyRange � ��������� �          |
//|                     every_tick ������ ��������� ���� 1 ���       |
//|                     �� ������ � ������ �������� ������ ���� �� �1|
//|   ������ 12.01.16 - ���������� �� �������� � Second Session ���  |
//|                     �������� � First Session                     |
//|   ������ 12.01.17 - ���������� ������ � ��������                 |
//|                   - ������� � ������ ������ CCI ���������� � ����|
//|   ������ 12.01.18 - ���������� ������ ������������  ���������    |
//|                     MaxDailyRange                                |
//|   ������ 12.01.19 - ������ � ��� Time to Time to������� MaxDailyRange �  |
//|                     ������� CCI                                  |
//|                   - BB_Deviation ������ ������������             |
//|   ������ 12.01.20 - ������� ������ ������ ����������� �� ����    |
//|                     Ask Time to �������� ������ ��� �������, � Time to   |
//|                     �������� ��� ������                          |
//|   ������ 12.01.21 - ������ �������� OnlyBid, Time to true ��������   |
//|                     ��������� ��������� � 12.01.20               |
//|                   - �������� ShowCandleDots ������� ��           |
//|                     VisualDebug, ������ ����� � �������������    |
//|                     ��������, � ���������� ������������ �������� |
//|                   - ��������� �������� SetName, ���� ������      |
//|                     ������� ���� � SetName � ���������� ����� �  |
//|                     ����� ������� SetName � [OK] ����� [WARNING] |
//|                   - ����������� ��� ��������� ������             |
//|   ������ 12.01.22 - ���������� ������ � ������������ ������      |
//|                   - Time to Max_Spread=0 ������ �� ��������� ������  |
//|                     NORMAL/HIGH ��� ������                       |
//|                   - ��������� ������� OnTester, ������������     |
//|                     ��������� Time to���� � ������������ ��������    |
//|   ������ 12.01.23 - ��������� RollOver Filter                    |
//|                   - � VisualDebug �������� ������ ����� bb       |
//|                     �� ���������� Exit_Distance, Time to �������     |
//|                     �������� �������, Time to ������ �� ������� �����| 
//|                     �������� �� ������ ������� �����. �������    |
//|                     ������ ������ �� ����, �� �������� ������.   |
//|                   - ��� ��������� � ������� Time to������ �          |
//|                     ������������                                 |
//|   ������ 12.02.24 - ������� ��� ������ ������� ��������          |
//|   ������ 12.02.25 - ����������� ��� ��������� ������             |
//|   ������ 12.26    - ���������� use_rollover_filter �������� ��   |
//|                     OpenOrdersInRollover � CloseOrdersInRollover |
//|                   - ���������� ������� ������� �������� �����    |
//|                     �������                                      |
//|   ������ 12.27    - �������� ����� ��������, �������, ��� 24,    |
//|                     ����������� �� ��������� ����                |
//|   ������ 12.28    - ��������� ����� ������� � ����-������        |
//|   ������ 12.30    - Time to Off������� ������������ ���������       |
//|                     ��������������                               |
//|                   - �������� �������� TradeAllDaysInTheSameTime, |
//|                     ���������� �������� �� ��� ��� � ���� � �� ��|
//|                     ����� (������������ ����� ������������)      |
//|                   - ��������� ��������� ���������/Off������     |
//|                     ������ ��������� ����                        |
//|                   - �������� �������� CheckMDRAfter0, ���������� |
//|                     �������� MaxDailyRange ����� ��������        |
//|                   - ������ ��������������� �� �������� ������ �  |
//|                     �������                                      |
//|                   - �������� �������� MM_Depo, ��� ������ ����   |
//|                     ������ ���� (Lots) �� ������ ���� (MM_Depo)  |
//|   ������ 12.31    - �������� maxcandle ������ ������ ����� ����� |
//|                     ����� �������������� ��� �������� ��������   |
//|                     ���� �� �������� ���������� �����            |
//|                   - ��������� ����� � ����-������                |
//|                   - �������� �� �����������/������������ ���     |
//|                   - ������� VisualDebug ���������� �������       |
//|                   - �������� Max_Spread_mode2 ��� ������������   |
//|                     �������� �������, Time to �������� ������ ����   |
//|                     Max_Spread ��������� ������ ������� Time to      |
//|                     ��������, � Max_Spread_mode2 - ������� Time to   |
//|                     ��������                                     |
//|   ������ 12.32    - ��������� ����� � ������                     |
//|                   - ���������� ������                            |
//|                   - ����������� ��������� ��������� �������      |
//|   ������ 12.33    - �������� ���������� ����� ������ ������      |
//|                   - ��������� ������ �� ��                       |
//|                   - ��������� ������������ ������ ������ �� ��I  |
//|   ������ 12.34    - ���������� ������ � ���������� ������ ������ |
//|                   - ��������� ������ ��������-����� � ��������   |
//|                   - ������ ���������� ������� ������� �����     |
//|                     ����������                                   |
//|   ������ 12.35    - �������� ������ ��� ������������ ����� ������|
//|                     ��������������� ���� MaxAmountCurrencyMult=2 |
//|   ������ 12.36    - �������� Max_Spread_mode2 ������ ��         |
//|                     CheckSpreadOnSellOpen, ���������� ��������   |
//|                     ������ Time to ��������, � Max_Spread_On_Close,  |
//|                     ����������� ����� Time to �������� ������        |
//|                   - �������� �������� ����� ����� �������� Time to   |
//|                     �������� ���������� �������                  |
//|                   - �������� �������� �������������� ��������    |
//|                     ������ �������                               |
//|                   - � ������ ������ ������� ���� ��������        |
//|                     � Time to����                                    |
//|                   - ��������� ����� ��������� � ������           |
//|                     Time to Off������� ������������                 |
//|                   - �������� �� ���������� ������� Time to ��������� |
//|                     �������� ������ ����� ��������� �������      |
//|                   - �������� ������ ��������                     |
//|   ������ 12.37    - ����������� ��������� ������ VisualDebug     |
//|                   - �������� �� ������ Time to ��������� ������      |
//|                     � ������ �����������                         |
//|                   - ������������� ��������� ��� CCI              |
//|   ������ 12.37.1  - ���������� ������ ���������� ���������� �����|
//|                     ��������� ��������                           |
//|                   - ���� �������� �� ������ Time to ��������� ������ |
//|                     � ������ �����������                         |
//|   ������ 12.37.2  - ������ ����� Time to���� ���������� ������� -    |
//|                     ������ ��������� ������� Time to����             |
//|                   - ���������� ������ � ��������������� ���������|
//|                     ���������� �������                           |
//|                   - �������� �������� MDRFromHiLo, ����������    |
//|                     �������� ��������� � ���������/�������� ���  |
//|                   - �������� �������� ������ �������             |
//|                   - MaxDailyRange ������ �������������� � ������ |
//|                     ������ �������                               |
//|   ������ 12.38    - ���������� ���������� �������                |
//|                     Time to Exit_Distance = 0                        |
//|                   - ��������� ������ MaxDailyRange ����� ��������|
//|                   - Time to ���������� ��������� CheckMDRAfter0      |
//|                     �������� � ����������� ��� �� �����������    |
//|                     � �����������                                |
//|                   - ���������� ����������� ������� ������, ����� |
//|                     ������� Time to�������� �� ����� �������� �����  |
//|                   - �������� TimeShift ������ �� GMT_Offset     |
//+------------------------------------------------------------------+

//         author of the idea
//            Sergey5  bashni2001@mail.ru  

//         Senior Programmer
//            yur4ello

//         Huge gratitude to for the help in realization of the idea
//            Alexandr69
//            nixxer
//            grabli
//            LeoK

#property copyright "Trade Like A Pro"
#property link      "http://tradelikeapro.ru"
#property version   "12.38" 
#property strict

#include <stderror.mqh>
#include <..\Libraries\stdlib.mq4>

enum     _EventLogs{
                              ALL                        = 1,                 //All
                              MAIN                       = 2,                 //Error
							         NONE						      = 3				      //Nothing
         };
         
enum     _MaxMin{
         MAX = 1,
         MIN = 2
         };
  
//--------------------------------------
extern   string S_1          = "<==== General settings ====>"; // >   >   >   >   >   >    >    >    >    > 
extern   string               SetName                    = "";                //The Set file name
extern   ENUM_TIMEFRAMES      TimeFrame                  = PERIOD_CURRENT;    //TimeFrame for Bollinger Bands
extern   bool                 every_tick                 = 1;                 //Trade every tick
extern   int                  MagicNumber                = 1234321;
extern   int                  Slippage                   = 5;
extern   double               Lots                       = 0.01;
extern   double               Auto_Risk                  = 0.0;               //Auto Risk
input    bool                 Martingale                 = 1;                 //Use Martingale
input    double               Multiplier                 = 2;                 //Lot Multiplier
input    int                  MM_Depo                    = 0;                 //Depo per Lots
extern   double               MaxAmountCurrencyMult      = 0;                 //MaxAmountCurrencyMult
input    bool                 CheckSpreadOnSellOpen      = true;
extern   double               Max_Spread                 = 0;                 //Max Spread
extern   double               Max_Spread_On_Close        = 0;
extern   double               Stop_Loss                  = 30;                //Stop Loss
extern   double               Take_Profit                = 35;                //Take Profit
extern   int                  TP_perc                    = 80;                //Percentage of TP from the size of the channel
extern   int                  min_TP                     = 10;                //Minimum Take Profit
extern   int                  MaxDailyRange              = 1000;
extern   bool                 MDRFromHiLo                = false;
extern   bool                 CheckMDRAfter0             = false;
extern   bool                 Hedging                    = true;              //Hedging

extern   string S_8          = "<==== MULTI ORDERS ====>";                    // >   >   >   >   >   >    >    >    >    > 
extern   int                  TotalOrders                = 1;
extern   double               OrdersDistance             = 5;
extern   int                  MinPause                   = 0;                 //Minimum time between deals (Sec)
input    bool                 CloseSimultaneously        = false;             //Close Simultaneously

extern   string S_2          = "<==== ENTER SETTINGS ====>";                  // >   >   >   >   >   >    >    >    >    > 
extern   string S_21         = "Bollinger Bands Setting";                    //Bollinger Band
extern   int                  BB_Period                  = 13;                //BB: Period
extern   double               BB_Deviation               = 2;                 //BB: Deviation
extern   int                  Entry_Break                = 1;                 //BB: Delta Channel Borders
extern   int                  Min_Volatility             = 20;                //BB: The minimum width of the channel
input    bool                 OnlyBid                    = true;              //BB: Check if the channel touches the Bid price

extern   string S_22         = "CCI Setting";                               //CCI
input    ENUM_TIMEFRAMES      TimeFrame_CCI              = PERIOD_CURRENT;    //CCI: TimeFrame
extern   int                  cci_Period_open            = 14;                //CCI: Period
extern   int                  cci_level_open             = 100;               //CCI: Upper and Lower level

extern   string S_23         = "Maximum candle Height= 0 - Off.";            //Filter by Candle size (on input)
extern   double               maxcandle                  = 100;               //Maximum candle Height
extern   int                  barcount                   = 8;                 //Number of bars for the Maximum Candle
extern   string S_24         = "Pause Timing = 0 - Off.";                       //Filter based on loss of previous orders
extern   int                  pause                      = 120;               //Pausing time after a losing deal (minutes)
extern   double               sizeloss                   = 60;                // Loss size to enable pausing (in pips)

extern   string S_4          = "<==== Trailing stop/Breakeven ====>";         // >   >   >   >   >   >    >    >    >    > 
extern   double               Trail_Start                = 0;                 //Trailling Start
extern   double               Trail_Size                 = 7;                 //Trailling Distance
extern   double               Trail_Step                 = 1;                 //Trailling Step
extern   bool                 rollover_trall_end         = true;              //Ban Trall in Rollover
extern   double               X                          = 0;                 //BE: If the current price is better than entry price on pips
extern   double               Y                          = 1;                 //BE: Move the SL Y points in profit

extern   string S_3          = "<==== EXIT SETTINGS ====>";                   // >   >   >   >   >   >    >    >    >    > 
extern   string S_31         = "Exit Time_Minutes = 0 - Off.";                //1. Time Filter. 
extern   int                  Exit_Minutes               = 140;
extern   int                  Time_Profit_Pips           = 5;
extern   string S_32         = "Exit_Distance > 100 - Off.";               //2. Channel Filter. 
extern   int                  Exit_Distance              = -13;
extern   int                  Exit_Profit_Pips           = -12;
extern   string S_33         = "MA_period = 0 - Off.";                   //3. MA Filter.
extern   ENUM_TIMEFRAMES      TimeFrameMA                = PERIOD_M1; 
extern   ENUM_MA_METHOD       MA_type                    = MODE_SMA;
         ENUM_APPLIED_PRICE   MA_price                   = PRICE_CLOSE;
extern   int                  MA_period                  = 2;
extern   int                  Reverse_Profit             = 20;
extern   string S_34         = "Time to CCI: Period = 0 - Off.";                 //4. CCI Filter. 
extern   int                  cci_Period_close           = 0;                 //CCI: Period
extern   int                  cci_level_close            = 100;               //CCI: Upper and Lower level
extern   int                  CCI_Profit_Pips            = 20;

extern   string S_5          = "<==== Trade Time Filter ====>";               // >   >   >   >   >   >    >    >    >    > 
extern   int                  GMT_Offset                 = 2;                 //GMT Offset
extern   bool                 DST                        = true;              //DaylightSavingsTime
extern   bool                 TradeAllDaysInTheSameTime  = false;             //Trade once everyday
extern   string S_51          = "<== MONDAY / ALL DAYS ==>";        // MONDAY 
extern   bool                 MONDAY_Enabled             = true;              //Trading on Mondays
extern   int                  MONDAY_Start_Trade_Hour    = 22;                //Start Trade Hour
extern   int                  MONDAY_Start_Trade_Minute  = 0;                 //Start Trade Minute
extern   int                  MONDAY_End_Trade_Hour      = 1;                 //End Trade Hour
extern   int                  MONDAY_End_Trade_Minute    = 0;                 //End Trade Minute 
extern   string S_52          = "<== TUESDAY ==>";       // TUESDAY 
extern   bool                 TUESDAY_Enabled             = true;              //Trading on Tuesdays
extern   int                  TUESDAY_Start_Trade_Hour   = 22;                //Start Trade Hour
extern   int                  TUESDAY_Start_Trade_Minute = 0;                 //Start Trade Minute
extern   int                  TUESDAY_End_Trade_Hour     = 1;                 //End Trade Hour
extern   int                  TUESDAY_End_Trade_Minute   = 0;                 //End Trade Minute 
extern   string S_53          = "<== WEDNESDAY ==>";     // WEDNESDAY
extern   bool                 WEDNESDAY_Enabled             = true;              //Trading on Wednesdays
extern   int                  WEDNESDAY_Start_Trade_Hour = 22;                //Start Trade Hour
extern   int                  WEDNESDAY_Start_Trade_Minute=0;                //Start Trade Minute
extern   int                  WEDNESDAY_End_Trade_Hour   = 1;                 //End Trade Hour
extern   int                  WEDNESDAY_End_Trade_Minute = 0;                 //End Trade Minute 
extern   string S_54          = "<== THURSDAY ==>";      // THURSDAY 
extern   bool                 THURSDAY_Enabled             = true;              //Trading on Thursdays
extern   int                  THURSDAY_Start_Trade_Hour  = 22;                //Start Trade Hour
extern   int                  THURSDAY_Start_Trade_Minute= 0;                 //Start Trade Minute
extern   int                  THURSDAY_End_Trade_Hour    = 1;                 //End Trade Hour
extern   int                  THURSDAY_End_Trade_Minute  = 0;                 //End Trade Minute 
extern   string S_55          = "<== FRIDAY ==>";        // FRIDAY 
extern   bool                 FRIDAY_Enabled             = true;              //Trading on Fridays
extern   int                  FRIDAY_Start_Trade_Hour    = 22;                //Start Trade Hour
extern   int                  FRIDAY_Start_Trade_Minute  = 0;                 //Start Trade Minute
extern   int                  FRIDAY_End_Trade_Hour      = 1;                 //End Trade Hour
extern   int                  FRIDAY_End_Trade_Minute    = 0;                 //End Trade Minute 

extern   string S_6          = "<==== Roll Over Filter ====>";                // >   >   >   >   >   >    >    >    >    > 
//input    bool                 use_rollover_filter        = 0;                 //No trade in rollover
input    bool                 OpenOrdersInRollover       = true;             //Open deals in Rollover
input    bool                 CloseOrdersInRollover      = true;             //Close deals in Rollover
input    string               rollover_start             = "23:55";           //Start of Rollover
input    string               rollover_end               = "00:35";           //End of Rollover
   
extern   string S_9          = "<==== News Filter ====>";                // >   >   >   >   >   >    >    >    >    > 
extern   bool                UseNewsFilter               = false;             //Use a news filter
extern   int                 TimeBeforeNews              = 60;                //Do not open deal before news, Minutes
extern   int                 CloseTimeBeforeNews         = 30;                //Close all open deals before the news, Minutes (0=Off)
extern   int                 TimeAfterNews               = 120;               //Do not open deal after news, Minutes
//extern   int                 news_offset                 = 3;                 //Server GMT offset
extern   bool                Vhigh                       = true;              //Show High impact news
extern   bool                Vmedium                     = true;              //Show Medium impact news
extern   bool                Vlow                        = true;              //Show Low impact news
extern   string              NewsSymb = "USD,EUR,GBP,CHF,CAD,AUD,NZD,JPY";    //Currencies Filter (Empty Only shows current currencies) 
extern   color               highc                       = clrRed;            //High impact news color
extern   color               mediumc                     = clrLime;           //Medium impact news color
extern   color               lowc                        = clrBlue;           //Low impact news color

extern   string S_7          = "<==== Other Settings ====>";                  // >   >   >   >   >   >    >    >    >    > 
extern   bool                 showinfopanel              = true;              //Show Info Panel
input    color                Col_info                   = C'176,162,168';    //Dashboard Color
input    color                Col_info2                  = clrGray;           //Dashboard Color once EA do not trade
extern   bool                 VisualDebug                = true;
input    color                ChannelEnterColor          = clrYellow;         //Entry Channel Color
input    color                ChannelExitColor           = clrCornflowerBlue; //Exit Channel color
input    color                WarnEnterColor             = clrLightSkyBlue;   //Warning Boundry to enter
input    color                WarnExitColor              = clrPlum;           //Warning Boundry to exit
extern   _EventLogs           LogMode                    = 1;                 //Logging Mode
input    bool                 WriteLogFile               = 0;                 //Write logs to a file

//--------------------------------------
         int                  stoplevel                  = 0;
         double               lots                       = 0;
         int                  PipsDivided                = 1;
         int                  count_sell, count_buy;       
         int                  day_of_trade               = 0;
         string               tp_info                    = "";
         string               be_info                    = "";
         string               f1_info                    = "";
         string               f2_info                    = "";
         string               f3_info                    = "";
         string               f4_info                    = "";
         string               filter_info                = "";
         string               risk_info                  = "";
         string               info_panel                 = "";
         string               maxspread                  = ""; 
         string               TradeHoursFirst            = "";
         string               TradeHoursSecond           = "";
         datetime             _TimeCurrent, _TimeM1, _TimePeriod, iTimeTF, iTimeM1, iTime_MA, iTime_CCI;
         datetime             need_to_verify_channel     = iTime(NULL,TimeFrame,1);
         datetime             ma_period_check,cci_period_check;
         datetime             time_open_buy              = TimeCurrent();
         datetime             time_open_sell             = TimeCurrent();
   
         double               channel_width              = 0;
         double               channel_upper, channel_lower;
         double               ma_shift[5];
         int                  cci_signal_open,cci_signal_close;
         double               stoploss, takeprofit;
         int                  lastticket, lasthistorytotal;  
         datetime             lasttimewrite, lastbarwrite, lastbarwrite1, last_closetime = 0;                                        
//--- ��������� ��� ������ ������ � ����
         string               InpFileName;                                    // ��� �����
         string               fileName                   = "EA_Generic";      // ��� �����
         string               InpDirectoryName           = "Generic LOGS";    // ��� ��������

         bool                 _RealTrade = (!IsTesting() && !IsOptimization() && !IsVisualMode());
         bool                 SetEqSymbol;
         datetime             StartTime, EndTime, PrevDayStartTime, PrevDayEndTime;
         datetime             rtime1, rtime2;
         int                  Start_Trade_Hour[7], End_Trade_Hour[7], Start_Trade_Minute[7], End_Trade_Minute[7];
         bool                 Day_Trade_Enabled[7];
         string               First_StartTimeStr,First_EndTimeStr,Second_StartTimeStr,Second_EndTimeStr;
         string               daystring[7]={"Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"};
         int                  day_of_year_trade=0;
         string               set_name_info = "";
//         bool                 _IsFirstSession;
         color                infopanelcolor,previnfopanelcolor;
         int                  _DayOfWeek;
         double               buy_total_profit,sell_total_profit;
         datetime             first_buy_open_time, first_sell_open_time;
         datetime             lastnewsupdate;
         bool                 _IsNews;
         bool                 skip_tick;
         string               _period;
         bool                 timer_active;
         int                  CloseReason_Multi;
         datetime             DayStartTimeShift;
         int                  TimeShift;
         int                  _DayOfWeekShift;
         double               range;
//---- ���������� ��� ���������� ������������� �������� �������

         #define STR_SHABLON "<!--STR-->"

         #define MAX_CURRENCY 20  // ������������ ���������� ����� (�� ���)

         string Currencies = "AUD, EUR, USD, CHF, JPY, NZD, GBP, CAD, SGD, NOK, SEK, DKK, ZAR, MXN, HKD, HUF, CZK, PLN, RUR, TRY";

         string Shablon = "<!--STR-->, <!--STR-->";  // ������ ��� ��������� ����� �� ������� ������ Currencies

         int AmountCurrency,lotdigit;  // ����� ���������� ����������� �����
         string Currency[MAX_CURRENCY]; // ����������� ������
         double Volumes[MAX_CURRENCY];
         datetime LastUpdateTime;
 //---- ���������� ��� ���������� ������������� �������� �������

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   int height=357;
   //if(!IsTradeAllowed()) { Print("Error: Trade Expert is not Allowed"); return(INIT_FAILED); }
   
   if(IsOptimization()){LogMode=3; showinfopanel=false; VisualDebug=false;}
   
   StringToUpper(SetName);
   SetEqSymbol = StringFind(SetName, _Symbol)>=0;
   if (MaxAmountCurrencyMult>0) height += 55;
   if(showinfopanel) fRectLabelCreate(0,"info_panel",0,0,28,170,height,Col_info);
   
   string s = "";
   switch(Period()){
      case PERIOD_M1:   s = "M1"; break;
      case PERIOD_M5:   s = "M5"; break;
      case PERIOD_M15:  s = "M15"; break;
      case PERIOD_M30:  s = "M30"; break;
      case PERIOD_H1:   s = "H1"; break;
      case PERIOD_H4:   s = "H4"; break;
      case PERIOD_D1:   s = "D1"; break;
      case PERIOD_W1:   s = "W1"; break;
      case PERIOD_MN1:  s = "MN1"; break;
   }
   
   _period=s;
    
   if (LogMode < 3) InpFileName = fileName + "_" + Symbol() + "_" + s + ".txt";
   
   stoplevel = (int)MarketInfo(Symbol(), MODE_STOPLEVEL);
   maxspread = (Max_Spread > 0 ? DoubleToStr(Max_Spread,1) + " pips" : "OFF");
   
   _TimeM1 = StrToTime(TimeToStr(TimeCurrent(),TIME_MINUTES));
   
   day_of_year_trade=0; // Time to ����������������� ��������� ��������� ���������� �� ����������, � ��������� ���� ��������� ��������
   infopanelcolor=0;
   previnfopanelcolor=0;
   lastnewsupdate=0;
   ma_period_check=0;
   cci_period_check=0;
   need_to_verify_channel=0;
   timer_active=false;
   
   if (!TradeAllDaysInTheSameTime) {
      Start_Trade_Hour[1] = MONDAY_Start_Trade_Hour; 
      Start_Trade_Hour[2] = TUESDAY_Start_Trade_Hour; 
      Start_Trade_Hour[3] = WEDNESDAY_Start_Trade_Hour; 
      Start_Trade_Hour[4] = THURSDAY_Start_Trade_Hour; 
      Start_Trade_Hour[5] = FRIDAY_Start_Trade_Hour; 
      End_Trade_Hour[1] = MONDAY_End_Trade_Hour;
      End_Trade_Hour[2] = TUESDAY_End_Trade_Hour;
      End_Trade_Hour[3] = WEDNESDAY_End_Trade_Hour;
      End_Trade_Hour[4] = THURSDAY_End_Trade_Hour;
      End_Trade_Hour[5] = FRIDAY_End_Trade_Hour;
      Start_Trade_Minute[1] = MONDAY_Start_Trade_Minute;
      Start_Trade_Minute[2] = TUESDAY_Start_Trade_Minute;
      Start_Trade_Minute[3] = WEDNESDAY_Start_Trade_Minute;
      Start_Trade_Minute[4] = THURSDAY_Start_Trade_Minute;
      Start_Trade_Minute[5] = FRIDAY_Start_Trade_Minute;
      End_Trade_Minute[1]= MONDAY_End_Trade_Minute;
      End_Trade_Minute[2]= TUESDAY_End_Trade_Minute;
      End_Trade_Minute[3]= WEDNESDAY_End_Trade_Minute;
      End_Trade_Minute[4]= THURSDAY_End_Trade_Minute;
      End_Trade_Minute[5]= FRIDAY_End_Trade_Minute;
      }
   else {
      for(int i=1; i<=5; i++) {
         Start_Trade_Hour[i]=MONDAY_Start_Trade_Hour;
         Start_Trade_Minute[i]=MONDAY_Start_Trade_Minute;
         End_Trade_Hour[i]=MONDAY_End_Trade_Hour;
         End_Trade_Minute[i]=MONDAY_End_Trade_Minute;
         }
      }

   for(int i=1; i<=5; i++) {
      while (Start_Trade_Hour[i]*60+Start_Trade_Minute[i]>End_Trade_Hour[i]*60+End_Trade_Minute[i]) End_Trade_Hour[i] += 24;
      }

   Day_Trade_Enabled[1]=MONDAY_Enabled;   
   Day_Trade_Enabled[2]=TUESDAY_Enabled;   
   Day_Trade_Enabled[3]=WEDNESDAY_Enabled;   
   Day_Trade_Enabled[4]=THURSDAY_Enabled;   
   Day_Trade_Enabled[5]=FRIDAY_Enabled;   
      
   for(int i=0; i<=6; i++) {
      if (!Day_Trade_Enabled[i]) {
         Start_Trade_Hour[i]=0;
         Start_Trade_Minute[i]=0;
         End_Trade_Hour[i]=0;
         End_Trade_Minute[i]=0;
         }
      }
        
   if (OrdersDistance < 0.5) OrdersDistance=0.5;
   if (TotalOrders < 1) TotalOrders=1;
   if (TotalOrders > 10) { Print("���������� ������� ����������� ������ 10. ������������ ���������� �������: 10"); TotalOrders=10; }

   TimeShift = GMT_Offset-2;
   if (!DST) TimeShift -= fGetDSTShift();


   if (Digits == 3 || Digits == 5) { //�������� �� 4�, 5-� �������� ����
      Slippage *= 10; Max_Spread *= 10; Take_Profit *= 10; Stop_Loss *= 10; min_TP *= 10; Entry_Break *= 10; 
      Min_Volatility *= 10; maxcandle *= 10; sizeloss *= 10; Time_Profit_Pips *= 10; Exit_Distance *= 10; 
      Exit_Profit_Pips *= 10; Reverse_Profit *= 10; CCI_Profit_Pips *= 10; Trail_Start *= 10; Trail_Size *= 10; Trail_Step *= 10; X *= 10; Y *= 10;
      PipsDivided = 10; MaxDailyRange *= 10; Max_Spread_On_Close *= 10; OrdersDistance *= 10;
   }   
 
   /*if (sizeloss < 0) {
      Print("������ ����� ��� ����� ����� ��������� ���������� ������, ������ ���� ������ � ������������� ������!");
      Print("������� ��������: ",sizeloss/PipsDivided," Time to������ � ��������: ", MathAbs(sizeloss/PipsDivided));
      sizeloss = MathAbs(sizeloss);
   }  */
      
   //if (!_RealTrade && MarketInfo(NULL,MODE_SPREAD) > Max_Spread && Max_Spread > 0) { Print("Error: Current Spread (" + DoubleToStr(MarketInfo(NULL,MODE_SPREAD)/PipsDivided,1) +  ") > MaxSpread (" + DoubleToStr(Max_Spread/PipsDivided,1) + ")"); return(INIT_FAILED); }
    
   Take_Profit = MathMax(Take_Profit,NormalizeDouble(stoplevel,1));
   Stop_Loss = MathMax(Stop_Loss,NormalizeDouble(stoplevel,1));
   
   /*if (Auto_Risk > 0) */lots = AutoMM_Count(); //������ ��������� ����
   //else lots = Lots;
   
   f1_info = "\n  1. Time Filter: OFF";
   f2_info = "\n  2. Channel Filter: OFF";
   f3_info = "\n  3. MA Filter: OFF";
   f4_info = "\n  4. CCI Filter: OFF";

   if(Exit_Minutes > 0) f1_info = "\n  1. Time Filter: ON";
   if(Exit_Distance/PipsDivided < 100) f2_info = "\n  2. Channel Filter: ON";
   if(MA_period > 0) f3_info = "\n  3. MA Filter: ON";
   if(cci_Period_close > 0) f4_info = "\n  4. CCI Filter: ON";
   filter_info = f1_info + f2_info + f3_info + f4_info;

   if(Auto_Risk > 0.0) {
      if (MM_Depo == 0) {
         risk_info = "\n  AutoRisk = " + DoubleToStr(Auto_Risk, 1) + "%"+(TotalOrders>1?"*"+IntegerToString(TotalOrders)+" = "+DoubleToStr(Auto_Risk*TotalOrders, 1)+"%":""); 
         }
      else
         {
         risk_info = "\n  AutoRisk = " + DoubleToStr(Lots, 2) + " Lot / " + IntegerToString(MM_Depo) + " "+AccountCurrency(); 
         }   
      }
   else
      risk_info = "\n  AutoRisk - Not activated";
   
   set_name_info=(StringLen(SetName)>28 ? StringSubstr(SetName,0,28)+"..." : SetName);
   
   _IsNews=false;
   if (_RealTrade) {
      if (UseNewsFilter && !IsDllsAllowed()) {
   		string q = "��� ������ ���������� ������� ���������� � ���������� ��������� ������������� DLL. ��������� ������ Off����.";
		   Print(q);
   		fWriteDataToFile(q);
         Alert(q);
         UseNewsFilter=false;
         }
      /*if (!UseNewsFilter) */ObjectsDeleteAll(0,"urdala_",0,OBJ_VLINE);
      if (UseNewsFilter) _IsNews=IsNews(TimeBeforeNews,TimeAfterNews); 
      }
   
   if(MarketInfo(Symbol(),MODE_LOTSTEP)==1) lotdigit=0;
   if(MarketInfo(Symbol(),MODE_LOTSTEP)==0.1) lotdigit=1;   
   if(MarketInfo(Symbol(),MODE_LOTSTEP)==0.01) lotdigit=2;  
   AmountCurrency = StrToStringS(Currencies, ",", Currency);
   
   Comment("");
   
   OnTick();   

   UpdateInfoPanel();
   
   return(INIT_SUCCEEDED);
}
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(_RealTrade){
      Comment("");
      if (showinfopanel) fRectLabelDelete(0,"info_panel");
   }
}
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{

   skip_tick = false;
   _TimeCurrent = TimeCurrent();
   _DayOfWeek = DayOfWeek();   

   iTimeTF = iTime(NULL,TimeFrame,0); 
   if (_RealTrade) if (IsError("iTime",iTimeTF)) skip_tick=true;
   iTimeM1 = iTime(NULL,PERIOD_M1,0); 
   if (_RealTrade) if (IsError("iTimeM1",iTimeM1)) skip_tick=true;
   iTime_MA = iTime(NULL,TimeFrameMA,0); 
   if (_RealTrade) if (IsError("iTime MA",iTime_MA)) skip_tick=true;
   iTime_CCI = iTime(NULL,TimeFrame_CCI,0); 
   if (_RealTrade) if (IsError("iTime CCI",iTime_CCI)) skip_tick=true;
   
   if ((_DayOfWeek>=0) && (_DayOfWeek<=6)) {
      if (day_of_year_trade != DayOfYear()) {
      
         TimeShift = GMT_Offset-2;
         if (!DST) TimeShift -= fGetDSTShift();
         
         int weekends_shift=0;
         if (TimeDayOfWeek(TimeCurrent()) == 0 && TimeShift < 0) { weekends_shift = -24*60*60; }
         if (TimeDayOfWeek(TimeCurrent()) == 6 && TimeShift > 0) { weekends_shift = +24*60*60; }
         _DayOfWeekShift = TimeDayOfWeek(TimeCurrent()+weekends_shift-TimeShift*60*60); 
         if (_DayOfWeekShift > 5) _DayOfWeekShift=1;
         if (_DayOfWeekShift < 1) _DayOfWeekShift=5;
         int _PrevDayOfWeekShift = _DayOfWeekShift-1;   
         if (_PrevDayOfWeekShift < 1) _PrevDayOfWeekShift=5;
         if (_PrevDayOfWeekShift > 5) _PrevDayOfWeekShift=1;

         datetime _CurrentDate = StrToTime(TimeToStr(_TimeCurrent,TIME_DATE));
         DayStartTimeShift = _CurrentDate+TimeShift*60*60;

         PrevDayStartTime=_CurrentDate-24*60*60+(Start_Trade_Hour[_PrevDayOfWeekShift])*60*60+Start_Trade_Minute[_PrevDayOfWeekShift]*60+TimeShift*60*60;
         PrevDayEndTime = _CurrentDate-24*60*60+(End_Trade_Hour[_PrevDayOfWeekShift])*60*60+End_Trade_Minute[_PrevDayOfWeekShift]*60+TimeShift*60*60;
         StartTime=_CurrentDate+(Start_Trade_Hour[_DayOfWeekShift])*60*60+Start_Trade_Minute[_DayOfWeekShift]*60+TimeShift*60*60;
         EndTime = _CurrentDate+(End_Trade_Hour[_DayOfWeekShift])*60*60+End_Trade_Minute[_DayOfWeekShift]*60+TimeShift*60*60;
         
         if (_DayOfWeek == 0 && StartTime < DayStartTimeShift+24*60*60) { StartTime=DayStartTimeShift+24*60*60; }
         if (_DayOfWeek == 5 && EndTime > DayStartTimeShift+24*60*60) { EndTime=DayStartTimeShift+24*60*60; }

         if (showinfopanel) {
            First_StartTimeStr=TimeToStr(PrevDayStartTime,TIME_MINUTES);
            if (TimeDayOfWeek(PrevDayStartTime) != _DayOfWeek) First_StartTimeStr = "00:00";
            First_EndTimeStr=TimeToStr(PrevDayEndTime,TIME_MINUTES);
            if (TimeDayOfWeek(PrevDayEndTime) != _DayOfWeek) First_EndTimeStr = "00:00";
            Second_StartTimeStr=TimeToStr(StartTime,TIME_MINUTES);
            if (TimeDayOfWeek(StartTime) != _DayOfWeek) Second_StartTimeStr = "00:00";
            Second_EndTimeStr=TimeToStr(EndTime,TIME_MINUTES);
            if (TimeDayOfWeek(EndTime) != _DayOfWeek) Second_EndTimeStr = "00:00";
            }
         
         rtime1 = StrToTime(StringConcatenate(TimeToStr(_TimeCurrent,TIME_DATE)," ",rollover_start))+TimeShift*60*60;
         rtime2 = StrToTime(StringConcatenate(TimeToStr(_TimeCurrent,TIME_DATE)," ",rollover_end))+TimeShift*60*60;

         day_of_year_trade = DayOfYear();
         }
      }


   if (_RealTrade) {
      if ((UseNewsFilter) && (lastnewsupdate != iTimeTF)) { // ��� ���������� � ����������
         _IsNews=IsNews(TimeBeforeNews,TimeAfterNews); 
         lastnewsupdate = iTimeTF;
         }
      }

   UpdateInfoPanel();
   
   if (_TimeM1 == iTimeM1 && !every_tick ) return ;
   _TimeM1 =  iTimeM1;

   if(!IsTime() && CountOrder(-1)<1) return;

   if(need_to_verify_channel != iTimeTF){ //��������� ������ ���������� ��� � ������
      channel_upper = iBands(NULL,TimeFrame,BB_Period,BB_Deviation,0,PRICE_CLOSE,MODE_UPPER,1);
      if (_RealTrade) if (IsError("iBands",channel_upper)) skip_tick=true;
      channel_lower = iBands(NULL,TimeFrame,BB_Period,BB_Deviation,0,PRICE_CLOSE,MODE_LOWER,1);
      if (_RealTrade) if (IsError("iBands 2",channel_upper)) skip_tick=true;

      if(!skip_tick && VisualDebug){
         if(IsTime()) {
            DrawChannel("up", channel_upper + Entry_Break*_Point, ChannelEnterColor);
            DrawChannel("down", channel_lower - Entry_Break*_Point, ChannelEnterColor);
         }
         //if(CountOrder(OP_BUY) > 0) {
            DrawChannel("up_exit", channel_upper + Exit_Distance*Point, ChannelExitColor);

         //}
         //if(CountOrder(OP_SELL) > 0) {
            DrawChannel("down_exit", channel_lower - Exit_Distance*Point, ChannelExitColor);
         //}
      }
      
      channel_width = channel_upper - channel_lower;
      need_to_verify_channel = iTimeTF;
      
      if(TP_perc > 0) Take_Profit = MathMax(NormalizeDouble(channel_width/_Point /100 * TP_perc,1), min_TP); //�������� �� ������������ ��
   }
   
   if(ma_period_check != iTime_MA){
      for (int i=1;i<=4;i++) {
         ma_shift[i] = NormalizeDouble(iMA(NULL, TimeFrameMA, MA_period, 0, MA_type, MA_price, i),Digits);
         if (_RealTrade) if (IsError("iMA "+IntegerToString(i),ma_shift[i])) skip_tick=true;
         }
      ma_period_check = iTime_MA;
      }   
      
   if(cci_period_check != iTime_CCI){
      if (cci_Period_open>0) {
         cci_signal_open=fGetCCISignal(cci_Period_open,PRICE_CLOSE,cci_level_open,1);
         if (_RealTrade) if (IsError("iCCI open",cci_signal_open,false)) skip_tick=true; // iCCI ����� ������� 0 � ��� ������
         } else cci_signal_open=-1;
      if (cci_Period_close>0) {
         cci_signal_close=fGetCCISignal(cci_Period_close,PRICE_CLOSE,cci_level_close,1);
         if (_RealTrade) if (IsError("iCCI close",cci_signal_close,false)) skip_tick=true;
         } else cci_signal_close=-1;
      cci_period_check = iTime_CCI;
      }   

   if (skip_tick) { 
      //Print("skipping tick");
      if (_RealTrade) EventSetTimer(1);
      return; 
      }
      
   if (timer_active) { EventKillTimer();timer_active=false; }
 

   range = 0;
   count_buy = 0;
   count_sell = 0;
   int _OrdersTotal = OrdersTotal();
   CloseReason_Multi=0;
   
   if (TotalOrders>1 && CloseSimultaneously) {
      int cnt_buy=CountOrder(OP_BUY);
      int cnt_sell=CountOrder(OP_SELL);
      if (cnt_buy > 0) buy_total_profit=GetOrdersTotalProfit(OP_BUY)/cnt_buy;
      if (cnt_sell > 0) sell_total_profit=GetOrdersTotalProfit(OP_SELL)/cnt_sell;
      first_buy_open_time=GetFirstOpenTime(OP_BUY);
      first_sell_open_time=GetFirstOpenTime(OP_SELL);
      }

   for(int pos = _OrdersTotal - 1; pos >= 0; pos--){
      if(!OrderSelect(pos, SELECT_BY_POS, MODE_TRADES)) {
			if (LogMode < 3) {
				string q = __FUNCTION__ + ": �� ������� �������� �����! " + fMyErDesc();
				Print(q);
				fWriteDataToFile(q);
			}   
      }
      else if(OrderType() <= OP_SELL && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber) {

         if(OrderType() == OP_BUY) {
            count_buy++;
            if(OrderStopLoss() == 0.0){ //��������������� ������
               stoploss = NormalizeDouble(OrderOpenPrice() - Stop_Loss*_Point, Digits);
               takeprofit = NormalizeDouble(OrderOpenPrice() + Take_Profit*_Point, Digits);
               
               fModifyPosition(OrderTicket(), OrderOpenPrice(), stoploss, takeprofit, 0, clrGreen);
               continue;
            }
            
            if(X > 0 && Y > 0 && X > Y && Bid - OrderOpenPrice() >= X*_Point && OrderStopLoss() < OrderOpenPrice()){ //���������
               stoploss = NormalizeDouble(OrderOpenPrice() + Y*_Point, Digits);
               fModifyPosition(OrderTicket(), OrderOpenPrice(), stoploss, OrderTakeProfit(), 0, clrGreen);
            }
            
            Modify_and_exit_condition(OrderType(),OrderLots(),OrderOpenPrice(),OrderOpenTime(),OrderTicket(),OrderStopLoss()); //����������� ������� � �������� ������� �� �����
         }
         
         else if(OrderType() == OP_SELL){
            count_sell++;
            if (OrderStopLoss() == 0.0) { //����������� ������
               stoploss = NormalizeDouble(OrderOpenPrice() + Stop_Loss*_Point, Digits);
               takeprofit = NormalizeDouble(OrderOpenPrice() - Take_Profit*_Point, Digits);
               
               fModifyPosition(OrderTicket(), OrderOpenPrice(), stoploss, takeprofit, 0, clrGreen);
               continue;
            }
            
            if(X > 0 && Y > 0 && X > Y && OrderOpenPrice() - Ask >= X*_Point && OrderStopLoss() > OrderOpenPrice()){ //��
               stoploss = NormalizeDouble(OrderOpenPrice() - Y*_Point, Digits);
               fModifyPosition(OrderTicket(), OrderOpenPrice(), stoploss, OrderTakeProfit(), 0, clrGreen);
            }
            
            Modify_and_exit_condition(OrderType(),OrderLots(),OrderOpenPrice(),OrderOpenTime(),OrderTicket(),OrderStopLoss()); //����������� ������� � �������� ������� �� �����
         }
      }     
   }       
   
   if(!IsTime() || (!OpenOrdersInRollover && fGetRollOver())) return;

   double h_price;
   bool is_order_dist;
   bool openresult;
   
   h_price=GetMaxMinOpenPrice(OP_BUY,MIN); // ����������� ���� �������� �������
   if (h_price != 0) is_order_dist = Ask < h_price-OrdersDistance*_Point; else is_order_dist = true;  // ���������� �� �������� ������� �� ������ ���������

   count_buy=CountOrder(OP_BUY);
   count_sell=CountOrder(OP_SELL);
   
  // �������� ������, ���� ��������� ��� �������
   if (count_buy < 1 &&  //���� ��� �������� �������
      (OnlyBid ? Bid : Ask) < channel_lower - Entry_Break*_Point) {  //���� ��������� ������� ������ BB
        openresult=OpenTradeConditions("BUY",OP_BUY,time_open_buy,count_buy,count_sell);
   }
   else if(count_buy >= 1 && count_buy < TotalOrders && is_order_dist && _TimeCurrent-GetLastOpenTime(OP_BUY)>MinPause) { 
        openresult=OpenTradeConditions("BUY",OP_BUY,time_open_buy,count_buy,count_sell);
   } 

   h_price=GetMaxMinOpenPrice(OP_SELL,MAX); // ������������ ���� �������� �������
   if (h_price != 0) is_order_dist = Bid > h_price+OrdersDistance*_Point; else is_order_dist = true; // ���������� �� �������� ������� �� ������ ���������
   
   count_buy=CountOrder(OP_BUY);
   count_sell=CountOrder(OP_SELL);

   if (count_sell < 1 && //���� ��� �������� ������
      Bid > channel_upper + Entry_Break*_Point) {  //���� ��������� ������� ������ BB
        openresult=OpenTradeConditions("SELL",OP_SELL,time_open_sell,count_sell,count_buy);
   }         
   else if(count_sell >= 1 && count_sell < TotalOrders && is_order_dist && _TimeCurrent-GetLastOpenTime(OP_SELL)>MinPause) {  
        openresult=OpenTradeConditions("SELL",OP_SELL,time_open_sell,count_sell,count_buy);
   }   
   
   int Error = GetLastError(); //����� ������ � ����������
   if(Error != 0) Print("OnTick() Error ",Error,": ",ErrorDescription(Error));
}

void OnTimer()
{
timer_active=true;
OnTick();
}

//+------------------------------------------------------------------+
bool Modify_and_exit_condition(int _OrderType, double _OrderLots, double _OrderOpenPrice, datetime _OrderOpenTime,int _OrderTicket, double _OrderStopLoss) {
               
   if(Trail_Start > 0 && Trail_Size > 0) { //����� �������� �������
      if (!rollover_trall_end || (rollover_trall_end && !fGetRollOver())) fTrailingStopFunc();
   }

   string logstr="";
   string warnstr="";
   string filterstr="";
   int filterno=0;

   bool condition_for_the_exit = false;
   bool channel_condition = false;
   
   double PriceType = Ask; //���� ����� ����.
   double OrderDistance = _OrderOpenPrice - PriceType;
   double orderdist;
   string orderstr="";
   string ordertotalstr="";
   
   if (_OrderType == OP_BUY) { //���� ����� ���
      PriceType = Bid;
      OrderDistance = PriceType - _OrderOpenPrice;
   }
  
   if (TotalOrders>1 && CloseSimultaneously) {
      orderdist=OrderDistance;
      orderstr="; Time to���� ������ - "+DoubleToStr(orderdist/_Point/PipsDivided,1)+" �������";
      ordertotalstr=" �������";
      if (_OrderType == OP_BUY) {
         OrderDistance=buy_total_profit;
         _OrderOpenTime=first_buy_open_time;
         }
      if (_OrderType == OP_SELL) {
         OrderDistance=sell_total_profit;
         _OrderOpenTime=first_sell_open_time;
         }
      }
  
   if (_OrderType == OP_SELL && (OnlyBid ? Bid : Ask) <= channel_lower - Exit_Distance*_Point) channel_condition = true;
   else if (_OrderType == OP_BUY && Bid >= channel_upper + Exit_Distance*_Point) channel_condition = true;
      
      
   if((Exit_Minutes > 0 && _TimeCurrent - _OrderOpenTime > 60 * Exit_Minutes && // ����� ������ ����� Exit_Minutes �
      OrderDistance > Time_Profit_Pips*_Point) || (CloseReason_Multi == 1)) {                                 // ��������� Time to���� ����� Time_Profit_Pips (0)
      condition_for_the_exit = true;
      filterno=1;
	   if (LogMode < 2) {
		   logstr = "�������� ������ #" + IntegerToString(_OrderTicket) + " �� ���� "+ DoubleToStr(PriceType,Digits) + ". ����� ������������� ����� " + IntegerToString(Exit_Minutes) + 
		         " ����� �"+ordertotalstr+" ��������� Time to���� ���������� ����� " + DoubleToStr(Time_Profit_Pips/PipsDivided,1) + " �������: " + DoubleToStr(OrderDistance/_Point/PipsDivided,1)+" �������" + orderstr + "; ����� = " + DoubleToStr((Ask - Bid) / _Point/PipsDivided, 1)+ " �������.";
		   warnstr="1. Time";
		   filterstr="1. Time";
	   }
   }
   if((Exit_Distance/PipsDivided < 100 && channel_condition &&    // ���� ����� �� ������� ������ �
      OrderDistance > Exit_Profit_Pips*_Point) || (CloseReason_Multi == 2)) {           // ��������� Time to���� ����� Exit_Profit_Pips (-12)
      condition_for_the_exit = true;
      filterno=2;
	   if (LogMode < 2) {
		   logstr = "�������� ������ #" + IntegerToString(_OrderTicket) + " �� ���� "+ DoubleToStr(PriceType,Digits) + ". ���� ��������� ������� ������ " + DoubleToStr(channel_lower,Digits) + " �� " + 
			      DoubleToStr(Exit_Distance/PipsDivided,1) + " ������� �"+ordertotalstr+" ��������� Time to���� ��������� ����� " + DoubleToStr(Exit_Profit_Pips/PipsDivided,1) + " �������: " + DoubleToStr(OrderDistance/_Point/PipsDivided,1)+" �������" + orderstr + "; ����� = " + DoubleToStr((Ask - Bid) / _Point/PipsDivided, 1)+ " �������.";
		   warnstr="2. BB Channel";
		   filterstr="2. BB Channel";
      }
   }

   if(MA_period > 0 && _OrderOpenTime < iTime(NULL,TimeFrameMA,0) ) { //������� �� ��
   
      int MA_Type_Exit = -1; //���������
      if (ma_shift[1] > ma_shift[2] && ma_shift[2] <= ma_shift[3] && ma_shift[3] <= ma_shift[4]) MA_Type_Exit = OP_SELL;
      else if (ma_shift[1] < ma_shift[2] && ma_shift[2] >= ma_shift[3] && ma_shift[3] >= ma_shift[4]) MA_Type_Exit = OP_BUY;
      
      if((((_OrderType == OP_SELL && MA_Type_Exit == OP_SELL) || (_OrderType == OP_BUY && MA_Type_Exit == OP_BUY)) && //���������� ������� ���������� ��� ����������
         OrderDistance > Reverse_Profit*_Point) || (CloseReason_Multi == 3)) { // � ��������� Time to���� ����� Reverse_Profit (20)
         condition_for_the_exit = true;
         filterno=3;
		   if (LogMode < 2) {
			   logstr = "�������� ������ #" + IntegerToString(OrderTicket()) + " �� ���� "+ DoubleToStr(PriceType,Digits) + ". ���������� ������� ����������" +
				      " �"+ordertotalstr+" ��������� Time to���� ����� " + DoubleToStr(Reverse_Profit/PipsDivided,1) + " �������: " + DoubleToStr(OrderDistance/_Point/PipsDivided,1)+" �������" + orderstr + "; ����� = " + DoubleToStr((Ask - Bid) / _Point/PipsDivided, 1)+ " �������.";
			   warnstr="3. MA";
			   filterstr="3. MA";
		   }
      }
   }

   if((cci_Period_close > 0 && OrderDistance > CCI_Profit_Pips*_Point && cci_signal_close != _OrderType &&
      cci_signal_close != -1) || (CloseReason_Multi == 4)) { //������ �� CCI
      condition_for_the_exit = true;
      filterno=4;
	   if (LogMode < 2) {
		   logstr = "�������� ������ #" + IntegerToString(_OrderTicket) + " �� ���� "+ DoubleToStr(PriceType,Digits) + ". ����� �� ���������� CCI" + "; ����� = " + DoubleToStr((Ask - Bid) / _Point/PipsDivided, 1)+ " �������.";
		   warnstr="4. CCI";
		   filterstr="4. CCI";
	   }
   }
   
   if (UseNewsFilter && CloseTimeBeforeNews != 0 && IsNews(CloseTimeBeforeNews,TimeAfterNews)) { //������ ��������
      condition_for_the_exit = true;
      if (LogMode < 2) {
		   logstr = "�������� ������ #" + IntegerToString(_OrderTicket) + " �� ���� "+ DoubleToStr(PriceType,Digits) + ". ������ ��������" + "; ����� = " + DoubleToStr((Ask - Bid) / _Point/PipsDivided, 1)+ " �������.";
		   warnstr="5. News";
		   filterstr="5. News";
      }   
   }   

   if(CloseReason_Multi == 0 && condition_for_the_exit) {
      if ((!CloseOrdersInRollover) && fGetRollOver()) {
         if (LogMode < 2 && _TimeCurrent-lasttimewrite > 60) {
            logstr = "������� �������� ������ #" + IntegerToString(OrderTicket()) + " (������ "+filterstr+"): ������ ���������. ����� �� ��� ������.";
            warnstr += " + rollover";
            DrawWarn(warnstr, WarnExitColor);
            Print(logstr);
            fWriteDataToFile(logstr);
            lasttimewrite = _TimeCurrent;
            }   
         return(false);
      }
      if(Max_Spread_On_Close>0){
         if ((Ask - Bid) > Max_Spread_On_Close*_Point && OrderType()==OP_SELL) { //������ �� ������������� ������
            if (LogMode < 2 && _TimeCurrent-lasttimewrite > 60) {
               logstr = "������� �������� ������ #" + IntegerToString(OrderTicket()) + " (������ "+filterstr+"): ����� (" + DoubleToStr((Ask - Bid) / _Point/PipsDivided, 1) + " �������) ������ ������������� (" + DoubleToStr(Max_Spread_On_Close/PipsDivided, 1) + ")! ����� �� ��� ������.";
               warnstr += " + spread";
               DrawWarn(warnstr, WarnExitColor);
               Print(logstr);
               fWriteDataToFile(logstr);               
               lasttimewrite = _TimeCurrent;
            }   
             return(false);
         }
      }
   }
      
   if(condition_for_the_exit) { //���� �������� ���� ���� ������:
   
      if (TotalOrders>1 && CloseSimultaneously) CloseReason_Multi=filterno;

      if (warnstr != "") DrawWarn(warnstr, WarnExitColor);
      if (logstr != "") {
         Print(logstr);
         fWriteDataToFile(logstr);
         }
 
      RefreshRates();
      if(!OrderClose(_OrderTicket, _OrderLots, NormalizeDouble(PriceType, Digits), Slippage, clrViolet)){ //�������� ������
			if (LogMode < 3) {
			   string q = __FUNCTION__ + ": ����� " + Symbol() + " #" + IntegerToString(OrderTicket()) + " �� ��� ������! " + fMyErDesc();
			   Print(q);
			   fWriteDataToFile(q);
			}
			Sleep(3000);
			return(false);
      }
      else{
         Sleep(1000);
         return(true);
      }
      //Sleep(3000);
      return(true);
   }

   return(true);
}

//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
bool OpenTradeConditions(string _OrderType, int OP_TYPE, datetime &time_open, int order_cnt, int opposite_order) {     

   if(/*order_cnt == 0 && */cci_Period_open > 0 && cci_signal_open != OP_TYPE) { //������ �� CCI // ��������� ������ Time to �������� ������� ������
      if (LogMode < 2 && lastbarwrite != iTimeTF) {
         string q = Symbol() + "| CCI �� ����� �� �������: �" + IntegerToString(cci_level_open) + " ����� " + _OrderType + " �� ��� ������.";
         Print(q); fWriteDataToFile(q);
         DrawWarn("filter CCI",WarnEnterColor);
	      lastbarwrite = iTimeTF;
      }   
      return(false);
   }
   if (channel_width != 0 && channel_width < Min_Volatility*_Point) { //������ �� ������ ������
      if (LogMode < 2 && lastbarwrite != iTimeTF) {
         string q = Symbol() + "| ������� ������ ������: (" + DoubleToStr(channel_width/_Point/PipsDivided, 1) + " �������) ������, ��� ����������� ������ ������: " + DoubleToStr(Min_Volatility/PipsDivided, 1) + " ����� " + _OrderType + " �� ��� ������.";
         Print(q); fWriteDataToFile(q);
         DrawWarn("filter channel width",WarnEnterColor);
	      lastbarwrite = iTimeTF;
      }   
      return(false);
   }
   if (maxcandle > 0) { 
      double candlehl=0;
      double candlerange=0;
      if (OP_TYPE==OP_BUY) {
         candlehl=iHigh(NULL,0,iHighest(NULL,0,MODE_HIGH,barcount,0));
         if (_RealTrade) if (IsError("iHigh",candlehl)) { EventSetTimer(1);return(false); }
         candlerange=candlehl-Ask;
         }
      if (OP_TYPE==OP_SELL) {
         candlehl=iLow(NULL,0,iLowest(NULL,0,MODE_LOW,barcount,0));
         if (_RealTrade) if (IsError("iLow",candlehl)) { EventSetTimer(1);return(false); }
         candlerange=Bid-candlehl;
         }
      if (candlerange>maxcandle*_Point) {   
//      if (((OP_TYPE==OP_BUY) && (candlehl-Ask>maxcandle*_Point)) ||
//         ((OP_TYPE==OP_SELL) && (Bid-candlehl>maxcandle*_Point))) {
      //if ((candle() > maxcandle*_Point)){ // ������ �� ������� ���������� ������
         if (LogMode < 2 && lastbarwrite != iTimeTF) {
            //string q = Symbol() + "| ������� ����� �� ���������� " + IntegerToString(barcount) + " �����, ������ " + DoubleToStr(maxcandle/PipsDivided, 1) + " �������. ����� " + _OrderType + " �� ��� ������.";
            string q = Symbol() + "| �������� �������� ���� �� ���������� " + IntegerToString(barcount) + " ����� "+_period+" (" + DoubleToStr(candlerange/_Point/PipsDivided, 1) + " �������) ������ " + DoubleToStr(maxcandle/PipsDivided, 1) + " �������. ����� " + _OrderType + " �� ��� ������.";
            Print(q); fWriteDataToFile(q);
            DrawWarn("filter maxcandle",WarnEnterColor);
	         lastbarwrite = iTimeTF;
         }   
         return(false);
      }   
   }
   if (!Hedging && opposite_order >= 1) { //������ ������� �� ������������
      if (LogMode < 2 && _TimeCurrent-lasttimewrite > 120) {
         string q = Symbol() + "| ������ ����������� ������� ������� � ������� ��������������� ������. ����� " + _OrderType + " �� ��� ������.";
         Print(q); fWriteDataToFile(q);
         DrawWarn("filter hedge",WarnEnterColor);
	      lasttimewrite = _TimeCurrent;
      }   
      return(false);
   }   
   if(MaxDailyRange > 0) { //������ MaxDailyRange
      int mdr_shift=0;
      double OpenD1;
      datetime timeopend1=DayStartTimeShift;
      if (TimeShift < 0 && _TimeCurrent > DayStartTimeShift+24*60*60 && (!CheckMDRAfter0 || (CheckMDRAfter0 && _DayOfWeekShift == 1))) {
         timeopend1 = DayStartTimeShift+24*60*60;
         }
      if (/*TimeShift == 0 && */CheckMDRAfter0 && _TimeCurrent < StartTime && _DayOfWeekShift != 1) {
         timeopend1 = DayStartTimeShift-24*60*60;
         }
      //if (CheckMDRAfter0 && _IsFirstSession) mdr_shift=1;
      //iOpenD1=iOpen(NULL,PERIOD_D1,mdr_shift);
      int barshift;
      //int a=0;
      //do {
         barshift=iBarShift(NULL, _Period, timeopend1); 
         //if (barshift==-1) timeopend1 -= 24*60*60; // ���� �������� �� ��������
         //if (barshift==-1) Print("BarShift = -1, timeopend1 = ",TimeToStr(timeopend1));
         //a++;
         //} while (barshift == -1 && a < 10);
      //if (a==10) { Print("BarShift Error"); return(false); }
      if (MDRFromHiLo) {
         if (OP_TYPE==OP_BUY) OpenD1=iHigh(NULL,0,iHighest(NULL,0,MODE_HIGH,barshift,0));
                         else OpenD1=iLow(NULL,0,iLowest(NULL,0,MODE_LOW,barshift,0));
         }  else OpenD1 = iOpen(NULL, _Period, barshift);
      if (_RealTrade) if (IsError("iOpen mdr",OpenD1)) { EventSetTimer(1); return(false); }
      double mdr = OP_TYPE==OP_BUY ? OpenD1 - Bid : Bid - OpenD1;
      range=mdr;
      //if (mdr == 0) { Print("mdr = 0"); return(false); }
      if(mdr > MaxDailyRange*_Point){
         if (LogMode < 2 && lastbarwrite != iTimeTF) {
            string q = Symbol() + "| ������� �������� �������� ����: (" + DoubleToStr(mdr/_Point/PipsDivided, 1) + " �������) ,������ MaxDailyRange: " + DoubleToString(MaxDailyRange/PipsDivided,1) + " ����� " + _OrderType + " �� ��� ������.";
            Print(q);
            DrawWarn("filter MDR",WarnEnterColor);
	         lastbarwrite = iTimeTF;
         }   
         return(false);
      }
   }
   if (UseNewsFilter && IsNews(TimeBeforeNews,TimeAfterNews)) { //������ ��������
      if (LogMode < 2 && lastbarwrite != iTimeTF) {
         string q = Symbol() + "| ������ ��������. ����� " + _OrderType + " �� ��� ������.";
         Print(q); fWriteDataToFile(q);
         DrawWarn("filter news",WarnEnterColor);
   	   lastbarwrite = iTimeTF;
      }   
      return(false);
   }

   if (_RealTrade && MaxAmountCurrencyMult>0) { //������ �� ����������� ������������ ������
      string Str,Str1,Str2;
      double MaxLot;
      MaxLot=MaxAmountCurrencyMult*lots;
      CheckArbitrage();
      Str1 = StringSubstr(Symbol(), 0, 3);
      Str2 = StringSubstr(Symbol(), 3, 3);
      bool allow_by_max_amount=true;
      if (OP_TYPE==OP_BUY) {
         if (Volumes[CurrencyPos(Str1)] >= MaxLot) { Str=Str1; allow_by_max_amount=false; }
         if (Volumes[CurrencyPos(Str2)] <= -MaxLot) { Str=Str2; allow_by_max_amount=false; }
         }
      if (OP_TYPE==OP_SELL) {
         if (Volumes[CurrencyPos(Str1)] <= -MaxLot) { Str=Str1; allow_by_max_amount=false; }
         if (Volumes[CurrencyPos(Str2)] >= MaxLot) { Str=Str2; allow_by_max_amount=false; }
         }
      if (!allow_by_max_amount) {
         if (LogMode < 2 && lastbarwrite != Time[0]) {
            string q = Symbol() + "| ����������� ����������� ����� " + Str + DoubleToString(MaxLot,2) + " ����� ��������. " + " ����� " + _OrderType + " �� ��� ������.";
            Print(q);
            DrawWarn("filter MaxAmount",WarnEnterColor);
   	      lastbarwrite = Time[0];
         }   
         return(false);      
      }
   }
   if (Max_Spread > 0 && (Ask - Bid) > Max_Spread*_Point && (CheckSpreadOnSellOpen || OP_TYPE==OP_BUY)) { //������ �� ������������� ������
      if (LogMode < 2 && _TimeCurrent-lasttimewrite > 60) {
         string q = Symbol() + "| ����� (" + DoubleToStr((Ask - Bid) / _Point/PipsDivided, 1) + " �������) ������ ������������� (" + DoubleToStr(Max_Spread/PipsDivided, 1) + ")! ����� " + _OrderType + " �� ��� ������.";
         Print(q); fWriteDataToFile(q);
         DrawWarn("filter spread",WarnEnterColor);
	      lasttimewrite = _TimeCurrent;
      }   
      return(false);
   }   
  
  
  
   if (pause > 0 && lastloss() != 0) { //���� ��������� ����� �� ������ � ������ (��� ������ ������ ���������)
      return(false);
      }
         
   if (!IsTradeAllowed()) {
      if (LogMode < 3 && lastbarwrite1 != iTimeTF) {
         string q = Symbol() + "| �������� �� ���������. ���������� �������� ����� <��������� " +
                              "��������� ���������> � ��������� ��������." + " ����� " + _OrderType + " �� ��� ������.";
         Print(q); fWriteDataToFile(q);
         DrawWarn("trade is not allowed",WarnEnterColor);
	      lastbarwrite1 = iTimeTF;
      }   
      return(false);
      }

   bool res=false;
   
      res=OpenTrade(_OrderType);  //��������� �������� �����
      if (res) time_open = _TimeCurrent; 
      
   return(res); 
}
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+      
bool OpenTrade(string type){ //�������� �������
   double price = 0;
   int cmd = -1;
   int err = 0;
   color col_type = clrNONE;
   
   /*if (Auto_Risk > 0) */lots = AutoMM_Count(); //������ ��������� ����
   //else lots = Lots;
         
   if(type == "BUY"){
      cmd = OP_BUY;
      col_type = clrAqua;
   }
   if(type == "SELL"){
      cmd = OP_SELL;
      col_type = clrRed;
   }
   int order_count=CountOrder(cmd);
   for(int count = 0; count < 5; count++){
      RefreshRates();
      int ticket=0;
      if (CountOrder(cmd)<=order_count) { 
         if(type == "BUY") price = Ask;
         if(type == "SELL") price = Bid;
         ticket = OrderSend(Symbol(), cmd, lots, price, Slippage, 0, 0, "Generic A-TLP|"+IntegerToString(MagicNumber), MagicNumber, 0, col_type);   
         Sleep(3000);
         }
      else { // ���� �� ���������� ������� ���� ���������� ������, �� ����� ��� ������
         ticket=GetLastOrderTicket(cmd); // ���������� ����� ���������� ��������� ������
         }   
      if(ticket > 0){
         lastticket = ticket;
		   if (LogMode < 3) {
			   if(!OrderSelect(ticket, SELECT_BY_TICKET, MODE_TRADES)) {
   				string q = __FUNCTION__ + ": �� ������� �������� ����� " +
   						 IntegerToString(ticket) + "! " + fMyErDesc();
   				Print(q);
   				fWriteDataToFile(q);
   				//break;
		      }
         }
		   if (LogMode < 2) {
				string q = __FUNCTION__ + ": ����� " + type + " ������ �� ���� " + DoubleToStr(OrderOpenPrice(),Digits) + "; ������ ������  = " + DoubleToStr(channel_width/_Point/PipsDivided,1) + "; DailyRange  = " + DoubleToStr(range/_Point/PipsDivided,1) + "; ����� = " + DoubleToStr((Ask - Bid) / _Point/PipsDivided, 1)+ " �������.";
				Print(q);
				fWriteDataToFile(q);
				//break;
		   }
         return(true);
      } 
      else {
         lastticket = 0;
         err=GetLastError();
         if(err != ERR_INVALID_PRICE && err != ERR_PRICE_CHANGED && err != ERR_REQUOTE && err != ERR_OFF_QUOTES && err != ERR_TRADE_CONTEXT_BUSY) break;
		   if (LogMode < 3) {
   		   string q = __FUNCTION__ + ": ������ �������� ������ " + type + ": " + fMyErDesc(err);
   		   Print(q);
   		   fWriteDataToFile(q);
		   }
         Sleep(3000);
      }
   }
   if (LogMode < 3) {
	   string q = __FUNCTION__ + ": ����� " + type + " �� ��� ������!: " + fMyErDesc(err);
	   Print(q);
	   fWriteDataToFile(q);
      }
   Sleep(3000);
   return(false);
}

//+------------------------------------------------------------------+          
//+------------------------------------------------------------------+
double AutoMM_Count() { //������ ���� Time to �������� �������� ����� � ����������.
   double lot=Lots;int loss = TotalLoss();
   
   if (Auto_Risk > 0.0) {
      if (MM_Depo == 0) {
         double TickValue = (MarketInfo(Symbol(), MODE_TICKVALUE) == 0 ? 1 : MarketInfo(Symbol(), MODE_TICKVALUE));
         double Balance = (AccountEquity() > AccountBalance() ? AccountBalance() : AccountEquity());
         lot = ((Balance - AccountCredit()) * (Auto_Risk / 100)) / Stop_Loss / TickValue; 
         //lot = MathFloor(lot/MarketInfo(Symbol(), MODE_LOTSTEP))* MarketInfo(Symbol(), MODE_LOTSTEP); //���������� ����������� ���� ����
         //lot = MathMin(MathMax(lot, MarketInfo(Symbol(), MODE_MINLOT)), MarketInfo(Symbol(), MODE_MAXLOT)); //��������� ���������� ���� � �����������/������������.
         //Print(lot + " " + StopLoss + " / " + MarketInfo(Symbol(), MODE_TICKSIZE)/0.00001);
         //return (lot);
         }
      else 
         lot = NormalizeDouble(Lots * MathFloor(AccountBalance()/MM_Depo), 2);
         //lot = MathMin(MathMax(lot, MarketInfo(Symbol(), MODE_MINLOT)), MarketInfo(Symbol(), MODE_MAXLOT)); //��������� ���������� ���� � �����������/������������.
      }
      
   if(Martingale && loss <= 100){ lot = NormalizeDouble(lot * MathPow(Multiplier, loss), lotdigit);}
      
   if (MarketInfo(Symbol(), MODE_LOTSTEP) != 0) lot = MathFloor(lot/MarketInfo(Symbol(), MODE_LOTSTEP))* MarketInfo(Symbol(), MODE_LOTSTEP); //���������� ����������� ���� ����
   lot = MathMin(MathMax(lot, MarketInfo(Symbol(), MODE_MINLOT)), MarketInfo(Symbol(), MODE_MAXLOT)); //��������� ���������� ���� � �����������/������������.
   return(lot);         
}
//HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH//
//+------------------------------------------------------------------+
//| get last loss                                                    |
//+------------------------------------------------------------------+
int TotalLoss()
{
   int counter=0,PrevTicket=0, CurrTicket=0;
   
   for(int x=0;x<=OrdersHistoryTotal()-1;x++)
      {
       bool os = OrderSelect(x,SELECT_BY_POS,MODE_HISTORY); 
       if(OrderSymbol()==Symbol() && (MagicNumber == 0 || OrderMagicNumber() == MagicNumber) && OrderType()<=1)
         {
          CurrTicket = OrderTicket();
          if(CurrTicket > PrevTicket) 
            {
             PrevTicket = CurrTicket;  
             if(OrderProfit() < 0) counter++;
             if(OrderProfit() > 0 || counter >= 100) counter=0;}}
            }  
   return(counter);
}  
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
int lastloss()
{   
   int signal = 0; 
   int oldticket = 0;
   int ticket = 0;
   double priceopen = 0;
   double priceclose = 0;
   datetime closetime = 0;
   int otype = -1;   
    
   if (IsTesting() || IsOptimization()) {
   	if (lastticket > 0) {
   		oldticket = lastticket;
   	} else {
   	   if (LogMode < 2) {Print(__FUNCTION__ + ": �� ���������� ����� �������� ������, ������� ���������");}
   	}
   } 
   else {    
   	if (OrdersHistoryTotal() != lasthistorytotal) {
   		lasthistorytotal = OrdersHistoryTotal();
   		for(int i=lasthistorytotal;i>=1;i--) {
   	      if(OrderSelect(i-1,SELECT_BY_POS,MODE_HISTORY)) {
   	         if (TimeCurrent() - OrderCloseTime() > 60*60*24*14) break; //���� ����� ������ 2� ������ - ���������� �����.
   				if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber) {
   				   ticket = OrderTicket();
                  if (ticket > oldticket) oldticket = ticket;			   
   	         }                 
   	      }
            else {
      	      if (LogMode < 3) {
   			      string q = (__FUNCTION__ + ": ����� �� ������. ������ ������ ������.");
                  Print(q);
                  fWriteDataToFile(q);
               }
            }		 
         }
   	} 
   	else {
   		if (lastticket > 0) oldticket = lastticket; // "�������� ��� �� �������� �� ������ ������" - �����������. ��������� ����� ��� ��������� � �������, � ������������� - ��� ������.
   		else if (LogMode < 2) {/*Print("�������� ��� �� �������� �� ������ ������, ������� "+__FUNCTION__ +" ���������.");*/ return(0);}
   	}
   }
   
   if (oldticket > 0) {
   	 if(OrderSelect(oldticket,SELECT_BY_TICKET) && OrderCloseTime() != 0) {
   		if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber) {
   		   priceopen = OrderOpenPrice();
   		   priceclose = OrderClosePrice();               
   		   closetime = OrderCloseTime();
   		   otype = OrderType();
   		}  
   		else {
   			if (LogMode < 3) {
   			   string q = (__FUNCTION__ + ": ���������� � ��������� ������ �� ��������� � ����������� ���������.");
               Print(q);
               fWriteDataToFile(q);   			   
   			}
   		}
      }
   } 
   else {
   	if (OrdersHistoryTotal() > 0 && LogMode < 2) {Print(__FUNCTION__ + ": ����� ����������� ������ �� ������. ��������, �������� ����� �� �������� ������.");}
   }
   
   if (otype == OP_BUY && priceopen-priceclose > sizeloss*_Point && TimeCurrent()-closetime < pause*60) signal = 1;
   
   if (otype == OP_SELL && priceclose-priceopen > sizeloss*_Point && TimeCurrent()-closetime < pause*60) signal = 2;        
   
   if (signal > 0) {
      if (LogMode<2 && lastbarwrite != Time[0]) {

         string q1 = __FUNCTION__ + ": �������� ������ �� ������ ���������� ��������� ������. ����� ����� �� ��� ������.";
         string q2 = __FUNCTION__ + ": �� ����������� �������� ������ ������ �������� " + DoubleToStr(pause-(TimeCurrent()-closetime)/60,0)+" �����";
 
         Print(q1);
         Print(q2);
         fWriteDataToFile(q1);
         fWriteDataToFile(q2);
         lastbarwrite = Time[0];
      }
      last_closetime = closetime;
   }
       
   return (signal);   
}
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
/*double candle()
{
   double c = 0;
   for (int i = 0; i<=barcount; i++) {
      if (iHigh (NULL,TimeFrame,i)-iLow(NULL,TimeFrame,i)>= c) c = iHigh(NULL,TimeFrame,i)-iLow(NULL,TimeFrame,i);
   }
return (c);
}*/

//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
void fTrailingStopFunc(){
   
   double tr_start = Trail_Start*_Point;
   double tr_size = Trail_Size*_Point;
   double tr_step = MathMax(Trail_Step*_Point, MarketInfo(Symbol(),MODE_TICKSIZE)*_Point);
   
   if(OrderType() == OP_BUY){
      if(OrderStopLoss() < OrderOpenPrice() && Bid - OrderOpenPrice() >= tr_start){ //���� SL ��� �� �������
         fModifyPosition(OrderTicket(),OrderOpenPrice(),NormalizeDouble(Bid - tr_start,Digits),OrderTakeProfit(),0,clrGreen);
         return;
      }
      if(OrderStopLoss() >= OrderOpenPrice()){ //���� SL ��� �������
         double dif = Bid - OrderStopLoss() - tr_size;
         if(dif >= tr_step)
            fModifyPosition(OrderTicket(),OrderOpenPrice(),NormalizeDouble(OrderStopLoss() + dif,Digits),
                              OrderTakeProfit(),0,clrGreen);
         return;
      }
   }
   else if(OrderType() == OP_SELL){
      if(OrderStopLoss() > OrderOpenPrice() && OrderOpenPrice() - Ask >= tr_start){ //���� SL ��� �� �������
         fModifyPosition(OrderTicket(),OrderOpenPrice(),NormalizeDouble(Ask + tr_start,Digits),OrderTakeProfit(),0,clrTomato);
         return;
      }
      if(OrderStopLoss() <= OrderOpenPrice()){ //���� SL ��� �������
         double dif = OrderStopLoss() - Ask - tr_size;
         if(dif >= tr_step)
            fModifyPosition(OrderTicket(),OrderOpenPrice(),NormalizeDouble(OrderStopLoss() - dif,Digits),
                              OrderTakeProfit(),0,clrTomato);
         return;
      }
   }
}
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
void fModifyPosition(int ticket,double price,double sl,double tp,datetime expir = 0,color col = clrNONE){
   if (!IsTradeAllowed()) return;
   if(!OrderModify(ticket, price, sl, tp, expir, col)){
	   if (LogMode < 3) {
		  string q = __FUNCTION__ + ": �� ������� �������������� ����� #" +
					  IntegerToString(ticket) + "! " + fMyErDesc();
		  Print(q);
		  fWriteDataToFile(q);
	   }
   }
   Sleep(1000);
}
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
int fGetCCISignal(int period,ENUM_APPLIED_PRICE typeprice,int toplevel,int sh){
   double cci = iCCI(NULL,TimeFrame_CCI,period,typeprice,sh);
   if(cci > toplevel) return(OP_SELL);
   if(cci < -toplevel) return(OP_BUY);
   return(-1);
}
//+------------------------------------------------------------------+
//    ���������� ������, ���� ������ ����� ���������                 +
//    ����� - ����                                                   +
//+------------------------------------------------------------------+
bool fGetRollOver(void){
   //if(use_rollover_filter) { //�� ��������� ������ � ��������.
      if(rtime1 > rtime2){
         if ((_TimeCurrent >= rtime1 && _TimeCurrent < rtime2+60*60*60*24) || (_TimeCurrent >= rtime1-60*60*60*24 && _TimeCurrent < rtime2)) { 
//         if(_TimeCurrent >= rtime1 || _TimeCurrent < rtime2)
            return(true);
      }
   }  
      else{
         if(_TimeCurrent >= rtime1 && _TimeCurrent < rtime2)
            return(true);
      }
   //}
   return(false);
}
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
string timestr(int h,int m) {
   int hh=h;
   if (hh==24) hh=0;
   string h1,m1;
   h1=IntegerToString(hh,2,'0');
   m1=IntegerToString(m,2,'0');
   return h1+":"+m1;
}

string fInfoTradeHours(){

int day=DayOfWeek();
string str="",dayupcase;
//bool showday;

dayupcase=daystring[day];
StringToUpper(dayupcase);

str  = "\n                 "+dayupcase;
//showday = (DayStart[day] != DayEnd[day]) || (DayStart[day] != day);

if ((First_StartTimeStr != First_EndTimeStr) ||
   (Second_StartTimeStr != Second_EndTimeStr)) {
   if (First_StartTimeStr != First_EndTimeStr) {
      str += "\n  First Session:     " + First_StartTimeStr + " - " + First_EndTimeStr;
      }
   else str += "\n"; 
   
   if (Second_StartTimeStr != Second_EndTimeStr) {
      str += "\n  Second Session: " + Second_StartTimeStr + " - " + Second_EndTimeStr;
      }
   else str += "\n"; 
   }
else {
   str += "\n  No trade today\n";
   }

/*if (First_StartTimeStr[day] != First_EndTimeStr[day]) {
   str += "\n  Start Time: " + First_StartTimeStr[day];
   if (showday)    str += ", "+daystring[DayStart[day]];
   str += "\n  End Time:  " + First_EndTimeStr[day];
   if (showday)    str += ", "+daystring[DayEnd[day]];
   }
else {
   str += "\n  No trade today\n";
   }*/
   
return(str);
                  
}

//+------------------------------------------------------------------+
bool IsTime()
{
//   switch(DayOfWeek())
//   {                         
//        case 1: return(IsNow(MONDAY_Start_Trade_Hour,MONDAY_Start_Trade_Minute,MONDAY_End_Trade_Hour,MONDAY_End_Trade_Minute)); 
//        case 2: return(IsNow(TUESDAY_Start_Trade_Hour,TUESDAY_Start_Trade_Minute,TUESDAY_End_Trade_Hour,TUESDAY_End_Trade_Minute));
//        case 3: return(IsNow(WEDNESDAY_Start_Trade_Hour,WEDNESDAY_Start_Trade_Minute,WEDNESDAY_End_Trade_Hour,WEDNESDAY_End_Trade_Minute));
//        case 4: return(IsNow(THURSDAY_Start_Trade_Hour,THURSDAY_Start_Trade_Minute,THURSDAY_End_Trade_Hour,THURSDAY_End_Trade_Minute)); 
//        case 5: return(IsNow(FRIDAY_Start_Trade_Hour,FRIDAY_Start_Trade_Minute,FRIDAY_End_Trade_Hour,FRIDAY_End_Trade_Minute));
//        default: return(false);   
//        
//   }
   
   //_IsFirstSession=false;
   
/*   if(PrevDayStartTime > PrevDayEndTime) {
      if ((_TimeCurrent >= PrevDayStartTime && _TimeCurrent < PrevDayEndTime+60*60*60*24) || (_TimeCurrent >= PrevDayStartTime-60*60*60*24 && _TimeCurrent < PrevDayEndTime)) { 
         _IsFirstSession=true;
         return(true);
      }
   }*/
   if(PrevDayStartTime < PrevDayEndTime) {
      if (_TimeCurrent >= PrevDayStartTime && _TimeCurrent < PrevDayEndTime) { 
         //_IsFirstSession=true;
         return(true);
      }
   }

/*   if(StartTime > EndTime) {
      if ((_TimeCurrent >= StartTime && _TimeCurrent < EndTime+60*60*60*24) || (_TimeCurrent >= StartTime-60*60*60*24 && _TimeCurrent < EndTime)) { 
         return(true);
      }
   }*/
   if(StartTime < EndTime) {
      if (_TimeCurrent >= StartTime && _TimeCurrent < EndTime) { 
         return(true);
      }
   }
   
   return(false);
}
//+------------------------------------------------------------------+
//bool IsNow(int Start_Trade_Hour, int Start_Trade_Minute, int End_Trade_Hour, int End_Trade_Minute)
//{
//
//   if(Start_Trade_Hour > End_Trade_Hour) {
//      if ((_TimeCurrent >= StartTime && _TimeCurrent < EndTime+60*60*60*24) || (_TimeCurrent >= StartTime-60*60*60*24 && _TimeCurrent < EndTime)) { 
//
//         return(true);
//      }
//   }
//   else if(Start_Trade_Hour < End_Trade_Hour) {
//      if (_TimeCurrent >= StartTime && _TimeCurrent < EndTime) { 
//
//         return(true);
//      }
//   }
//   
//   return(false);
//}

//+------------------------------------------------------------------+
//int fGetGMTOffset() {
//   int time = (int)(TimeCurrent() - TimeGMT());
//   double offset = time;
//   offset *= 0.01;
//   offset = MathCeil(offset) * 100;
//   offset = offset/3600;
//   int gmtoffset = (int)NormalizeDouble(offset,0);
//   return(gmtoffset);
//}

int fGetDSTShift() {
   datetime timecurrent=TimeCurrent();
   int _month=TimeMonth(timecurrent);
   if (_month > 11 || _month < 3) { return(0); } // WINTER TIME
   if (_month > 3 && _month < 11) { return(1); } // SUMMER TIME
   
   if (_month == 3) {
      if (TimeDay(timecurrent) > 15) return(1); // SUMMER TIME
      datetime tSeek = StrToTime(IntegerToString(TimeYear(timecurrent))+".03.01");  // determine the first day of march
      int sundaycnt=0;
      // search for the second sunday from the beginning
      for (int i=1;i<=15;i++) {
         if (TimeDayOfWeek(tSeek) == 0) sundaycnt++;
         if (sundaycnt==2) break; // found the second summer of march
         tSeek = tSeek + 86400; // another day
         }
      if (timecurrent >= tSeek) return(1); // already daylight savings time
                           else return(0); // no, it's standard time yet
      }

   if (_month == 11) {
      if (TimeDay(timecurrent) > 10) return(0); // WINTER TIME
      datetime tSeek = StrToTime(IntegerToString(TimeYear(timecurrent))+".11.01");  // determine the first day of november
      // search for the first sunday from the beginning
      for (int i=1;i<=10;i++) {
         if (TimeDayOfWeek(tSeek) == 0)  break; // found the first summer of november
         tSeek = tSeek + 86400; // another day
         }
      if (timecurrent >= tSeek) return(0); // already standard time
                           else return(1); // no, it's daylight savings time yet
      }
      
   return(0);
}

//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//|          ���������� ������ 's' � ���� InpFileName                |
//+------------------------------------------------------------------+
void fWriteDataToFile(string s){

   if (!WriteLogFile) return;
   if(IsTesting() || IsOptimization()) return;
   //--- ������� ���� ��� ������ ������ (���� ��� ���, �� ��������� �������������)
   ResetLastError();
   string data = "";
   data = IntegerToString(Day()) + "." + IntegerToString(Month()) + "." + IntegerToString(Year()) + "  " +
         IntegerToString(Hour()) + ":" + IntegerToString(Minute());
   int file_handle = FileOpen(InpDirectoryName+"//"+InpFileName,FILE_TXT|FILE_READ|FILE_WRITE);
   if(file_handle != INVALID_HANDLE){
      PrintFormat("���� %s ������ ��� ������",InpFileName);
      PrintFormat("���� � �����: %s\\Files\\",TerminalInfoString(TERMINAL_DATA_PATH));
      
      FileSeek(file_handle,0,SEEK_END); //������������ ������ � ����� �����
      
      s = data + "   " + s;
      
      //--- ������� �������� � ����
      FileWrite(file_handle,s);
      //--- ��������� ����
      FileClose(file_handle);
      PrintFormat("������ ��������, ���� %s ������",InpFileName);
     }
   else PrintFormat("�� ������� ������� ���� %s, " + fMyErDesc(),InpFileName);
}
//+------------------------------------------------------------------+ 
//| ������� ������������� �����                                      | 
//+------------------------------------------------------------------+ 
bool fRectLabelCreate(const long             chart_ID    = 0,                 // ID ������� 
                      const string           name        = "RectLabel",       // ��� ����� 
                      const int              sub_window  = 0,                 // ����� ������� 
                      const int              x           = 0,                 // ���������� �� ��� X 
                      const int              y           = 0,                 // ���������� �� ��� Y 
                      const int              width       = 50,                // ������ 
                      const int              height      = 18,                // ������ 
                      const color            back_clr    = C'236,233,216',    // ���� ���� 
                      const ENUM_BORDER_TYPE border      = BORDER_SUNKEN,     // ��� ������� 
                      const ENUM_BASE_CORNER corner      = CORNER_LEFT_UPPER, // ���� ������� ��� Time to����� 
                      const color            clr         = clrRed,            // ���� ������� ������� (Flat) 
                      const ENUM_LINE_STYLE  style       = STYLE_SOLID,       // ����� ������� ������� 
                      const int              line_width  = 1,                 // ������� ������� ������� 
                      const bool             back        = false,             // �� ������ ����� 
                      const bool             selection   = false,             // �������� ��� ����������� 
                      const bool             hidden      = true,              // ����� � ������ �������� 
                      const long             z_order     = 0)                 // Time to������ �� ������� �����
{ 
//--- ������� �������� ������ 
   ResetLastError(); 
   if(ObjectFind(chart_ID,name)==0) return true;
//--- �������� ������������� ����� 
   if(!ObjectCreate(chart_ID,name,OBJ_RECTANGLE_LABEL,sub_window,0,0)) 
     { 
	  if (LogMode<3) {
		  Print(__FUNCTION__, 
				": �� ������� ������� ������������� �����! " + fMyErDesc()); 
      } 
	  return(false); 
     } 
//--- ��������� ���������� ����� 
   ObjectSetInteger(chart_ID,name,OBJPROP_XDISTANCE,x); 
   ObjectSetInteger(chart_ID,name,OBJPROP_YDISTANCE,y); 
//--- ��������� ������� ����� 
   ObjectSetInteger(chart_ID,name,OBJPROP_XSIZE,width); 
   ObjectSetInteger(chart_ID,name,OBJPROP_YSIZE,height); 
//--- ��������� ���� ���� 
   ObjectSetInteger(chart_ID,name,OBJPROP_BGCOLOR,back_clr); 
//--- ��������� ��� ������� 
   ObjectSetInteger(chart_ID,name,OBJPROP_BORDER_TYPE,border); 
//--- ��������� ���� �������, ������������ �������� ����� ������������ ���������� ����� 
   ObjectSetInteger(chart_ID,name,OBJPROP_CORNER,corner); 
//--- ��������� ���� ������� ����� (� ������ Flat) 
   ObjectSetInteger(chart_ID,name,OBJPROP_COLOR,clr); 
//--- ��������� ����� ����� ������� ����� 
   ObjectSetInteger(chart_ID,name,OBJPROP_STYLE,style); 
//--- ��������� ������� ������� ������� 
   ObjectSetInteger(chart_ID,name,OBJPROP_WIDTH,line_width); 
//--- ��������� �� �������� (false) ��� ������ (true) ����� 
   ObjectSetInteger(chart_ID,name,OBJPROP_BACK,back); 
//--- ������� (true) ��� �������� (false) ����� ����������� ����� ����� 
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTABLE,selection); 
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTED,selection); 
//--- ������ (true) ��� ��������� (false) ��� ������������ ������� � ������ �������� 
   ObjectSetInteger(chart_ID,name,OBJPROP_HIDDEN,hidden); 
//--- ��������� Time to������ �� ��������� ������� ������� ���� �� ������� 
   ObjectSetInteger(chart_ID,name,OBJPROP_ZORDER,z_order); 
//--- �������� ���������� 
   return(true); 
}
//+------------------------------------------------------------------+ 
//| ������� ������������� �����                                      | 
//+------------------------------------------------------------------+ 
bool fRectLabelDelete(const long   chart_ID   = 0,           // ID ������� 
                      const string name       = "RectLabel") // ��� ����� 
{ 
//--- ������� �������� ������ 
   ResetLastError(); 
//--- ������ ����� 
   if (ObjectFind(chart_ID,name) != -1) {
      if(!ObjectDelete(chart_ID,name)) 
        { 
   	  if (LogMode<3) {
   		  Print(__FUNCTION__, 
   				": �� ������� ������� ������������� �����! " + fMyErDesc()); 
   	  }	
         return(false); 
        }
    }   
//--- �������� ���������� 
   return(true); 
}
//+------------------------------------------------------------------+
void DrawChannel(string dir, double pr1, color clr=clrYellow){ //��������� ����� ��� ������ �� ������ BB

   if (Bars<2) return;
   string name=_Symbol+" "+dir+TimeToStr(Time[0]);
   string name_prev=_Symbol+" "+dir+TimeToStr(Time[1]);
   
   if(ObjectFind(name) < 0){
      double pr2 = pr1;
      if(ObjectFind(name_prev) == 0) pr2 = ObjectGetDouble(0,name_prev,OBJPROP_PRICE,1);
      
      ObjectCreate(name, OBJ_TREND,0,Time[1],pr2,Time[0],pr1);
      ObjectSet(name, OBJPROP_COLOR, clr);
      ObjectSetInteger(0,name,OBJPROP_RAY_RIGHT,false); 
      ObjectSetInteger(0,name,OBJPROP_WIDTH,1);
      //ObjectSet(name,OBJPROP_BACK,true);
      ObjectSet(name,OBJPROP_SELECTABLE,false);
   }
}
//+------------------------------------------------------------------+
void DrawWarn(string text, color col){
   if(!VisualDebug) return;
   
   string obid=_Symbol+TimeToStr(Time[0]);
   if(ObjectFind(obid)!=-1) ObjectDelete(obid);
   
   ENUM_ANCHOR_POINT anc;
   double price;

   if((WindowPriceMax()+WindowPriceMin())/2 < Bid){
      price=channel_lower-Entry_Break*2*_Point;
      anc=ANCHOR_RIGHT;
   }else{
      price=channel_upper+Entry_Break*2*_Point;
      anc=ANCHOR_LEFT;
   }
   
   ObjectCreate(obid, OBJ_TEXT, 0, Time[0], price);
   ObjectSet(obid, OBJPROP_ANGLE, 90);
   ObjectSet(obid,OBJPROP_ANCHOR,anc);
   ObjectSet(obid,OBJPROP_BACK,false);
   ObjectSetText(obid,text,10,"Arial",col);
}

void UpdateInfoPanel(){
   if(_RealTrade || IsVisualMode()){
      //if(day_of_trade != DayOfWeek()){
         if(TP_perc > 0) tp_info = "\n  Dynamic Take Profit = " + IntegerToString(TP_perc) + "%";
         else tp_info = "\n  Dynamic Take Profit: OFF";
         if(X > 0 && Y > 0) be_info = "\n  Breakeven: ON"; //���������
         else be_info = "\n  Breakeven: OFF";
         TradeHoursFirst = fInfoTradeHours();
         if (LastUpdateTime<TimeCurrent()) {
            CheckArbitrage();
            LastUpdateTime=TimeCurrent();
            }
      //   day_of_trade = DayOfWeek();
      //}

      if(showinfopanel) {
         string warn_trading="";
         if (SetEqSymbol) warn_trading="Set name: [OK] "; else warn_trading="Set name: [WARNING] ";
         if (!IsTradeAllowed()) warn_trading = "Trading is disabled !!!"; 
         info_panel =
              "\n ----------------------------------------------------" 
            + "\n              GENERIC A-TLP" 
            + "\n ----------------------------------------------------" 
            + "\n  A FREE PRODUCT POWERED BY" 
            + "\n       http://TRADELIKEAPRO.ru" 
            + "\n ----------------------------------------------------" 
            + "\n "+set_name_info
            + "\n "+warn_trading
            + "\n ----------------------------------------------------" 
            + TradeHoursFirst
            + "\n  ---------------------------------------------------"
            + tp_info
            + "\n  Take Profit = " + DoubleToStr(Take_Profit/PipsDivided,1) + " pips"
            + "\n  Stop Loss = " + DoubleToStr(Stop_Loss/PipsDivided,1) + " pips"
            + be_info
            + "\n  ---------------------------------------------------"
            + "\n  Max Spread = " + maxspread /*+ (Max_Spread_On_Close>0?" / Close: "+DoubleToStr(Max_Spread_On_Close/ PipsDivided,1):"")*/
            + "\n  Spread = " + DoubleToStr((Ask - Bid) / _Point / PipsDivided, 1) + " pips";

         if (Max_Spread > 0) {
            if (Ask - Bid > Max_Spread * _Point) info_panel = info_panel + " - HIGH !!!";
            else info_panel = info_panel + " - NORMAL";
            }
         info_panel = info_panel
            + risk_info
            + "\n  Max Orders = "+IntegerToString(TotalOrders) /*+ (Hedging?", with hedging = "+IntegerToString(TotalOrders*2):"")*/
            + "\n  Trading Lots = " + DoubleToStr(lots, 2) + (TotalOrders>1?"*" + IntegerToString(TotalOrders) + " = " + DoubleToStr(lots*TotalOrders, 2):"") /*+(Hedging?", hedg = "+DoubleToStr(lots*TotalOrders*2, 2):"")*/
            + "\n  ---------------------------------------------------"
            + filter_info
            + "\n  5. News Filter: "+(UseNewsFilter?"ON"+(_IsNews?", Activated":""):"OFF");
          if (MaxAmountCurrencyMult>0)  info_panel = info_panel 
            + "\n  ---------------------------------------------------"
            +CheckString();
          info_panel = info_panel 			
            + "\n  ---------------------------------------------------";
         previnfopanelcolor=infopanelcolor;   
         infopanelcolor=(IsTradeAllowed()?Col_info:Col_info2);
         if (infopanelcolor != previnfopanelcolor) {
            ObjectSetInteger(0,"info_panel",OBJPROP_BGCOLOR,infopanelcolor); 
            }

         Comment(info_panel);
      }
   }
}

int CountOrder(int Order_Type) {
   int orders=0;
   
   for(int i=OrdersTotal()-1;i>=0;i--){
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES)==false) continue;
      if(OrderSymbol()!=_Symbol || OrderMagicNumber() != MagicNumber || OrderType() > OP_SELL) continue;
      
      if(Order_Type == OrderType() || Order_Type == -1) orders++;
   }
   return orders;
}

double GetMaxMinOpenPrice(int ordertype,_MaxMin maxtype){
   double maxminprice=0;
      for(int i=OrdersTotal()-1;i>=0;i--){
         if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES)==false) continue;
         if(OrderSymbol()!=_Symbol || OrderMagicNumber() != MagicNumber || OrderType() > OP_SELL) continue;
         if ((ordertype != -1) && (OrderType() != ordertype)) continue;
         if (maxminprice==0) maxminprice=OrderOpenPrice();
         if ((maxtype==MAX) && (OrderOpenPrice()>maxminprice)) maxminprice=OrderOpenPrice();
         if ((maxtype==MIN) && (OrderOpenPrice()<maxminprice)) maxminprice=OrderOpenPrice();
      }
   return maxminprice;
}

datetime GetLastOpenTime(int ordertype){
   datetime opentime=0;
      for(int i=OrdersTotal()-1;i>=0;i--){
         if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES)==false) continue;
         if(OrderSymbol()!=_Symbol || OrderMagicNumber() != MagicNumber || OrderType() > OP_SELL) continue;
         if ((ordertype != -1) && (OrderType() != ordertype)) continue;
         if (opentime==0) opentime=OrderOpenTime();
         if  (OrderOpenTime()>opentime) opentime=OrderOpenTime();
      }
   return opentime;
}

datetime GetFirstOpenTime(int ordertype){
   datetime opentime=0;
      for(int i=OrdersTotal()-1;i>=0;i--){
         if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES)==false) continue;
         if(OrderSymbol()!=_Symbol || OrderMagicNumber() != MagicNumber || OrderType() > OP_SELL) continue;
         if ((ordertype != -1) && (OrderType() != ordertype)) continue;
         if (opentime==0) opentime=OrderOpenTime();
         if  (OrderOpenTime()<opentime) opentime=OrderOpenTime();
      }
   return opentime;
}

int GetLastOrderTicket(int ordertype){
   datetime opentime=0;
   int last_order_ticket=0;
      for(int i=OrdersTotal()-1;i>=0;i--){
         if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES)==false) continue;
         if(OrderSymbol()!=_Symbol || OrderMagicNumber() != MagicNumber || OrderType() > OP_SELL) continue;
         if ((ordertype != -1) && (OrderType() != ordertype)) continue;
         if (opentime==0) { opentime=OrderOpenTime();last_order_ticket=OrderTicket(); }
         if  (OrderOpenTime()>opentime) { opentime=OrderOpenTime(); last_order_ticket=OrderTicket(); }
      }
   return last_order_ticket;
}


double GetOrdersTotalProfit(int ordertype){
   double orderprofit=0;
      for(int i=OrdersTotal()-1;i>=0;i--){
         if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES)==false) continue;
         if(OrderSymbol()!=_Symbol || OrderMagicNumber() != MagicNumber || OrderType() > OP_SELL) continue;
         if ((ordertype != -1) && (OrderType() != ordertype)) continue;
         if (OrderType()==OP_BUY) { orderprofit += Bid - OrderOpenPrice(); }
         if (OrderType()==OP_SELL) { orderprofit += OrderOpenPrice() - Ask; }
      }
   return orderprofit;
}
   
bool IsNews(int TimeBefore, int TimeAfter) {
   double _news;
   int news_offset = GMT_Offset;
   if (DST) news_offset += fGetDSTShift();
   _news=iCustom(Symbol(),0,"urdala_news_investing.com_mod",TimeBefore,TimeAfter,news_offset,Vhigh,Vmedium,Vlow,NewsSymb,true,false,false,highc,mediumc,lowc,0, 0);
   int Error = GetLastError();
   if(Error != 0) {
      Print("IsNews() Error ",Error,": ",ErrorDescription(Error),". News filter disabled.");
      UseNewsFilter=false;
      }
   if (_news==1) return(true); else return(false);
}   

double OnTester(){
  double Res = 0;
  double MaxDD = TesterStatistics(STAT_EQUITY_DD);
  if (MaxDD != 0)
      Res = TesterStatistics(STAT_PROFIT) / MaxDD;
  return Res;
}

template<typename T>
bool IsError(string fname, T val, bool skipif0=true) {
   bool result=false;
   int Error = GetLastError(); 
   if (Error == 4066 || Error == 4073 || Error == 4074 || (skipif0 && val == 0)) result=true;
   else if (Error != 0) { 
      Print(fname," Error ",Error,": ",ErrorDescription(Error)); 
      Print(fname," = ",val); 
      }
   //if (Error != 0 && result) Print("IsError(): error ",Error,", function ",fname);   
   return(result);
}

//+------------------------------------------------------------------+

string StrDelSpaces( string Str )
{
  int Pos, Length;

  Str = StringTrimLeft(Str);
  Str = StringTrimRight(Str);

  Length = StringLen(Str) - 1;
  Pos = 1;

  while (Pos < Length)
    if (StringGetChar(Str, Pos) == ' ')
    {  
      Str = StringSubstr(Str, 0, Pos) + StringSubstr(Str, Pos + 1, 0);
      Length--;
    }
    else 
      Pos++;

  return(Str);
}

int StrToStringS( string Str, string Razdelitel, string &Output[] )
{
  int Pos, LengthSh;
  int Count = 0;

  Str = StrDelSpaces(Str);
  Razdelitel = StrDelSpaces(Razdelitel);

  LengthSh = StringLen(Razdelitel);

  while (TRUE)
  {
    Pos = StringFind(Str, Razdelitel);
    Output[Count] = StringSubstr(Str, 0, Pos);
    Count++;
 
    if (Pos == -1)
      break;
 
    Pos += LengthSh;
    Str = StringSubstr(Str, Pos);
  }

  return(Count);
}

int CurrencyPos( string Str )
{
  int i = 0;
  
  while (Currency[i] != Str)  
    i++;
  
  return(i);
}

void CheckArbitrage()
{
  int i;
  string Str;
  
  for (i = 0; i < AmountCurrency; i++)
    Volumes[i] = 0;
  
  for (i = OrdersTotal() - 1; i >= 0; i--)
  {
    if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES)==false) continue;
    
    if (OrderType() == OP_BUY)
    {
      Str = StringSubstr(OrderSymbol(), 0, 3);
      Volumes[CurrencyPos(Str)] += OrderLots();
      
      Str = StringSubstr(OrderSymbol(), 3, 3);
      Volumes[CurrencyPos(Str)] -= OrderLots();
    }
    else if (OrderType() == OP_SELL)
    {
      Str = StringSubstr(OrderSymbol(), 0, 3);
      Volumes[CurrencyPos(Str)] -= OrderLots();
      
      Str = StringSubstr(OrderSymbol(), 3, 3);
      Volumes[CurrencyPos(Str)] += OrderLots();
    }
  }
  
  return;
}

string CheckString()
{
  int i,x=1;
  string Str = "";
  Str = Str + "\n  Max"+StringSubstr(Symbol(), 0, 3)+ " " + DoubleToStr(MaxAmountCurrencyMult*lots, 2) + " lots  Max"+StringSubstr(Symbol(), 3, 3)+ " " + DoubleToStr(MaxAmountCurrencyMult*lots, 2) + " lots ";
  for (i = 0; i < AmountCurrency; i++)
    if (Volumes[i] != 0)
       {
      if(x%2 != 0 ) Str = Str + "\n  ";
      x++;
      Str = Str + Currency[i] + " = " + DoubleToStr(Volumes[i], 2) + " lots  ";
      
      
      }
  return(Str);
}


//+------------------------------------------------------------------+
string fMyErDesc(int err=-1){
   int aErrNum;
   if(err == -1){
      aErrNum = GetLastError();
   }
   else{
      aErrNum = err;
   }

   string pref="������ �: "+IntegerToString(aErrNum)+" - ";
   switch(aErrNum){
      case 0:   return(pref+"��� ������. �������� �������� ������ �������.");
      case 1:   return(pref+"��� ������, �� ��������� ����������. (OrderModify �������� " +
                              "�������� ��� ������������� �������� ������ �� ����������. " +
                              "���������� �������� ���� ��� ��������� �������� � ��������� �������.)");
      case 2:   return(pref+"����� ������. ���������� ��� ������� �������� �������� �� " +
                              "��������� �������������. �������� ������������� ������������ " +
                              "������� � ���������� ��������.");
      case 3:   return(pref+"������������ ���������. � �������� ������� �������� ������������ " +
                              "���������, ��Time to���, ������������ ������, ������������ �������� " +
                              "��������, ������������� ���������� ���������� ����, " +
                              "�������������� ����� ������ � �.�. ���������� �������� ������ ���������.");
      case 4:   return(pref+"�������� ������ �����. ����� ��������� ������� ����� ���������� " +
                              "������� ���������� ������� (�� ���������� �����).");
      case 5:   return(pref+"������ ������ ����������� ���������. ���������� ���������� " +
                              "��������� ������ ����������� ���������.");
      case 6:   return(pref+"��� ����� � �������� ��������.  ���������� ���������, ��� ����� " +
                              "�� �������� (��Time to���, Time to ������ ������� IsConnected) � " +
                              "����� ��������� ���������� ������� (�� 5 ������) ��������� �������.");
      case 7:   return(pref+"������������ ����.");
      case 8:   return(pref+"������� ������ �������. ���������� ��������� ������� ��������, " +
                              "�������� ������ ���������.");
      case 9:   return(pref+"������������ ��������, ���������� ���������������� �������");
      case 64:  return(pref+"���� ������������. ���������� ���������� ��� ������� �������� ��������.");
      case 65:  return(pref+"������������ ����� �����. ���������� ���������� ��� ������� " +
                              "�������� ��������.");
      case 128: return(pref+"����� ���� �������� ���������� ������. ������, ��� ����������� " +
                              "��������� ������� (�� �����, ��� ����� 1 ������), ���������� " +
                              "���������, ��� �������� �������� ������������� �� ������ " +
                              "(����� ������� �� ���� �������, ���� ������������ ����� �� " +
                              "��� ������ ��� �����, ���� ������������ ������� �� ���� �������)");
      case 129: return(pref+"������������ ���� bid ��� ask, ��������, ����������������� " +
                              "����. ���������� ����� �������� �� 5 ������ �������� " +
                              "������ Time to ������ ������� RefreshRates � ��������� �������. " +
                              "���� ������ �� ��������, ���������� ���������� ��� ������� " +
                              "�������� �������� � �������� ������ ���������.");
      case 130: return(pref+"������������ �����. ������� ������� ����� ��� ����������� " +
                              "������������ ��� ����������������� ���� � ������ (��� � " +
                              "���� �������� ����������� ������). ������� ����� ��������� " +
                              "������ � ��� ������, ���� ������ ��������� ��-�� ����������� " +
                              "����. ���������� ����� �������� �� 5 ������ �������� ������ " +
                              "Time to ������ ������� RefreshRates � ��������� �������. ���� " +
                              "������ �� ��������, ���������� ���������� ��� ������� " +
                              "�������� �������� � �������� ������ ���������.");
      case 131: return(pref+"������������ �����, ������ � ���������� ������. ���������� " +
                              "���������� ��� ������� �������� �������� � �������� ������ ���������.");
      case 132: return(pref+"����� ������. ����� ��������� ������� ����� ���������� ������� " +
                              "���������� ������� (�� ���������� �����).");
      case 133: return(pref+"�������� ���������. ���������� ���������� ��� ������� �������� ��������.");
      case 134: return(pref+"������������ ������� ��� ���������� ��������. ��������� ������ " +
                              "� ���� �� ����������� ������. ������� ����� ��������� ����� " +
                              "�������� �� 5 ������, �������� �����, �� ���� ���� ��������� � " +
                              "������������� ������� ��� ���������� ��������.");
      case 135: return(pref+"���� ����������. ����� ��� �������� �������� ������ Time to ������ " +
                              "������� RefreshRates � ��������� �������.");
      case 136: return(pref+"��� ���. ������ �� �����-�� Time to���� (��Time to���, � ������ ������ " +
                              "��� ���, ���������������� ����, ������� �����) �� ��� ��� ��� " +
                              "�������. ���������� ����� �������� �� 5 ������ �������� ������ " +
                              "Time to ������ ������� RefreshRates � ��������� �������.");
      case 137: return(pref+"������ �����");
      case 138: return(pref+"����� ����. ����������� ���� ��������, ���� ���������� bid � " +
                              "ask. ����� ��� �������� �������� ������ Time to ������ ������� " +
                              "RefreshRates � ��������� �������. ���� ������ �� ��������, " +
                              "���������� ���������� ��� ������� �������� �������� � �������� " +
                              "������ ���������.");
      case 139: return(pref+"����� ������������ � ��� ��������������. . ���������� ���������� " +
                              "��� ������� �������� �������� � �������� ������ ���������.");
      case 140: return(pref+"��������� ������ �������. ��������� �������� SELL ������.");
      case 141: return(pref+"������� ����� ��������. ���������� ��������� ������� " +
                              "��������, �������� ������ ���������.");
      case 142: return(pref+"����� ��������� � �������. ��� �� ������, � ���� �� ����� " +
                              "�������������� ����� ���������� ���������� � �������� " +
                              "��������. ���� ��� ����� ���� ������� � ������ ������, " +
                              "����� �� ����� ���������� �������� �������� ��������� " +
                              "����� � ����������� �������������� �����. ���������� " +
                              "������������ ��� �� ��� � ������ 128.");
      case 143: return(pref+"����� Time to��� ������� � ����������. ���� �� ����� �������������� " +
                              "����� ���������� ���������� � �������� ��������. ����� " +
                              "���������� �� ��� �� Time to����, ��� � ��� 142. ���������� " +
                              "������������ ��� �� ��� � ������ 128.");
      case 144: return(pref+"����� ����������� ����� �������� Time to ������ ������������� " +
                              "������. ���� �� ����� �������������� ����� ���������� " +
                              "���������� � �������� ��������.");
      case 145: return(pref+"����������� ���������, ��� ��� ����� ������� ������ � " +
                              "����� � ������������ ��-�� ���������� ������� ����������. " +
                              "����� �� �����, ��� ����� 15 ������, �������� ������ Time to " +
                              "������ ������� RefreshRates � ��������� �������.");
      case 146: return(pref+"���������� �������� ������. ��������� ������� ������ ����� " +
                              "����, ��� ������� IsTradeContextBusy ������ FALSE.");
      case 147: return(pref+"������������� ���� ��������� ������ ��������� ��������. " +
                              "�������� ����� ��������� ������ � ��� ������, ���� " +
                              "�������� �������� expiration.");
      case 148: return(pref+"���������� �������� � ���������� ������� �������� �������, " +
                              "�������������� ��������. ����� �������� ������� � " +
                              "���������� ������ �������� ������ ����� �������� ��� " +
                              "�������� ������������ ������� ��� �������.");
      case 149: return(pref+"������� ������� ��������������� ������� � ��� ������������ " +
                              "� ������, ���� ������������ ���������. ������� ���������� " +
                              "������� ������������ ��������������� �������, ���� ���������� " +
                              "�� ���� ������� ����� �������� ��������, ���� �������� " +
                              "������ ���������.");
      case 150: return(pref+"������� ������� ������� �� ����������� � ������������ � �������� FIFO");
      //---- ���� ������ ���������� MQL4-��������� (���������)
      case 4000: return(pref+"��� ������");
      case 4001: return(pref+"������������ ��������� �������");
      case 4002: return(pref+"������ ������� - ��� ���������");
      case 4003: return(pref+"��� ������ ��� ����� �������");
      case 4004: return(pref+"������������ ����� ����� ������������ ������");
      case 4005: return(pref+"�� ����� ��� ������ ��� �������� ����������");
      case 4006: return(pref+"��� ������ ��� ���������� ���������");
      case 4007: return(pref+"��� ������ ��� ��������� ������");
      case 4008: return(pref+"�������������������� ������");
      case 4009: return(pref+"�������������������� ������ � �������");
      case 4010: return(pref+"��� ������ ��� ���������� �������");
      case 4011: return(pref+"������� ������� ������");
      case 4012: return(pref+"������� �� ������� �� ����");
      case 4013: return(pref+"������� �� ����");
      case 4014: return(pref+"����������� �������");
      case 4015: return(pref+"������������ �������");
      case 4016: return(pref+"�������������������� ������");
      case 4017: return(pref+"������ DLL �� ���������");
      case 4018: return(pref+"���������� ��������� ����������");
      case 4019: return(pref+"���������� ������� �������");
      case 4020: return(pref+"������ ������� ������������ ������� �� ���������");
      case 4021: return(pref+"������������ ������ ��� ������, ������������ �� �������");
      case 4022: return(pref+"������� ������");
      case 4050: return(pref+"������������ ���������� ���������� �������");
      case 4051: return(pref+"������������ �������� ��������� �������");
      case 4052: return(pref+"���������� ������ ��������� �������");
      case 4053: return(pref+"������ �������");
      case 4054: return(pref+"������������ ������������� �������-���������");
      case 4055: return(pref+"������ ����������������� ����������");
      case 4056: return(pref+"������� ������������");
      case 4057: return(pref+"������ ��������� ����������� ����������");
      case 4058: return(pref+"���������� ���������� �� ����������");
      case 4059: return(pref+"������� �� ��������� � �������� ������");
      case 4060: return(pref+"������� �� ���������");
      case 4061: return(pref+"������ �������� �����");
      case 4062: return(pref+"��������� �������� ���� string");
      case 4063: return(pref+"��������� �������� ���� integer");
      case 4064: return(pref+"��������� �������� ���� double");
      case 4065: return(pref+"� �������� ��������� ��������� ������");
      case 4066: return(pref+"����������� ������������ ������ � ��������� ����������");
      case 4067: return(pref+"������ Time to ���������� �������� ��������");
      case 4099: return(pref+"����� �����");
      case 4100: return(pref+"������ Time to ������ � ������");
      case 4101: return(pref+"������������ ��� �����");
      case 4102: return(pref+"������� ����� �������� ������");
      case 4103: return(pref+"���������� ������� ����");
      case 4104: return(pref+"������������� ����� ������� � �����");
      case 4105: return(pref+"�� ���� ����� �� ������");
      case 4106: return(pref+"����������� ������");
      case 4107: return(pref+"������������ �������� ���� ��� �������� �������");
      case 4108: return(pref+"�������� ����� ������");
      case 4109: return(pref+"�������� �� ���������. ���������� �������� ����� <��������� " +
                              "��������� ���������> � ��������� ��������");
      case 4110: return(pref+"������� ������� �� ��������� - ���������� ��������� �������� ��������");
      case 4111: return(pref+"�������� ������� �� ��������� - ���������� ��������� �������� ��������");
      case 4200: return(pref+"������ ��� ����������");
      case 4201: return(pref+"��������� ����������� �������� �������");
      case 4202: return(pref+"������ �� ����������");
      case 4203: return(pref+"����������� ��� �������");
      case 4204: return(pref+"��� ����� �������");
      case 4205: return(pref+"������ ��������� �������");
      case 4206: return(pref+"�� ������� ��������� �������");
      case 4207: return(pref+"������ Time to ������ � ��������");
      default:   return(pref+"�������������� ����� ������");
   }
}