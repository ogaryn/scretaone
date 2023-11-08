//+------------------------------------------------------------------+
//|                                                Generic A-TLP.mq4 |
//|                                                 Trade Like A Pro |
//|                                          http://tradelikeapro.ru |
//|                                                      open source |
//|   Версия 9.5  - filters have been added to the CCI                         |
//|   Версия 10.0 - held small code optimization             |
//|   Версия 11.1 - разрешены встречные сделки, введен параметр,     |
//|               ограничивающий открытие сделок в одну сторону      |
//|               в течение заданного количества минут               |
//|   Версия 11.2 - все вышесказанное соединено в один советник      |
//|               и далее будет неотемлемой его частью;              |
//|               - параметр расчета периода StDev вынесен во внешние|
//|               переменные                                         |
//|   Версия 11.3 - исправлена ошибка, связанная с CCI               |
//|   Версия 11.4 - исправлена ошибка информ-панели;                 |
//|               - добавлена возможность торговли на каждом тике;   |
//|               - проверка на минимальную волатильность теперь     |
//|               осуществляется Time to входе в блок открытия позиций   |
//|   Версия 11.6 - добавлен трейлинг стоп                           |
//|   Версия 11.7 - исправлены ошибки                                |
//|   Версия 11.7.31 - расчет канала отвязан от ТФ15                 |
//|   Версия 11.9 - добавлена пауза после убыточной сделки,          |
//|               - фильтр по макс. длине свечи,                     |
//|               - авторазмер лота от баланса, а не от эквити,      |
//|               - возможность запрета тралла в ролловер,           |
//|               - Time toнт в журнале о наличии сиглала Time to высоком    |
//|               спреде                                             |
//|   Версия 11.9.1 - фильтры сделаны отключаемыми (пауза после      |
//|               убытка и макс.свеча)                               |
//|               - внедрены два режима работы функции lastloss (для |
//|               тестера и для торговли)                            |
//|               - настраиваемое логирование (переменная LogMode)   |
//|               - небольшая оптимизация кода                       |
//|   Версия 11.9.2 - подправлено информирование                     |
//|   Версия 11.9.3 - исправлены ошибки                              |
//|   Версия 11.9.4 - небольшие дополнения кода                      |
//|   Версия 11.10.6 - изменен вид выбора времени торговли для       |
//|                    возможность его оптимизации.                  |
//|                  - убраны лишние(дублирующие) внешние параметры. |
//|                    Теперь функции отключаются Time to значении 0.    |
//|                  - сортировка внешних параметров                 | 
//|   Версия 11.10.7 - исправлен расчет Auto_Risk                    |  
//|                  - удаление лишних переменных                    |
//|   Версия 11.10.8 - мелкие правки                                 | 
//|   Версия 11.10.9 - убраны лишние переменные                      |
//|                  - подкорректировано сообщение фильтра по паузе  |
//|   Версия 12.01.14 -переработка кода, улучшения для оптимизации   |
//|                  - дополнительные правки, улучшения              |
//|                     every_tick теперь запускает цикл 1 раз       |
//|                     за минуту в момент открытия нового бара по М1|
//|   Версия 12.01.15 - возврат MaxDailyRange в настройки и          |
//|                     every_tick теперь запускает цикл 1 раз       |
//|                     за минуту в момент открытия нового бара по М1|
//|   Версия 12.01.16 - разрешение на торговлю в Second Session без  |
//|                     торговли в First Session                     |
//|   Версия 12.01.17 - исправлена ошибка с сессиями                 |
//|                   - верхний и нижний уровни CCI объединены в один|
//|   Версия 12.01.18 - исправлена ошибка нормализации  параметра    |
//|                     MaxDailyRange                                |
//|   Версия 12.01.19 - запись в лог Time to Time toвышении MaxDailyRange и  |
//|                     фильтру CCI                                  |
//|                   - BB_Deviation теперь используется             |
//|   Версия 12.01.20 - граница канала теперь проверяется по цене    |
//|                     Ask Time to открытии сделки для покупок, и Time to   |
//|                     закрытии для продаж                          |
//|   Версия 12.01.21 - Введен параметр OnlyBid, Time to true отменяет   |
//|                     изменение введенное в 12.01.20               |
//|                   - Параметр ShowCandleDots заменен на           |
//|                     VisualDebug, рисует канал с установленным    |
//|                     отступом, и отображает срабатывание фильтров |
//|                   - Добавлена проверка SetName, если валюта      |
//|                     графика есть в SetName в инфопанели рядом с  |
//|                     будет выведен SetName и [OK] иначе [WARNING] |
//|                   - Оптимизация для ускорения работы             |
//|   Версия 12.01.22 - исправлена ошибка с отображением спреда      |
//|                   - Time to Max_Spread=0 теперь не выводится строка  |
//|                     NORMAL/HIGH для спреда                       |
//|                   - добавлена функция OnTester, возвращающая     |
//|                     отношение Time toбыли к максимальной просадке    |
//|   Версия 12.01.23 - исправлен RollOver Filter                    |
//|                   - в VisualDebug добавлен второй канал bb       |
//|                     на расстоянии Exit_Distance, Time to наличии     |
//|                     открытых позиций, Time to выходе на графике будет| 
//|                     написано по какому фильтру вышли. Надписи    |
//|                     сделал дальше от цены, за границей канала.   |
//|                   - все сообщения в пунктах Time toведены к          |
//|                     четырехзнаку                                 |
//|   Версия 12.02.24 - Изменен тип выбора времени торговли          |
//|   Версия 12.02.25 - Оптимизация для ускорения работы             |
//|   Версия 12.26    - Переменная use_rollover_filter заменена на   |
//|                     OpenOrdersInRollover и CloseOrdersInRollover |
//|                   - Корректный перенос времени торговли через    |
//|                     полночь                                      |
//|   Версия 12.27    - Значения часов торговли, большие, чем 24,    |
//|                     переносятся на следующий день                |
//|   Версия 12.28    - Исправлен вывод времени в инфо-панель        |
//|   Версия 12.30    - Time to Offюченной автоторговле выводится       |
//|                     предупреждение                               |
//|                   - Добавлен параметр TradeAllDaysInTheSameTime, |
//|                     включающий торговлю во все дни в одно и то же|
//|                     время (используется время понедельника)      |
//|                   - Добавлены параметры включения/Offючения     |
//|                     сессий отдельных дней                        |
//|                   - Добавлен параметр CheckMDRAfter0, включающий |
//|                     проверку MaxDailyRange после полуночи        |
//|                   - Расчёт манименеджмента от минимума эквити и  |
//|                     баланса                                      |
//|                   - Добавлен параметр MM_Depo, как расчёт лота   |
//|                     размер лота (Lots) на размер депо (MM_Depo)  |
//|   Версия 12.31    - Параметр maxcandle теперь вместо длины одной |
//|                     свечи рассчитывается как диапазон движения   |
//|                     цены за заданное количество баров            |
//|                   - Исправлен вывод в инфо-панель                |
//|                   - Проверка на минимальный/максимальный лот     |
//|                   - встроен VisualDebug трендовыми линиями       |
//|                   - добавлен Max_Spread_mode2 как альтернатива   |
//|                     ролловер фильтру, Time to значении больше нуля   |
//|                     Max_Spread фильтрует только покупки Time to      |
//|                     открытии, а Max_Spread_mode2 - продажи Time to   |
//|                     закрытии                                     |
//|   Версия 12.32    - Исправлен вывод в журнал                     |
//|                   - Исправлены ошибки                            |
//|                   - Возможность открывать несколько ордеров      |
//|   Версия 12.33    - Изменено построение линий границ канала      |
//|                   - Исправлен фильтр по МА                       |
//|                   - Исправлен неработающий фильтр выхода по ССI  |
//|   Версия 12.34    - Исправлена ошибка в построении границ канала |
//|                   - Исправлен запрет трейлинг-стопа в ролловер   |
//|                   - Теперь количество ордеров задаётся одним     |
//|                     параметром                                   |
//|   Версия 12.35    - Добавлен фильтр для максимальный обьем валюты|
//|                     Рекомендованное знач MaxAmountCurrencyMult=2 |
//|   Версия 12.36    - Параметр Max_Spread_mode2 заменён на         |
//|                     CheckSpreadOnSellOpen, включающий проверку   |
//|                     спреда Time to продажах, и Max_Spread_On_Close,  |
//|                     проверяющий спред Time to закрытии продаж        |
//|                   - Добавлен параметр паузы между сделками Time to   |
//|                     открытии нескольких ордеров                  |
//|                   - Добавлен параметр одновременного закрытия    |
//|                     группы ордеров                               |
//|                   - В журнал теперь пишется цена закрытия        |
//|                     и Time toбыль                                    |
//|                   - Исправлен вывод сообщений в журнал           |
//|                     Time to Offюченной автоторговле                 |
//|                   - Проверка на количество ордеров Time to повторном |
//|                     открытии сделки после неудачной попытки      |
//|                   - Добавлен фильтр новостей                     |
//|   Версия 12.37    - Возможность изменения цветов VisualDebug     |
//|                   - Проверка на ошибки Time to получении данных      |
//|                     с других таймфреймов                         |
//|                   - Настраиваемый таймфрейм для CCI              |
//|   Версия 12.37.1  - Исправлена ошибка вычисления расстояния между|
//|                     открытыми ордерами                           |
//|                   - Фикс проверки на ошибки Time to получении данных |
//|                     с других таймфреймов                         |
//|   Версия 12.37.2  - Вместо общей Time toбыли нескольких ордеров -    |
//|                     теперь считается средняя Time toбыль             |
//|                   - Исправлена ошибка с неодновременным закрытием|
//|                     нескольких ордеров                           |
//|                   - Добавлен параметр MDRFromHiLo, включающий    |
//|                     проверку диапазона с максимума/минимума дня  |
//|                   - Добавлен параметр сдвига времени             |
//|                   - MaxDailyRange теперь рассчитывается с учётом |
//|                     сдвига времени                               |
//|   Версия 12.38    - Исправлено отключение фильтра                |
//|                     Time to Exit_Distance = 0                        |
//|                   - Исправлен расчёт MaxDailyRange после полуночи|
//|                   - Time to включенном параметре CheckMDRAfter0      |
//|                     диапазон с предыдущего дня не проверяется    |
//|                     в понедельник                                |
//|                   - Исправлено отображение времени сессий, время |
//|                     которых Time toходилось на время закрытия рынка  |
//|                   - Параметр TimeShift заменён на GMT_Offset     |
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
//--- параметры для записи данных в файл
         string               InpFileName;                                    // Имя файла
         string               fileName                   = "EA_Generic";      // Имя файла
         string               InpDirectoryName           = "Generic LOGS";    // Имя каталога

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
//---- переменные для вычисления одновремменно открытых позиций

         #define STR_SHABLON "<!--STR-->"

         #define MAX_CURRENCY 20  // Максимальное количество валют (не пар)

         string Currencies = "AUD, EUR, USD, CHF, JPY, NZD, GBP, CAD, SGD, NOK, SEK, DKK, ZAR, MXN, HKD, HUF, CZK, PLN, RUR, TRY";

         string Shablon = "<!--STR-->, <!--STR-->";  // Шаблон для выдирания валют из входной строки Currencies

         int AmountCurrency,lotdigit;  // Общее количество учитываемых валют
         string Currency[MAX_CURRENCY]; // Учитываемые валюты
         double Volumes[MAX_CURRENCY];
         datetime LastUpdateTime;
 //---- переменные для вычисления одновремменно открытых позиций

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
   
   day_of_year_trade=0; // Time to переинициализации советника некоторые переменные не обнуляются, а сохраняют свои последние значения
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
   if (TotalOrders > 10) { Print("Количество ордеров установлено больше 10. Максимальное количество ордеров: 10"); TotalOrders=10; }

   TimeShift = GMT_Offset-2;
   if (!DST) TimeShift -= fGetDSTShift();


   if (Digits == 3 || Digits == 5) { //проверка на 4х, 5-и знаковый счет
      Slippage *= 10; Max_Spread *= 10; Take_Profit *= 10; Stop_Loss *= 10; min_TP *= 10; Entry_Break *= 10; 
      Min_Volatility *= 10; maxcandle *= 10; sizeloss *= 10; Time_Profit_Pips *= 10; Exit_Distance *= 10; 
      Exit_Profit_Pips *= 10; Reverse_Profit *= 10; CCI_Profit_Pips *= 10; Trail_Start *= 10; Trail_Size *= 10; Trail_Step *= 10; X *= 10; Y *= 10;
      PipsDivided = 10; MaxDailyRange *= 10; Max_Spread_On_Close *= 10; OrdersDistance *= 10;
   }   
 
   /*if (sizeloss < 0) {
      Print("Размер убыка для паузы перед открытием следующего ордера, должен быть указан в положительных числах!");
      Print("Текущее значение: ",sizeloss/PipsDivided," Time toведено к значению: ", MathAbs(sizeloss/PipsDivided));
      sizeloss = MathAbs(sizeloss);
   }  */
      
   //if (!_RealTrade && MarketInfo(NULL,MODE_SPREAD) > Max_Spread && Max_Spread > 0) { Print("Error: Current Spread (" + DoubleToStr(MarketInfo(NULL,MODE_SPREAD)/PipsDivided,1) +  ") > MaxSpread (" + DoubleToStr(Max_Spread/PipsDivided,1) + ")"); return(INIT_FAILED); }
    
   Take_Profit = MathMax(Take_Profit,NormalizeDouble(stoplevel,1));
   Stop_Loss = MathMax(Stop_Loss,NormalizeDouble(stoplevel,1));
   
   /*if (Auto_Risk > 0) */lots = AutoMM_Count(); //расчет торгового лота
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
   		string q = "Для работы новостного фильтра необходимо в настройках разрешить использование DLL. Новостной фильтр Offючен.";
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
      if ((UseNewsFilter) && (lastnewsupdate != iTimeTF)) { // для обновления в инфопанели
         _IsNews=IsNews(TimeBeforeNews,TimeAfterNews); 
         lastnewsupdate = iTimeTF;
         }
      }

   UpdateInfoPanel();
   
   if (_TimeM1 == iTimeM1 && !every_tick ) return ;
   _TimeM1 =  iTimeM1;

   if(!IsTime() && CountOrder(-1)<1) return;

   if(need_to_verify_channel != iTimeTF){ //обновлять данные индикатора раз в период
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
      
      if(TP_perc > 0) Take_Profit = MathMax(NormalizeDouble(channel_width/_Point /100 * TP_perc,1), min_TP); //проверка на динамический ТП
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
         if (_RealTrade) if (IsError("iCCI open",cci_signal_open,false)) skip_tick=true; // iCCI может вернуть 0 и без ошибки
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
				string q = __FUNCTION__ + ": не удалось выделить ордер! " + fMyErDesc();
				Print(q);
				fWriteDataToFile(q);
			}   
      }
      else if(OrderType() <= OP_SELL && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber) {

         if(OrderType() == OP_BUY) {
            count_buy++;
            if(OrderStopLoss() == 0.0){ //модифицирование ордера
               stoploss = NormalizeDouble(OrderOpenPrice() - Stop_Loss*_Point, Digits);
               takeprofit = NormalizeDouble(OrderOpenPrice() + Take_Profit*_Point, Digits);
               
               fModifyPosition(OrderTicket(), OrderOpenPrice(), stoploss, takeprofit, 0, clrGreen);
               continue;
            }
            
            if(X > 0 && Y > 0 && X > Y && Bid - OrderOpenPrice() >= X*_Point && OrderStopLoss() < OrderOpenPrice()){ //безубыток
               stoploss = NormalizeDouble(OrderOpenPrice() + Y*_Point, Digits);
               fModifyPosition(OrderTicket(), OrderOpenPrice(), stoploss, OrderTakeProfit(), 0, clrGreen);
            }
            
            Modify_and_exit_condition(OrderType(),OrderLots(),OrderOpenPrice(),OrderOpenTime(),OrderTicket(),OrderStopLoss()); //модификация ордеров и проверка условий на выход
         }
         
         else if(OrderType() == OP_SELL){
            count_sell++;
            if (OrderStopLoss() == 0.0) { //модификация ордера
               stoploss = NormalizeDouble(OrderOpenPrice() + Stop_Loss*_Point, Digits);
               takeprofit = NormalizeDouble(OrderOpenPrice() - Take_Profit*_Point, Digits);
               
               fModifyPosition(OrderTicket(), OrderOpenPrice(), stoploss, takeprofit, 0, clrGreen);
               continue;
            }
            
            if(X > 0 && Y > 0 && X > Y && OrderOpenPrice() - Ask >= X*_Point && OrderStopLoss() > OrderOpenPrice()){ //БУ
               stoploss = NormalizeDouble(OrderOpenPrice() - Y*_Point, Digits);
               fModifyPosition(OrderTicket(), OrderOpenPrice(), stoploss, OrderTakeProfit(), 0, clrGreen);
            }
            
            Modify_and_exit_condition(OrderType(),OrderLots(),OrderOpenPrice(),OrderOpenTime(),OrderTicket(),OrderStopLoss()); //модификация ордеров и проверка условий на выход
         }
      }     
   }       
   
   if(!IsTime() || (!OpenOrdersInRollover && fGetRollOver())) return;

   double h_price;
   bool is_order_dist;
   bool openresult;
   
   h_price=GetMaxMinOpenPrice(OP_BUY,MIN); // минимальная цена открытых ордеров
   if (h_price != 0) is_order_dist = Ask < h_price-OrdersDistance*_Point; else is_order_dist = true;  // расстояние от открытых ордеров не меньше заданного

   count_buy=CountOrder(OP_BUY);
   count_sell=CountOrder(OP_SELL);
   
  // Открытие сделок, если соблюдены все условия
   if (count_buy < 1 &&  //если нет открытых покупок
      (OnlyBid ? Bid : Ask) < channel_lower - Entry_Break*_Point) {  //если произошло касание канала BB
        openresult=OpenTradeConditions("BUY",OP_BUY,time_open_buy,count_buy,count_sell);
   }
   else if(count_buy >= 1 && count_buy < TotalOrders && is_order_dist && _TimeCurrent-GetLastOpenTime(OP_BUY)>MinPause) { 
        openresult=OpenTradeConditions("BUY",OP_BUY,time_open_buy,count_buy,count_sell);
   } 

   h_price=GetMaxMinOpenPrice(OP_SELL,MAX); // максимальная цена открытых ордеров
   if (h_price != 0) is_order_dist = Bid > h_price+OrdersDistance*_Point; else is_order_dist = true; // расстояние от открытых ордеров не меньше заданного
   
   count_buy=CountOrder(OP_BUY);
   count_sell=CountOrder(OP_SELL);

   if (count_sell < 1 && //если нет открытых продаж
      Bid > channel_upper + Entry_Break*_Point) {  //если произошло касание канала BB
        openresult=OpenTradeConditions("SELL",OP_SELL,time_open_sell,count_sell,count_buy);
   }         
   else if(count_sell >= 1 && count_sell < TotalOrders && is_order_dist && _TimeCurrent-GetLastOpenTime(OP_SELL)>MinPause) {  
        openresult=OpenTradeConditions("SELL",OP_SELL,time_open_sell,count_sell,count_buy);
   }   
   
   int Error = GetLastError(); //поиск ошибок в завершение
   if(Error != 0) Print("OnTick() Error ",Error,": ",ErrorDescription(Error));
}

void OnTimer()
{
timer_active=true;
OnTick();
}

//+------------------------------------------------------------------+
bool Modify_and_exit_condition(int _OrderType, double _OrderLots, double _OrderOpenPrice, datetime _OrderOpenTime,int _OrderTicket, double _OrderStopLoss) {
               
   if(Trail_Start > 0 && Trail_Size > 0) { //Тралл открытой позиции
      if (!rollover_trall_end || (rollover_trall_end && !fGetRollOver())) fTrailingStopFunc();
   }

   string logstr="";
   string warnstr="";
   string filterstr="";
   int filterno=0;

   bool condition_for_the_exit = false;
   bool channel_condition = false;
   
   double PriceType = Ask; //если ордер селл.
   double OrderDistance = _OrderOpenPrice - PriceType;
   double orderdist;
   string orderstr="";
   string ordertotalstr="";
   
   if (_OrderType == OP_BUY) { //если ордер бай
      PriceType = Bid;
      OrderDistance = PriceType - _OrderOpenPrice;
   }
  
   if (TotalOrders>1 && CloseSimultaneously) {
      orderdist=OrderDistance;
      orderstr="; Time toбыль ордера - "+DoubleToStr(orderdist/_Point/PipsDivided,1)+" пунктов";
      ordertotalstr=" средняя";
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
      
      
   if((Exit_Minutes > 0 && _TimeCurrent - _OrderOpenTime > 60 * Exit_Minutes && // ордер открыт более Exit_Minutes и
      OrderDistance > Time_Profit_Pips*_Point) || (CloseReason_Multi == 1)) {                                 // плавающая Time toбыль более Time_Profit_Pips (0)
      condition_for_the_exit = true;
      filterno=1;
	   if (LogMode < 2) {
		   logstr = "Закрытие ордера #" + IntegerToString(_OrderTicket) + " по цене "+ DoubleToStr(PriceType,Digits) + ". Время существования более " + IntegerToString(Exit_Minutes) + 
		         " минут и"+ordertotalstr+" плавающая Time toбыль составляет более " + DoubleToStr(Time_Profit_Pips/PipsDivided,1) + " пунктов: " + DoubleToStr(OrderDistance/_Point/PipsDivided,1)+" пунктов" + orderstr + "; Спред = " + DoubleToStr((Ask - Bid) / _Point/PipsDivided, 1)+ " пунктов.";
		   warnstr="1. Time";
		   filterstr="1. Time";
	   }
   }
   if((Exit_Distance/PipsDivided < 100 && channel_condition &&    // цена вышла за границу канала и
      OrderDistance > Exit_Profit_Pips*_Point) || (CloseReason_Multi == 2)) {           // плавающая Time toбыль более Exit_Profit_Pips (-12)
      condition_for_the_exit = true;
      filterno=2;
	   if (LogMode < 2) {
		   logstr = "Закрытие ордера #" + IntegerToString(_OrderTicket) + " по цене "+ DoubleToStr(PriceType,Digits) + ". Цена пересекла границу канала " + DoubleToStr(channel_lower,Digits) + " на " + 
			      DoubleToStr(Exit_Distance/PipsDivided,1) + " пунктов и"+ordertotalstr+" плавающая Time toбыль составила более " + DoubleToStr(Exit_Profit_Pips/PipsDivided,1) + " пунктов: " + DoubleToStr(OrderDistance/_Point/PipsDivided,1)+" пунктов" + orderstr + "; Спред = " + DoubleToStr((Ask - Bid) / _Point/PipsDivided, 1)+ " пунктов.";
		   warnstr="2. BB Channel";
		   filterstr="2. BB Channel";
      }
   }

   if(MA_period > 0 && _OrderOpenTime < iTime(NULL,TimeFrameMA,0) ) { //фильтра по МА
   
      int MA_Type_Exit = -1; //обнуление
      if (ma_shift[1] > ma_shift[2] && ma_shift[2] <= ma_shift[3] && ma_shift[3] <= ma_shift[4]) MA_Type_Exit = OP_SELL;
      else if (ma_shift[1] < ma_shift[2] && ma_shift[2] >= ma_shift[3] && ma_shift[3] >= ma_shift[4]) MA_Type_Exit = OP_BUY;
      
      if((((_OrderType == OP_SELL && MA_Type_Exit == OP_SELL) || (_OrderType == OP_BUY && MA_Type_Exit == OP_BUY)) && //скользящая средняя повышается или понижается
         OrderDistance > Reverse_Profit*_Point) || (CloseReason_Multi == 3)) { // и плавающая Time toбыль более Reverse_Profit (20)
         condition_for_the_exit = true;
         filterno=3;
		   if (LogMode < 2) {
			   logstr = "Закрытие ордера #" + IntegerToString(OrderTicket()) + " по цене "+ DoubleToStr(PriceType,Digits) + ". Скользящая средняя изменяется" +
				      " и"+ordertotalstr+" плавающая Time toбыль более " + DoubleToStr(Reverse_Profit/PipsDivided,1) + " пунктов: " + DoubleToStr(OrderDistance/_Point/PipsDivided,1)+" пунктов" + orderstr + "; Спред = " + DoubleToStr((Ask - Bid) / _Point/PipsDivided, 1)+ " пунктов.";
			   warnstr="3. MA";
			   filterstr="3. MA";
		   }
      }
   }

   if((cci_Period_close > 0 && OrderDistance > CCI_Profit_Pips*_Point && cci_signal_close != _OrderType &&
      cci_signal_close != -1) || (CloseReason_Multi == 4)) { //фильтр по CCI
      condition_for_the_exit = true;
      filterno=4;
	   if (LogMode < 2) {
		   logstr = "Закрытие ордера #" + IntegerToString(_OrderTicket) + " по цене "+ DoubleToStr(PriceType,Digits) + ". Выход по индикатору CCI" + "; Спред = " + DoubleToStr((Ask - Bid) / _Point/PipsDivided, 1)+ " пунктов.";
		   warnstr="4. CCI";
		   filterstr="4. CCI";
	   }
   }
   
   if (UseNewsFilter && CloseTimeBeforeNews != 0 && IsNews(CloseTimeBeforeNews,TimeAfterNews)) { //фильтр новостей
      condition_for_the_exit = true;
      if (LogMode < 2) {
		   logstr = "Закрытие ордера #" + IntegerToString(_OrderTicket) + " по цене "+ DoubleToStr(PriceType,Digits) + ". Фильтр новостей" + "; Спред = " + DoubleToStr((Ask - Bid) / _Point/PipsDivided, 1)+ " пунктов.";
		   warnstr="5. News";
		   filterstr="5. News";
      }   
   }   

   if(CloseReason_Multi == 0 && condition_for_the_exit) {
      if ((!CloseOrdersInRollover) && fGetRollOver()) {
         if (LogMode < 2 && _TimeCurrent-lasttimewrite > 60) {
            logstr = "Попытка закрытия ордера #" + IntegerToString(OrderTicket()) + " (фильтр "+filterstr+"): Фильтр ролловера. Ордер не был закрыт.";
            warnstr += " + rollover";
            DrawWarn(warnstr, WarnExitColor);
            Print(logstr);
            fWriteDataToFile(logstr);
            lasttimewrite = _TimeCurrent;
            }   
         return(false);
      }
      if(Max_Spread_On_Close>0){
         if ((Ask - Bid) > Max_Spread_On_Close*_Point && OrderType()==OP_SELL) { //фильтр по максимальному спреду
            if (LogMode < 2 && _TimeCurrent-lasttimewrite > 60) {
               logstr = "Попытка закрытия ордера #" + IntegerToString(OrderTicket()) + " (фильтр "+filterstr+"): Спред (" + DoubleToStr((Ask - Bid) / _Point/PipsDivided, 1) + " пунктов) больше максимального (" + DoubleToStr(Max_Spread_On_Close/PipsDivided, 1) + ")! Ордер не был закрыт.";
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
      
   if(condition_for_the_exit) { //если сработал хоть один фильтр:
   
      if (TotalOrders>1 && CloseSimultaneously) CloseReason_Multi=filterno;

      if (warnstr != "") DrawWarn(warnstr, WarnExitColor);
      if (logstr != "") {
         Print(logstr);
         fWriteDataToFile(logstr);
         }
 
      RefreshRates();
      if(!OrderClose(_OrderTicket, _OrderLots, NormalizeDouble(PriceType, Digits), Slippage, clrViolet)){ //закрытие ордера
			if (LogMode < 3) {
			   string q = __FUNCTION__ + ": Ордер " + Symbol() + " #" + IntegerToString(OrderTicket()) + " не был закрыт! " + fMyErDesc();
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

   if(/*order_cnt == 0 && */cci_Period_open > 0 && cci_signal_open != OP_TYPE) { //фильтр по CCI // проверять только Time to открытии первого ордера
      if (LogMode < 2 && lastbarwrite != iTimeTF) {
         string q = Symbol() + "| CCI не вышел за уровень: ±" + IntegerToString(cci_level_open) + " Ордер " + _OrderType + " не был открыт.";
         Print(q); fWriteDataToFile(q);
         DrawWarn("filter CCI",WarnEnterColor);
	      lastbarwrite = iTimeTF;
      }   
      return(false);
   }
   if (channel_width != 0 && channel_width < Min_Volatility*_Point) { //фильтр по ширине канале
      if (LogMode < 2 && lastbarwrite != iTimeTF) {
         string q = Symbol() + "| Текущая ширина канала: (" + DoubleToStr(channel_width/_Point/PipsDivided, 1) + " пунктов) меньше, чем минимальная ширина канала: " + DoubleToStr(Min_Volatility/PipsDivided, 1) + " Ордер " + _OrderType + " не был открыт.";
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
      //if ((candle() > maxcandle*_Point)){ // Фильтр по размеру предыдущих свечей
         if (LogMode < 2 && lastbarwrite != iTimeTF) {
            //string q = Symbol() + "| Найдена свеча за предыдущих " + IntegerToString(barcount) + " баров, больше " + DoubleToStr(maxcandle/PipsDivided, 1) + " пунктов. Ордер " + _OrderType + " не был открыт.";
            string q = Symbol() + "| Диапазон движения цены за предыдущих " + IntegerToString(barcount) + " баров "+_period+" (" + DoubleToStr(candlerange/_Point/PipsDivided, 1) + " пунктов) больше " + DoubleToStr(maxcandle/PipsDivided, 1) + " пунктов. Ордер " + _OrderType + " не был открыт.";
            Print(q); fWriteDataToFile(q);
            DrawWarn("filter maxcandle",WarnEnterColor);
	         lastbarwrite = iTimeTF;
         }   
         return(false);
      }   
   }
   if (!Hedging && opposite_order >= 1) { //фильтр запрета на хеджирования
      if (LogMode < 2 && _TimeCurrent-lasttimewrite > 120) {
         string q = Symbol() + "| Фильтр хеджирующих позиций включен и найдена противоположная сделка. Ордер " + _OrderType + " не был открыт.";
         Print(q); fWriteDataToFile(q);
         DrawWarn("filter hedge",WarnEnterColor);
	      lasttimewrite = _TimeCurrent;
      }   
      return(false);
   }   
   if(MaxDailyRange > 0) { //фильтр MaxDailyRange
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
         //if (barshift==-1) timeopend1 -= 24*60*60; // если попадает на выходные
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
            string q = Symbol() + "| Дневной диапазон движения цены: (" + DoubleToStr(mdr/_Point/PipsDivided, 1) + " пунктов) ,больше MaxDailyRange: " + DoubleToString(MaxDailyRange/PipsDivided,1) + " Ордер " + _OrderType + " не был открыт.";
            Print(q);
            DrawWarn("filter MDR",WarnEnterColor);
	         lastbarwrite = iTimeTF;
         }   
         return(false);
      }
   }
   if (UseNewsFilter && IsNews(TimeBeforeNews,TimeAfterNews)) { //фильтр новостей
      if (LogMode < 2 && lastbarwrite != iTimeTF) {
         string q = Symbol() + "| Фильтр новостей. Ордер " + _OrderType + " не был открыт.";
         Print(q); fWriteDataToFile(q);
         DrawWarn("filter news",WarnEnterColor);
   	   lastbarwrite = iTimeTF;
      }   
      return(false);
   }

   if (_RealTrade && MaxAmountCurrencyMult>0) { //фильтр по максимально разрешенному обьему
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
            string q = Symbol() + "| Максимально разрешенный обьем " + Str + DoubleToString(MaxLot,2) + " будет превышен. " + " Ордер " + _OrderType + " не был открыт.";
            Print(q);
            DrawWarn("filter MaxAmount",WarnEnterColor);
   	      lastbarwrite = Time[0];
         }   
         return(false);      
      }
   }
   if (Max_Spread > 0 && (Ask - Bid) > Max_Spread*_Point && (CheckSpreadOnSellOpen || OP_TYPE==OP_BUY)) { //фильтр по максимальному спреду
      if (LogMode < 2 && _TimeCurrent-lasttimewrite > 60) {
         string q = Symbol() + "| Спред (" + DoubleToStr((Ask - Bid) / _Point/PipsDivided, 1) + " пунктов) больше максимального (" + DoubleToStr(Max_Spread/PipsDivided, 1) + ")! Ордер " + _OrderType + " не был открыт.";
         Print(q); fWriteDataToFile(q);
         DrawWarn("filter spread",WarnEnterColor);
	      lasttimewrite = _TimeCurrent;
      }   
      return(false);
   }   
  
  
  
   if (pause > 0 && lastloss() != 0) { //если последний ордер не закрыт в профит (или убыток больше заданного)
      return(false);
      }
         
   if (!IsTradeAllowed()) {
      if (LogMode < 3 && lastbarwrite1 != iTimeTF) {
         string q = Symbol() + "| Торговля не разрешена. Необходимо включить опцию <Разрешить " +
                              "советнику торговать> в свойствах эксперта." + " Ордер " + _OrderType + " не был открыт.";
         Print(q); fWriteDataToFile(q);
         DrawWarn("trade is not allowed",WarnEnterColor);
	      lastbarwrite1 = iTimeTF;
      }   
      return(false);
      }

   bool res=false;
   
      res=OpenTrade(_OrderType);  //открываем торговый ордер
      if (res) time_open = _TimeCurrent; 
      
   return(res); 
}
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+      
bool OpenTrade(string type){ //открытие ордеров
   double price = 0;
   int cmd = -1;
   int err = 0;
   color col_type = clrNONE;
   
   /*if (Auto_Risk > 0) */lots = AutoMM_Count(); //расчет торгового лота
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
      else { // если на предыдущей попытке была возвращена ошибка, но ордер был открыт
         ticket=GetLastOrderTicket(cmd); // возвращаем тикет последнего открытого ордера
         }   
      if(ticket > 0){
         lastticket = ticket;
		   if (LogMode < 3) {
			   if(!OrderSelect(ticket, SELECT_BY_TICKET, MODE_TRADES)) {
   				string q = __FUNCTION__ + ": не удалось выделить ордер " +
   						 IntegerToString(ticket) + "! " + fMyErDesc();
   				Print(q);
   				fWriteDataToFile(q);
   				//break;
		      }
         }
		   if (LogMode < 2) {
				string q = __FUNCTION__ + ": ордер " + type + " открыт по цене " + DoubleToStr(OrderOpenPrice(),Digits) + "; Ширина канала  = " + DoubleToStr(channel_width/_Point/PipsDivided,1) + "; DailyRange  = " + DoubleToStr(range/_Point/PipsDivided,1) + "; Спред = " + DoubleToStr((Ask - Bid) / _Point/PipsDivided, 1)+ " пунктов.";
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
   		   string q = __FUNCTION__ + ": Ошибка открытия ордера " + type + ": " + fMyErDesc(err);
   		   Print(q);
   		   fWriteDataToFile(q);
		   }
         Sleep(3000);
      }
   }
   if (LogMode < 3) {
	   string q = __FUNCTION__ + ": ордер " + type + " не был открыт!: " + fMyErDesc(err);
	   Print(q);
	   fWriteDataToFile(q);
      }
   Sleep(3000);
   return(false);
}

//+------------------------------------------------------------------+          
//+------------------------------------------------------------------+
double AutoMM_Count() { //Расчет лота Time to указании процента риска в настройках.
   double lot=Lots;int loss = TotalLoss();
   
   if (Auto_Risk > 0.0) {
      if (MM_Depo == 0) {
         double TickValue = (MarketInfo(Symbol(), MODE_TICKVALUE) == 0 ? 1 : MarketInfo(Symbol(), MODE_TICKVALUE));
         double Balance = (AccountEquity() > AccountBalance() ? AccountBalance() : AccountEquity());
         lot = ((Balance - AccountCredit()) * (Auto_Risk / 100)) / Stop_Loss / TickValue; 
         //lot = MathFloor(lot/MarketInfo(Symbol(), MODE_LOTSTEP))* MarketInfo(Symbol(), MODE_LOTSTEP); //округление полученного лота вниз
         //lot = MathMin(MathMax(lot, MarketInfo(Symbol(), MODE_MINLOT)), MarketInfo(Symbol(), MODE_MAXLOT)); //сравнение полученнго лота с минимальным/максимальным.
         //Print(lot + " " + StopLoss + " / " + MarketInfo(Symbol(), MODE_TICKSIZE)/0.00001);
         //return (lot);
         }
      else 
         lot = NormalizeDouble(Lots * MathFloor(AccountBalance()/MM_Depo), 2);
         //lot = MathMin(MathMax(lot, MarketInfo(Symbol(), MODE_MINLOT)), MarketInfo(Symbol(), MODE_MAXLOT)); //сравнение полученнго лота с минимальным/максимальным.
      }
      
   if(Martingale && loss <= 100){ lot = NormalizeDouble(lot * MathPow(Multiplier, loss), lotdigit);}
      
   if (MarketInfo(Symbol(), MODE_LOTSTEP) != 0) lot = MathFloor(lot/MarketInfo(Symbol(), MODE_LOTSTEP))* MarketInfo(Symbol(), MODE_LOTSTEP); //округление полученного лота вниз
   lot = MathMin(MathMax(lot, MarketInfo(Symbol(), MODE_MINLOT)), MarketInfo(Symbol(), MODE_MAXLOT)); //сравнение полученнго лота с минимальным/максимальным.
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
   	   if (LogMode < 2) {Print(__FUNCTION__ + ": не обнаружены ранее закрытые ордера, функция пропущена");}
   	}
   } 
   else {    
   	if (OrdersHistoryTotal() != lasthistorytotal) {
   		lasthistorytotal = OrdersHistoryTotal();
   		for(int i=lasthistorytotal;i>=1;i--) {
   	      if(OrderSelect(i-1,SELECT_BY_POS,MODE_HISTORY)) {
   	         if (TimeCurrent() - OrderCloseTime() > 60*60*24*14) break; //если ордер старше 2х недель - прекращаем поиск.
   				if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber) {
   				   ticket = OrderTicket();
                  if (ticket > oldticket) oldticket = ticket;			   
   	         }                 
   	      }
            else {
      	      if (LogMode < 3) {
   			      string q = (__FUNCTION__ + ": ордер не выбран. Ошибка выбора ордера.");
                  Print(q);
                  fWriteDataToFile(q);
               }
            }		 
         }
   	} 
   	else {
   		if (lastticket > 0) oldticket = lastticket; // "Советник еще не открывал ни одного ордера" - неправильно. Последний ордер мог открыться с ошибкой, а предпоследний - без ошибки.
   		else if (LogMode < 2) {/*Print("Советник еще не открывал ни одного ордера, функция "+__FUNCTION__ +" пропущена.");*/ return(0);}
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
   			   string q = (__FUNCTION__ + ": информация о последнем ордере не совпадает с параметрами советника.");
               Print(q);
               fWriteDataToFile(q);   			   
   			}
   		}
      }
   } 
   else {
   	if (OrdersHistoryTotal() > 0 && LogMode < 2) {Print(__FUNCTION__ + ": тикет предыдущего ордера не найден. Возможно, советник ранее не открывал ордера.");}
   }
   
   if (otype == OP_BUY && priceopen-priceclose > sizeloss*_Point && TimeCurrent()-closetime < pause*60) signal = 1;
   
   if (otype == OP_SELL && priceclose-priceopen > sizeloss*_Point && TimeCurrent()-closetime < pause*60) signal = 2;        
   
   if (signal > 0) {
      if (LogMode<2 && lastbarwrite != Time[0]) {

         string q1 = __FUNCTION__ + ": Сработал фильтр по убытку последнего закрытого ордера. Новый ордер не был открыт.";
         string q2 = __FUNCTION__ + ": До возможности открытия нового ордера осталось " + DoubleToStr(pause-(TimeCurrent()-closetime)/60,0)+" минут";
 
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
      if(OrderStopLoss() < OrderOpenPrice() && Bid - OrderOpenPrice() >= tr_start){ //если SL еще не двигали
         fModifyPosition(OrderTicket(),OrderOpenPrice(),NormalizeDouble(Bid - tr_start,Digits),OrderTakeProfit(),0,clrGreen);
         return;
      }
      if(OrderStopLoss() >= OrderOpenPrice()){ //если SL уже двигали
         double dif = Bid - OrderStopLoss() - tr_size;
         if(dif >= tr_step)
            fModifyPosition(OrderTicket(),OrderOpenPrice(),NormalizeDouble(OrderStopLoss() + dif,Digits),
                              OrderTakeProfit(),0,clrGreen);
         return;
      }
   }
   else if(OrderType() == OP_SELL){
      if(OrderStopLoss() > OrderOpenPrice() && OrderOpenPrice() - Ask >= tr_start){ //если SL еще не двигали
         fModifyPosition(OrderTicket(),OrderOpenPrice(),NormalizeDouble(Ask + tr_start,Digits),OrderTakeProfit(),0,clrTomato);
         return;
      }
      if(OrderStopLoss() <= OrderOpenPrice()){ //если SL уже двигали
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
		  string q = __FUNCTION__ + ": не удалось модифицировать ордер #" +
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
//    возвращает ИСТИНА, если сейчас время ролловера                 +
//    иначе - ЛОЖЬ                                                   +
//+------------------------------------------------------------------+
bool fGetRollOver(void){
   //if(use_rollover_filter) { //Не открывать сделки в ролловер.
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
//|          Записывает строку 's' в файл InpFileName                |
//+------------------------------------------------------------------+
void fWriteDataToFile(string s){

   if (!WriteLogFile) return;
   if(IsTesting() || IsOptimization()) return;
   //--- откроем файл для записи данных (если его нет, то создастся автоматически)
   ResetLastError();
   string data = "";
   data = IntegerToString(Day()) + "." + IntegerToString(Month()) + "." + IntegerToString(Year()) + "  " +
         IntegerToString(Hour()) + ":" + IntegerToString(Minute());
   int file_handle = FileOpen(InpDirectoryName+"//"+InpFileName,FILE_TXT|FILE_READ|FILE_WRITE);
   if(file_handle != INVALID_HANDLE){
      PrintFormat("Файл %s открыт для записи",InpFileName);
      PrintFormat("Путь к файлу: %s\\Files\\",TerminalInfoString(TERMINAL_DATA_PATH));
      
      FileSeek(file_handle,0,SEEK_END); //переставляем курсор в конец файла
      
      s = data + "   " + s;
      
      //--- запишем значения в файл
      FileWrite(file_handle,s);
      //--- закрываем файл
      FileClose(file_handle);
      PrintFormat("Данные записаны, файл %s закрыт",InpFileName);
     }
   else PrintFormat("Не удалось открыть файл %s, " + fMyErDesc(),InpFileName);
}
//+------------------------------------------------------------------+ 
//| Создает прямоугольную метку                                      | 
//+------------------------------------------------------------------+ 
bool fRectLabelCreate(const long             chart_ID    = 0,                 // ID графика 
                      const string           name        = "RectLabel",       // имя метки 
                      const int              sub_window  = 0,                 // номер подокна 
                      const int              x           = 0,                 // координата по оси X 
                      const int              y           = 0,                 // координата по оси Y 
                      const int              width       = 50,                // ширина 
                      const int              height      = 18,                // высота 
                      const color            back_clr    = C'236,233,216',    // цвет фона 
                      const ENUM_BORDER_TYPE border      = BORDER_SUNKEN,     // тип границы 
                      const ENUM_BASE_CORNER corner      = CORNER_LEFT_UPPER, // угол графика для Time toвязки 
                      const color            clr         = clrRed,            // цвет плоской границы (Flat) 
                      const ENUM_LINE_STYLE  style       = STYLE_SOLID,       // стиль плоской границы 
                      const int              line_width  = 1,                 // толщина плоской границы 
                      const bool             back        = false,             // на заднем плане 
                      const bool             selection   = false,             // выделить для перемещений 
                      const bool             hidden      = true,              // скрыт в списке объектов 
                      const long             z_order     = 0)                 // Time toоритет на нажатие мышью
{ 
//--- сбросим значение ошибки 
   ResetLastError(); 
   if(ObjectFind(chart_ID,name)==0) return true;
//--- создадим прямоугольную метку 
   if(!ObjectCreate(chart_ID,name,OBJ_RECTANGLE_LABEL,sub_window,0,0)) 
     { 
	  if (LogMode<3) {
		  Print(__FUNCTION__, 
				": не удалось создать прямоугольную метку! " + fMyErDesc()); 
      } 
	  return(false); 
     } 
//--- установим координаты метки 
   ObjectSetInteger(chart_ID,name,OBJPROP_XDISTANCE,x); 
   ObjectSetInteger(chart_ID,name,OBJPROP_YDISTANCE,y); 
//--- установим размеры метки 
   ObjectSetInteger(chart_ID,name,OBJPROP_XSIZE,width); 
   ObjectSetInteger(chart_ID,name,OBJPROP_YSIZE,height); 
//--- установим цвет фона 
   ObjectSetInteger(chart_ID,name,OBJPROP_BGCOLOR,back_clr); 
//--- установим тип границы 
   ObjectSetInteger(chart_ID,name,OBJPROP_BORDER_TYPE,border); 
//--- установим угол графика, относительно которого будут определяться координаты точки 
   ObjectSetInteger(chart_ID,name,OBJPROP_CORNER,corner); 
//--- установим цвет плоской рамки (в режиме Flat) 
   ObjectSetInteger(chart_ID,name,OBJPROP_COLOR,clr); 
//--- установим стиль линии плоской рамки 
   ObjectSetInteger(chart_ID,name,OBJPROP_STYLE,style); 
//--- установим толщину плоской границы 
   ObjectSetInteger(chart_ID,name,OBJPROP_WIDTH,line_width); 
//--- отобразим на переднем (false) или заднем (true) плане 
   ObjectSetInteger(chart_ID,name,OBJPROP_BACK,back); 
//--- включим (true) или отключим (false) режим перемещения метки мышью 
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTABLE,selection); 
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTED,selection); 
//--- скроем (true) или отобразим (false) имя графического объекта в списке объектов 
   ObjectSetInteger(chart_ID,name,OBJPROP_HIDDEN,hidden); 
//--- установим Time toоритет на получение события нажатия мыши на графике 
   ObjectSetInteger(chart_ID,name,OBJPROP_ZORDER,z_order); 
//--- успешное выполнение 
   return(true); 
}
//+------------------------------------------------------------------+ 
//| Удаляет прямоугольную метку                                      | 
//+------------------------------------------------------------------+ 
bool fRectLabelDelete(const long   chart_ID   = 0,           // ID графика 
                      const string name       = "RectLabel") // имя метки 
{ 
//--- сбросим значение ошибки 
   ResetLastError(); 
//--- удалим метку 
   if (ObjectFind(chart_ID,name) != -1) {
      if(!ObjectDelete(chart_ID,name)) 
        { 
   	  if (LogMode<3) {
   		  Print(__FUNCTION__, 
   				": не удалось удалить прямоугольную метку! " + fMyErDesc()); 
   	  }	
         return(false); 
        }
    }   
//--- успешное выполнение 
   return(true); 
}
//+------------------------------------------------------------------+
void DrawChannel(string dir, double pr1, color clr=clrYellow){ //рисование линий для входов по каналу BB

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
         if(X > 0 && Y > 0) be_info = "\n  Breakeven: ON"; //безубыток
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

   string pref="Ошибка №: "+IntegerToString(aErrNum)+" - ";
   switch(aErrNum){
      case 0:   return(pref+"Нет ошибки. Торговая операция прошла успешно.");
      case 1:   return(pref+"Нет ошибки, но результат неизвестен. (OrderModify пытается " +
                              "изменить уже установленные значения такими же значениями. " +
                              "Необходимо изменить одно или несколько значений и повторить попытку.)");
      case 2:   return(pref+"Общая ошибка. Прекратить все попытки торговых операций до " +
                              "выяснения обстоятельств. Возможно перезагрузить операционную " +
                              "систему и клиентский терминал.");
      case 3:   return(pref+"Неправильные параметры. В торговую функцию переданы неправильные " +
                              "параметры, наTime toмер, неправильный символ, неопознанная торговая " +
                              "операция, отрицательное допустимое отклонение цены, " +
                              "несуществующий номер тикета и т.п. Необходимо изменить логику программы.");
      case 4:   return(pref+"Торговый сервер занят. Можно повторить попытку через достаточно " +
                              "большой промежуток времени (от нескольких минут).");
      case 5:   return(pref+"Старая версия клиентского терминала. Необходимо установить " +
                              "последнюю версию клиентского терминала.");
      case 6:   return(pref+"Нет связи с торговым сервером.  Необходимо убедиться, что связь " +
                              "не нарушена (наTime toмер, Time to помощи функции IsConnected) и " +
                              "через небольшой промежуток времени (от 5 секунд) повторить попытку.");
      case 7:   return(pref+"Недостаточно прав.");
      case 8:   return(pref+"Слишком частые запросы. Необходимо уменьшить частоту запросов, " +
                              "изменить логику программы.");
      case 9:   return(pref+"Недопустимая операция, нарушающая функционирование сервера");
      case 64:  return(pref+"Счет заблокирован. Необходимо прекратить все попытки торговых операций.");
      case 65:  return(pref+"Неправильный номер счета. Необходимо прекратить все попытки " +
                              "торговых операций.");
      case 128: return(pref+"Истек срок ожидания совершения сделки. Прежде, чем производить " +
                              "повторную попытку (не менее, чем через 1 минуту), необходимо " +
                              "убедиться, что торговая операция действительно не прошла " +
                              "(новая позиция не была открыта, либо существующий ордер не " +
                              "был изменён или удалён, либо существующая позиция не была закрыта)");
      case 129: return(pref+"Неправильная цена bid или ask, возможно, ненормализованная " +
                              "цена. Необходимо после задержки от 5 секунд обновить " +
                              "данные Time to помощи функции RefreshRates и повторить попытку. " +
                              "Если ошибка не исчезает, необходимо прекратить все попытки " +
                              "торговых операций и изменить логику программы.");
      case 130: return(pref+"Неправильные стопы. Слишком близкие стопы или неправильно " +
                              "рассчитанные или ненормализованные цены в стопах (или в " +
                              "цене открытия отложенного ордера). Попытку можно повторять " +
                              "только в том случае, если ошибка произошла из-за устаревания " +
                              "цены. Необходимо после задержки от 5 секунд обновить данные " +
                              "Time to помощи функции RefreshRates и повторить попытку. Если " +
                              "ошибка не исчезает, необходимо прекратить все попытки " +
                              "торговых операций и изменить логику программы.");
      case 131: return(pref+"Неправильный объем, ошибка в грануляции объема. Необходимо " +
                              "прекратить все попытки торговых операций и изменить логику программы.");
      case 132: return(pref+"Рынок закрыт. Можно повторить попытку через достаточно большой " +
                              "промежуток времени (от нескольких минут).");
      case 133: return(pref+"Торговля запрещена. Необходимо прекратить все попытки торговых операций.");
      case 134: return(pref+"Недостаточно средств для совершения операции. Повторять сделку " +
                              "с теми же параметрами нельзя. Попытку можно повторить после " +
                              "задержки от 5 секунд, уменьшив объем, но надо быть уверенным в " +
                              "достаточности средств для совершения операции.");
      case 135: return(pref+"Цена изменилась. Можно без задержки обновить данные Time to помощи " +
                              "функции RefreshRates и повторить попытку.");
      case 136: return(pref+"Нет цен. Брокер по какой-то Time toчине (наTime toмер, в начале сессии " +
                              "цен нет, неподтвержденные цены, быстрый рынок) не дал цен или " +
                              "отказал. Необходимо после задержки от 5 секунд обновить данные " +
                              "Time to помощи функции RefreshRates и повторить попытку.");
      case 137: return(pref+"Брокер занят");
      case 138: return(pref+"Новые цены. Запрошенная цена устарела, либо перепутаны bid и " +
                              "ask. Можно без задержки обновить данные Time to помощи функции " +
                              "RefreshRates и повторить попытку. Если ошибка не исчезает, " +
                              "необходимо прекратить все попытки торговых операций и изменить " +
                              "логику программы.");
      case 139: return(pref+"Ордер заблокирован и уже обрабатывается. . Необходимо прекратить " +
                              "все попытки торговых операций и изменить логику программы.");
      case 140: return(pref+"Разрешена только покупка. Повторять операцию SELL нельзя.");
      case 141: return(pref+"Слишком много запросов. Необходимо уменьшить частоту " +
                              "запросов, изменить логику программы.");
      case 142: return(pref+"Ордер поставлен в очередь. Это не ошибка, а один из кодов " +
                              "взаимодействия между клиентским терминалом и торговым " +
                              "сервером. Этот код может быть получен в редком случае, " +
                              "когда во время выполнения торговой операции произошёл " +
                              "обрыв и последующее восстановление связи. Необходимо " +
                              "обрабатывать так же как и ошибку 128.");
      case 143: return(pref+"Ордер Time toнят дилером к исполнению. Один из кодов взаимодействия " +
                              "между клиентским терминалом и торговым сервером. Может " +
                              "возникнуть по той же Time toчине, что и код 142. Необходимо " +
                              "обрабатывать так же как и ошибку 128.");
      case 144: return(pref+"Ордер аннулирован самим клиентом Time to ручном подтверждении " +
                              "сделки. Один из кодов взаимодействия между клиентским " +
                              "терминалом и торговым сервером.");
      case 145: return(pref+"Модификация запрещена, так как ордер слишком близок к " +
                              "рынку и заблокирован из-за возможного скорого исполнения. " +
                              "Можно не ранее, чем через 15 секунд, обновить данные Time to " +
                              "помощи функции RefreshRates и повторить попытку.");
      case 146: return(pref+"Подсистема торговли занята. Повторить попытку только после " +
                              "того, как функция IsTradeContextBusy вернет FALSE.");
      case 147: return(pref+"Использование даты истечения ордера запрещено брокером. " +
                              "Операцию можно повторить только в том случае, если " +
                              "обнулить параметр expiration.");
      case 148: return(pref+"Количество открытых и отложенных ордеров достигло предела, " +
                              "установленного брокером. Новые открытые позиции и " +
                              "отложенные ордера возможны только после закрытия или " +
                              "удаления существующих позиций или ордеров.");
      case 149: return(pref+"Попытка открыть противоположную позицию к уже существующей " +
                              "в случае, если хеджирование запрещено. Сначала необходимо " +
                              "закрыть существующую противоположную позицию, либо отказаться " +
                              "от всех попыток таких торговых операций, либо изменить " +
                              "логику программы.");
      case 150: return(pref+"Попытка закрыть позицию по инструменту в противоречии с правилом FIFO");
      //---- Коды ошибок выполнения MQL4-программы (советника)
      case 4000: return(pref+"Нет ошибки");
      case 4001: return(pref+"Неправильный указатель функции");
      case 4002: return(pref+"Индекс массива - вне диапазона");
      case 4003: return(pref+"Нет памяти для стека функций");
      case 4004: return(pref+"Переполнение стека после рекурсивного вызова");
      case 4005: return(pref+"На стеке нет памяти для передачи параметров");
      case 4006: return(pref+"Нет памяти для строкового параметра");
      case 4007: return(pref+"Нет памяти для временной строки");
      case 4008: return(pref+"Неинициализированная строка");
      case 4009: return(pref+"Неинициализированная строка в массиве");
      case 4010: return(pref+"Нет памяти для строкового массива");
      case 4011: return(pref+"Слишком длинная строка");
      case 4012: return(pref+"Остаток от деления на ноль");
      case 4013: return(pref+"Деление на ноль");
      case 4014: return(pref+"Неизвестная команда");
      case 4015: return(pref+"Неправильный переход");
      case 4016: return(pref+"Неинициализированный массив");
      case 4017: return(pref+"Вызовы DLL не разрешены");
      case 4018: return(pref+"Невозможно загрузить библиотеку");
      case 4019: return(pref+"Невозможно вызвать функцию");
      case 4020: return(pref+"Вызовы внешних библиотечных функций не разрешены");
      case 4021: return(pref+"Недостаточно памяти для строки, возвращаемой из функции");
      case 4022: return(pref+"Система занята");
      case 4050: return(pref+"Неправильное количество параметров функции");
      case 4051: return(pref+"Недопустимое значение параметра функции");
      case 4052: return(pref+"Внутренняя ошибка строковой функции");
      case 4053: return(pref+"Ошибка массива");
      case 4054: return(pref+"Неправильное использование массива-таймсерии");
      case 4055: return(pref+"Ошибка пользовательского индикатора");
      case 4056: return(pref+"Массивы несовместимы");
      case 4057: return(pref+"Ошибка обработки глобальныех переменных");
      case 4058: return(pref+"Глобальная переменная не обнаружена");
      case 4059: return(pref+"Функция не разрешена в тестовом режиме");
      case 4060: return(pref+"Функция не разрешена");
      case 4061: return(pref+"Ошибка отправки почты");
      case 4062: return(pref+"Ожидается параметр типа string");
      case 4063: return(pref+"Ожидается параметр типа integer");
      case 4064: return(pref+"Ожидается параметр типа double");
      case 4065: return(pref+"В качестве параметра ожидается массив");
      case 4066: return(pref+"Запрошенные исторические данные в состоянии обновления");
      case 4067: return(pref+"Ошибка Time to выполнении торговой операции");
      case 4099: return(pref+"Конец файла");
      case 4100: return(pref+"Ошибка Time to работе с файлом");
      case 4101: return(pref+"Неправильное имя файла");
      case 4102: return(pref+"Слишком много открытых файлов");
      case 4103: return(pref+"Невозможно открыть файл");
      case 4104: return(pref+"Несовместимый режим доступа к файлу");
      case 4105: return(pref+"Ни один ордер не выбран");
      case 4106: return(pref+"Неизвестный символ");
      case 4107: return(pref+"Неправильный параметр цены для торговой функции");
      case 4108: return(pref+"Неверный номер тикета");
      case 4109: return(pref+"Торговля не разрешена. Необходимо включить опцию <Разрешить " +
                              "советнику торговать> в свойствах эксперта");
      case 4110: return(pref+"Длинные позиции не разрешены - необходимо проверить свойства эксперта");
      case 4111: return(pref+"Короткие позиции не разрешены - необходимо проверить свойства эксперта");
      case 4200: return(pref+"Объект уже существует");
      case 4201: return(pref+"Запрошено неизвестное свойство объекта");
      case 4202: return(pref+"Объект не существует");
      case 4203: return(pref+"Неизвестный тип объекта");
      case 4204: return(pref+"Нет имени объекта");
      case 4205: return(pref+"Ошибка координат объекта");
      case 4206: return(pref+"Не найдено указанное подокно");
      case 4207: return(pref+"Ошибка Time to работе с объектом");
      default:   return(pref+"Несуществующий номер ошибки");
   }
}