# Estimating Power Outage Duration With Scikit-Learn and Pandas DataFrames

## Introduction
<!-- Provide an introduction to your dataset, and clearly state the one question your homework is centered around. Why should readers of your website care about the dataset and your question specifically? Report the number of rows in the dataset, the names of the columns that are relevant to your question, and descriptions of those relevant columns.-->

## Data Cleaning and Exploratory Data Analysis
### Data Cleaning
(Some columns have been cropped so the table will display better.)
|   YEAR |   MONTH | U.S._STATE   | POSTAL.CODE   | NERC.REGION   | CLIMATE.REGION     |   ANOMALY.LEVEL | CLIMATE.CATEGORY   | OUTAGE.START.DATE   | OUTAGE.START.TIME   | OUTAGE.RESTORATION.DATE   | OUTAGE.RESTORATION.TIME   | CAUSE.CATEGORY     | CAUSE.CATEGORY.DETAIL   |   HURRICANE.NAMES |   OUTAGE.DURATION | DEMAND.LOSS.MW   | CUSTOMERS.AFFECTED   |
|-------:|--------:|:-------------|:--------------|:--------------|:-------------------|----------------:|:-------------------|:--------------------|:--------------------|:--------------------------|:--------------------------|:-------------------|:------------------------|------------------:|------------------:|:-----------------|:---------------------|
|   2011 |       7 | Minnesota    | MN            | MRO           | East North Central |            -0.3 | normal             | 2011-07-01          | 17:00:00            | 2011-07-03                | 20:00:00                  | severe weather     | nan                     |               nan |              3060 | <NA>             | 70000                |       11.6  |        9.18 |        6.81 |          9.28 |     2332915 |     2114774 |     2113291 |       6562520 |      35.5491 |      32.225  |      32.2024 |         2308736 |          276286 |           10673 |           2595696 |        88.9448 |        10.644  |       0.411181 |              51268 |            47586 |          1.07738 |                 1.6 |           4802 |          274182 |       1.75139 |             2.2 |      5348119 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |
|   2014 |       5 | Minnesota    | MN            | MRO           | East North Central |            -0.1 | normal             | 2014-05-11          | 18:38:00            | 2014-05-11                | 18:39:00                  | intentional attack | vandalism               |               nan |                 1 | <NA>             | <NA>                 |       12.12 |        9.71 |        6.49 |          9.28 |     1586986 |     1807756 |     1887927 |       5284231 |      30.0325 |      34.2104 |      35.7276 |         2345860 |          284978 |            9898 |           2640737 |        88.8335 |        10.7916 |       0.37482  |              53499 |            49091 |          1.08979 |                 1.9 |           5226 |          291955 |       1.79    |             2.2 |      5457125 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |
|   2010 |      10 | Minnesota    | MN            | MRO           | East North Central |            -1.5 | cold               | 2010-10-26          | 20:00:00            | 2010-10-28                | 22:00:00                  | severe weather     | heavy wind              |               nan |              3000 | <NA>             | 70000                |       10.87 |        8.19 |        6.07 |          8.15 |     1467293 |     1801683 |     1951295 |       5222116 |      28.0977 |      34.501  |      37.366  |         2300291 |          276463 |           10150 |           2586905 |        88.9206 |        10.687  |       0.392361 |              50447 |            47287 |          1.06683 |                 2.7 |           4571 |          267895 |       1.70627 |             2.1 |      5310903 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |
|   2012 |       6 | Minnesota    | MN            | MRO           | East North Central |            -0.1 | normal             | 2012-06-19          | 04:30:00            | 2012-06-20                | 23:00:00                  | severe weather     | thunderstorm            |               nan |              2550 | <NA>             | 68200                |       11.79 |        9.25 |        6.71 |          9.19 |     1851519 |     1941174 |     1993026 |       5787064 |      31.9941 |      33.5433 |      34.4393 |         2317336 |          278466 |           11010 |           2606813 |        88.8954 |        10.6822 |       0.422355 |              51598 |            48156 |          1.07148 |                 0.6 |           5364 |          277627 |       1.93209 |             2.2 |      5380443 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |
|   2015 |       7 | Minnesota    | MN            | MRO           | East North Central |             1.2 | warm               | 2015-07-18          | 02:00:00            | 2015-07-19                | 07:00:00                  | severe weather     | nan                     |               nan |              1740 | 250              | 250000               |       13.07 |       10.16 |        7.74 |         10.43 |     2028875 |     2161612 |     1777937 |       5970339 |      33.9826 |      36.2059 |      29.7795 |         2374674 |          289044 |            9812 |           2673531 |        88.8216 |        10.8113 |       0.367005 |              54431 |            49844 |          1.09203 |                 1.7 |           4873 |          292023 |       1.6687  |             2.2 |      5489594 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |
<!-- The parts cropped off the table, in case I want them later -->
<!--   RES.PRICE |   COM.PRICE |   IND.PRICE |   TOTAL.PRICE |   RES.SALES |   COM.SALES |   IND.SALES |   TOTAL.SALES |   RES.PERCEN |   COM.PERCEN |   IND.PERCEN |   RES.CUSTOMERS |   COM.CUSTOMERS |   IND.CUSTOMERS |   TOTAL.CUSTOMERS |   RES.CUST.PCT |   COM.CUST.PCT |   IND.CUST.PCT |   PC.REALGSP.STATE |   PC.REALGSP.USA |   PC.REALGSP.REL |   PC.REALGSP.CHANGE |   UTIL.REALGSP |   TOTAL.REALGSP |   UTIL.CONTRI |   PI.UTIL.OFUSA |   POPULATION |   POPPCT_URBAN |   POPPCT_UC |   POPDEN_URBAN |   POPDEN_UC |   POPDEN_RURAL |   AREAPCT_URBAN |   AREAPCT_UC |   PCT_LAND |   PCT_WATER_TOT |   PCT_WATER_INLAND |-->
<!--------------:|------------:|------------:|--------------:|------------:|------------:|------------:|--------------:|-------------:|-------------:|-------------:|----------------:|----------------:|----------------:|------------------:|---------------:|---------------:|---------------:|-------------------:|-----------------:|-----------------:|--------------------:|---------------:|----------------:|--------------:|----------------:|-------------:|---------------:|------------:|---------------:|------------:|---------------:|----------------:|-------------:|-----------:|----------------:|-------------------:|-->

### Univariate Analysis
<iframe
  src="assets/outages_per_state.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### Bivariate Analysis
<iframe
  src="assets/duration_by_cause.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

<iframe
  src="assets/cause_and_type.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### Interesting Aggregates
| CLIMATE.REGION     |     cold |   normal |     warm |
|:-------------------|---------:|---------:|---------:|
| Central            |  97364.5 | 154126   |  91868.9 |
| East North Central | 120777   | 154613   | 101711   |
| Northeast          | 120320   | 125291   | 115444   |
| Northwest          |  63350.3 |  42961.1 | 118436   |
| South              | 186071   | 218077   | 107235   |
| Southeast          | 142545   | 164579   | 268594   |
| Southwest          |  63196.1 |  24918.2 |  29596.2 |
| West               | 171930   | 186294   | 231772   |
| West North Central |  53500   |  60106   |  25250   |


| NERC.REGION   |     cold |   normal |     warm |
|:--------------|---------:|---------:|---------:|
| ECAR          | 171452   |   297697 | 162807   |
| FRCC          | 111350   |   247543 | 492172   |
| HECO          |      0   |    29300 | 175443   |
| HI            | 294000   |        0 |      0   |
| MRO           |  64887.5 |   107845 |  67188.2 |
| NPCC          |  77485.2 |   126256 | 108519   |
| PR            |      0   |    62000 |      0   |
| RFC           | 125911   |   132717 | 114369   |
| SERC          | 133464   |   110576 |  68298.5 |
| SPP           | 307020   |   229598 |  67836.1 |
| TRE           | 170254   |   281004 | 153432   |
| WECC          | 120987   |   123306 | 159530   |
### Imputation
<iframe
  src="assets/distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

## Framing a Prediction Problem
<!-- -->

## Baseline Model
<!-- -->
|    training |   validation |     testing |
|------------:|-------------:|------------:|
| 2.74311e+07 |  6.36999e+06 | 4.19289e+07 |

## Final Model
<!-- -->
|    training |   validation |     testing |
|------------:|-------------:|------------:|
| 2.74311e+07 |  6.36999e+06 | 4.19289e+07 |
| 2.70623e+07 |  5.93904e+06 | 4.23442e+07 |
| 2.67602e+07 |  6.32626e+06 | 4.2012e+07  |
| 2.68503e+07 |  5.77629e+06 | 4.26924e+07 |
| 1.05517e+07 |  7.66414e+06 | 5.35662e+07 |
