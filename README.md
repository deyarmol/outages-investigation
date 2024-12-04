# Estimating Power Outage Duration With Scikit-Learn and Pandas DataFrames

## Introduction
Welcome! On this page I share the results of a project I completed for a data science course at the University of Michigan.
<!-- Provide an introduction to your dataset, and clearly state the one question your homework is centered around. Why should readers of your website care about the dataset and your question specifically? Report the number of rows in the dataset, the names of the columns that are relevant to your question, and descriptions of those relevant columns.-->
### Data Introduction and Question Identification
For this project I chose to work with [this spreadsheet](https://engineering.purdue.edu/LASCI/research-data/outages) from civil engineering researchers at Purdue University. It contains thorough data on major power outages across the United States from January 2000 to July 2016. Each row contains information on a single outage, reporting details like time, location, cause, and impact (which includes how long the outage lasted, how much it cost, and how many people were affected). The whole table has 1534 rows and 55 columns. 

The general question I wanted to answer with this data was: based on where you are, how much will you be affected by major power outages? For example, can you predict how long a power outage will last in a warm urban area in the southeast? 

The relevant columns of the data are singled out and described below.
- 'MONTH': When the outage took place in the year. 
- 'CLIMATE.REGION': Which National Centers for Environmental Information designated climate region the outage took place in. ([Interactive map here](https://www.ncei.noaa.gov/access/monitoring/reference-maps/us-climate-regions), though the regions seem to have been renamed since this data was released.)
- 'NERC.REGION': Which North American Electric Reliability Corporation (NERC) region the outage took place in. ([Map of these regions](https://www.eia.gov/electricity/data/eia411/#tabs_NERC-2)) I chose to use this as well as the more geographical 'CLIMATE.REGION' column because while they largely overlap, I think allowing for differences between "climate region" and "power region" is worthwhile.
- 'ANOMALY.LEVEL' and 'CLIMATE.CATEGORY': the effects of El Niño at the time and place of the outage, based on the [Oceanic Niño Index](https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php).
- 'AREAPCT_URBAN' and 'AREAPCT_UC': how much of the state's land is taken up by urban (population >50,000) and urban cluster (population 2,500 - 50,000 people) areas, in percent. 

## Data Cleaning and Exploratory Data Analysis
### Data Cleaning
<!-- Describe, in detail, the data cleaning steps you took and how they affected your analyses. The steps should be explained in reference to the data generating process. Show the head of your cleaned DataFrame (see Part 2: Report for instructions).-->

I started by reading the Excel spreadsheet into a Pandas DataFrame. I read the raw data, set the correct row indeces and column names, and converted numerical columns from strings to integers and floats. I also converted the start and end time and date columns from strings to timestamp objects. The first few rows of this cleaned DataFrame can be seen below, though most of the columns have been cropped for space. Some rows were missing almost all data (effectively only reporting that an outage occured in a state once), which were dropped due to not providing enough detail to meaningfully impute.

|   YEAR |   MONTH | POSTAL.CODE   | NERC.REGION   | CLIMATE.REGION     |   ANOMALY.LEVEL | CLIMATE.CATEGORY   | OUTAGE.START.DATE   | OUTAGE.START.TIME   | OUTAGE.RESTORATION.DATE   | OUTAGE.RESTORATION.TIME   | CAUSE.CATEGORY     | CAUSE.CATEGORY.DETAIL   |   OUTAGE.DURATION |
|-------:|--------:|:--------------|:--------------|:-------------------|----------------:|:-------------------|:--------------------|:--------------------|:--------------------------|:--------------------------|:-------------------|:------------------------|------------------:|
|   2011 |       7 | MN            | MRO           | East North Central |            -0.3 | normal             | 2011-07-01          | 17:00:00            | 2011-07-03                | 20:00:00                  | severe weather     | nan                     |              3060 |
|   2014 |       5 | MN            | MRO           | East North Central |            -0.1 | normal             | 2014-05-11          | 18:38:00            | 2014-05-11                | 18:39:00                  | intentional attack | vandalism               |                 1 |
|   2010 |      10 | MN            | MRO           | East North Central |            -1.5 | cold               | 2010-10-26          | 20:00:00            | 2010-10-28                | 22:00:00                  | severe weather     | heavy wind              |              3000 |
|   2012 |       6 | MN            | MRO           | East North Central |            -0.1 | normal             | 2012-06-19          | 04:30:00            | 2012-06-20                | 23:00:00                  | severe weather     | thunderstorm            |              2550 |
|   2015 |       7 | MN            | MRO           | East North Central |             1.2 | warm               | 2015-07-18          | 02:00:00            | 2015-07-19                | 07:00:00                  | severe weather     | nan                     |              1740 |


### Univariate Analysis
<!-- Embed at least one plotly plot you created in your notebook that displays the distribution of a single column (see Part 2: Report for instructions). Include a 1-2 sentence explanation about your plot, making sure to describe and interpret any trends present, and how they answer your initial question. (Your notebook will likely have more visualizations than your website, and that’s fine. Feel free to embed more than one univariate visualization in your website if you’d like, but make sure that each embedded plot is accompanied by a description.)-->
<iframe
  src="assets/outages_per_state.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

While I didn't use the state column directly, I started out by plotting the distribution of outages by state. A handful of states were the sources of a significant portion of the outages. 

<iframe
  src="assets/duration_by_state.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Later, I plotted the average duration of outages by state. This plot looks noticably different from the one above, suggesting outage frequency is unrelated to duration. 
(I was surprised to see Wisconsin of all states having the highest average; turns out the longest recorded duration for an outage in the data is 75 days, held by one from Wisconsin from 2014. [The state had severe storms that year](https://www.greenbaypressgazette.com/story/news/local/2014/06/30/crews-work-to-restore-power-in-southern-wisconsin/11817885/), so this checks out.)

### Bivariate Analysis
<!--Embed at least one plotly plot that displays the relationship between two columns. Include a 1-2 sentence explanation about your plot, making sure to describe and interpret any trends present and how they answer your initial question. (Your notebook will likely have more visualizations than your website, and that’s fine. Feel free to embed more than one bivariate visualization in your website if you’d like, but make sure that each embedded plot is accompanied by a description.) -->
<iframe
  src="assets/duration_by_cause.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This plot shows the distribution of outages based on cause. Many of them ('intentional attack' and 'severe weather in particular') have significant amounts of outliers, which will make prediction difficult.

### Interesting Aggregates
<!--Embed at least one grouped table or pivot table in your website and explain its significance. -->

The table below shows the average duration of outages by NCEI region and overall climate category. One could try to make takeaways like "outages in the West North Central region are 87 times worse when El Niño makes the climate warmer," but I think the relationships here are simple enough to allow for statements based on one pivot table.

| CLIMATE.REGION     |     cold |    normal |    warm |
|:-------------------|---------:|----------:|--------:|
| Central            | 2799.86  | 2708.7    | 2413.84 |
| East North Central | 6568.79  | 5271.22   | 3022.12 |
| Northeast          | 3657.25  | 2261.33   | 4175.91 |
| Northwest          |  874.681 |  733.612  | 3063.54 |
| South              | 2012.71  | 3753.06   | 1861.4  |
| Southeast          | 1707.07  | 2392.27   | 2528.94 |
| Southwest          |  544.591 |  296.136  | 5127.68 |
| West               | 1762.71  | 1249.84   | 2044.23 |
| West North Central |  200     |   28.4286 | 2486.5  |

### Imputation
<!--If you imputed any missing values, visualize the distributions of the imputed columns before and after imputation. Describe which imputation technique you chose to use and why. If you didn’t fill in any missing values, discuss why not. -->
To deal with missing duration values, I imputed them with the overall median duration (since no relationship was clear between the rows missing values). I chose to use the median over the mean due to how many outliers can be seen in the data. The effect this had on distribution is visible below; the added values end up filling out the curved shape of the distribution.
<iframe
  src="assets/distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>



## Framing a Prediction Problem
<!--Clearly state your prediction problem and type (classification or regression). If you are building a classifier, make sure to state whether you are performing binary classification or multiclass classification. Report the response variable (i.e. the variable you are predicting) and why you chose it, the metric you are using to evaluate your model and why you chose it over other suitable metrics (e.g. accuracy vs. F1-score).

Note: Make sure to justify what information you would know at the “time of prediction” and to only train your model using those features. For instance, if we wanted to predict your Final Exam grade, we couldn’t use your Portfolio Homework grade, because we (probably) won’t have the Portfolio Homework graded before the Final Exam! Feel free to ask questions if you’re not sure. -->

The specific problem my model was to solve is this: given information about the time, location, and environmental conditions of a theoretical major power outage, predict the duration of the outage. This is a straightforward regression problem. I will gauge performance using mean squared error, due to its moderate sensitivity toward outliers. 

## Baseline Model
<!--Describe your model and state the features in your model, including how many are quantitative, ordinal, and nominal, and how you performed any necessary encodings. Report the performance of your model and whether or not you believe your current model is “good” and why.

Tip: Make sure to hit all of the points above: many Portfolio Homeworks in the past have lost points for not doing so. -->
My baseline model was a basic linear regression model, with the only transformation involved being one-hot encoding all categorical columns (making all features quantitative or nominal). The model's performance is shown in the table below, showing mean squared error for the data it was trained on, cross-validation data, and the test data.

|    Training |   Validation |     Testing |
|------------:|-------------:|------------:|
| 3.12349e+07 |  6.2102e+06  | 4.68994e+07 |

This is servicable, but not great: it's effectively throwing all the variables together and taking whatever comes out. There's a lot of room to improve.

## Final Model
<!--State the features you added and why they are good for the data and prediction task. Note that you can’t simply state “these features improved my accuracy”, since you’d need to choose these features and fit a model before noticing that – instead, talk about why you believe these features improved your model’s performance from the perspective of the data generating process.

Describe the modeling algorithm you chose, the hyperparameters that ended up performing the best, and the method you used to select hyperparameters and your overall model. Describe how your Final Model’s performance is an improvement over your Baseline Model’s performance.

Optional: Include a visualization that describes your model’s performance, e.g. a confusion matrix, if applicable. -->

First, I put all numerical features through a pipeline consisting of a standard scaler and a polynomial feature generator. The polynomial degree was chosen from 1 to 5 via grid search. (I tried separating the different features out into separate pipelines, but I never found a combination that surpassed this naive one). Additionally, I changed the encoding of the climate category column from one-hot to ordinal (so the values [cold, normal, warm] became [0, 1, 2])—this better communicates that the three are on a scale. I tried several different modeling algorithms—specifically linear regression, LASSO, ridge regression, and k nearest neighbors. Ridge's alpha parameter and KNN's number of neighbors were also chosen with grid search. 

|   Algorithm |    Training |   Validation |     Testing |
|------------:|------------:|-------------:|------------:|
|    Baseline | 3.12349e+07 |  6.2102e+06  | 4.68994e+07 |
|      Linear | 3.033e+07   |  5.58904e+06 | 4.62628e+07 |
|       LASSO | 3.03334e+07 |  5.54924e+06 | 4.62721e+07 |
|       Ridge | 3.06578e+07 |  5.07339e+06 | 4.68769e+07 |
|         KNN | 2.50766e+07 |  9.96246e+06 | 4.42579e+07 |

Comparing each version's results, I decided to go with the one using ridge regression as my final model. With the lowest validation error by far, I trust it the most to generalize the most to unseen data.

Using this new model I can predict that, in Michigan, during April, while the climate is warmer than usual, the expected duration of a major power outage is... 78 hours! A similar outage at the same time in California would only be 30 hours. Maybe our regional outage response infrastructure needs some work...