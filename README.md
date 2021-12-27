# Project overview
<p>Machine learning approach to predict stocks prices, this is done to satisfy the requirements for the final project in the Udacity Data Scientist Nanodegree.</p>
<p> the project consists of two main parts: the Jupyter notebook and the Flask web app. 


### Table of Contents
- [Installation](#installation)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [File Descriptions](#file-descriptions)
- [Analysis Results](#analysis-results)
- [Acknowledgements](#acknowledgements)
- [References](#References)


## Installation 
the project uses python 3.<br>
In order to run the Jupyter notebook the following libraries are needed:<br>
1-yfinance.<br>
2-Pandas.<br>
3-seaborn.<br>
4-matplotlib.<br>
5-statsmodels.<br>
6-pmdarima.<br>
7-warnings.<br>
8-numpy.<br>
9-sklearn.preprocessing.<br>
10-datetime.<br>
11-math.<br>
12-keras.models.<br>
13-keras.layers.<br>
  
In order to run the Web app the following libraries are needed:<br>
 1-flask <br>
 2-yfinance <br>
 3-matplotlib <br>
 4-numpy <br>
 5-MinMaxScaler <br>
  6-math <br> 
  7-keras.models <br>
  8-keras.layers <br>
  
  To run the web app in the local host: 
  <br>1-Open the project folder in the command line interface.<br>
  2-Run the following line: python main.py <br>
  3-after the following message appear. Open the browser and enter the following url: http://127.0.0.1:3000/ <br>
  <img width="473" alt="Screen Shot 1443-05-23 at 3 59 02 PM" src="https://user-images.githubusercontent.com/41934760/147474016-ec65289d-83f1-45fa-8b0a-8677e1e50180.png">
  
  
  
## Motivation 
  
  Stock market prediction can be hard for many people as it's affected by social, economical and political factors.
One way to predict the prices is stocks prices analysis which is a method for investors to make buying and selling decisions. there are two ways to do this analysis:<br>
1-Fundamental Analysis: this method relies on studying the financial aspects of business such as financial statements and economic factors to analyze the market value of the shares.<br>
2-Technical Analysis: this method relies on studying the trends and patterns in the historical prices data.<br>
In this project a machine learning approach will be followed.
  

## Dataset
Dataset used are fetched from Yahoo finance through a python library called "yfinance". Two compaines were used to demonstrate the model.
<br>1-Tesla, Inc. designs, develops, manufactures, leases, and sells electric vehicles, and energy generation and storage systems in the United States, China, and internationally. The company operates in two segments, Automotive, and Energy Generation and Storage.

2-Amazon.com, Inc. engages in the retail sale of consumer products and subscriptions in North America and internationally. The company operates through three segments: North America, International, and Amazon Web Services (AWS). It sells merchandise and content purchased for resale from third-party sellers through physical and online stores

## File Descriptions
  
## Analysis Results 
 
  ARIMA results.
  
  As it can be seen from the following charts, ARIMA model performance isn't good. the difference between the test set and the predicted values is huge. 
  
  ![image](https://user-images.githubusercontent.com/41934760/147472306-2705a615-4336-4469-8cb7-16717c3ac700.png)

  ![image](https://user-images.githubusercontent.com/41934760/147472314-64e2805e-c0c3-4987-bde7-b23886ccad10.png)
  


  LSTM results. 
  
  As it can be seen from the following charts, LSTM model performance is good, the test set values and the predicted values are very close. 
  
  ![image](https://user-images.githubusercontent.com/41934760/147471030-dd52cb87-74c6-46d4-a44d-b63c8ab0cfa7.png)

  
  
![image](https://user-images.githubusercontent.com/41934760/147471878-e7dd2ca0-311f-45eb-aac6-2fc1029361c9.png)

LSTM did better than ARIMA, this could be due to LSTM being based on neural network while ARIMA is a Statistical model. Also ARIMA might be better for short-term forecasting while LSTM works better for long term forecasting. Moreover, ARIMA requires stationary dataset while the dataset used in this project are not stationary. Another difference between the two models is that LSTM is nonlinear while ARIMA is linear. 
  
## Acknowledgements
  Yahoo Finance for providing the data.<br>
  Udacity for its phenomenal data science course.<br>
  All websites mentioned for providing information that helped in this project.<br>
  
## References 
  https://www.wallstreetmojo.com/fundamental-analysis-vs-technical-analysis/ <br>
  https://analyticsindiamag.com/a-guide-to-different-evaluation-metrics-for-time-series-forecasting-models/ <br>
  https://finance.yahoo.com/quote/TSLA/profile?p=TSLA <br>
  https://finance.yahoo.com/quote/AMZN/profile?p=AMZN <br> 
  https://www.kaggle.com/minatverma/nse-stocks-data <br>
  https://www.analyticsvidhya.com/blog/2021/03/introduction-to-long-short-term-memory-lstm/ <br> 
  https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/ <br> 
  https://towardsdatascience.com/introduction-to-time-series-forecasting-part-2-arima-models-9f47bf0f476b <br>
  https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/ <br>
  https://www.youtube.com/watch?v=QIUxPv5PJOY <br>
  
  

