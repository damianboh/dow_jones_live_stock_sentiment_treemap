# libraries for webscraping, parsing and getting stock data
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import yfinance as yf
import time

# for plotting and data manipulation
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.express as px

# NLTK VADER for sentiment analysis
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# for getting current date and time to print 'last updated'
from datetime import datetime

# Get All Tickers from Dow Jones Index
df_dow_jones = pd.read_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")[1]
tickers = df_dow_jones['Symbol'].tolist()

# Scrape the Date, Time and News Headlines Data
finwiz_url = 'https://finviz.com/quote.ashx?t='
news_tables = {}

for ticker in tickers:
    url = finwiz_url + ticker
    req = Request(url=url,headers = { "user-Agent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'})

    try:       
       response = urlopen(req)   
    except:
       time.sleep(10) # if there is an error and request is blocked, do it more slowly by waiting for 10 seconds before requesting again
       response = urlopen(req)  
        
    # Read the contents of the file into 'html'
    html = BeautifulSoup(response, "html")
    # Find 'news-table' in the Soup and load it into 'news_table'
    news_table = html.find(id='news-table')
    # Add the table to our dictionary
    news_tables[ticker] = news_table	

# Parse the Date, Time and News Headlines into a Python List
parsed_news = []
# Iterate through the news
for file_name, news_table in news_tables.items():
	# Iterate through all tr tags in 'news_table'
	for x in news_table.findAll('tr'):
		# occasionally x (below) may be None when the html table is poorly formatted, skip it in try except instead of throwing an error and exiting
		# may also use an if loop here to check if x is None first
		try: 
			# read the text from each tr tag into text
			# get text from a only
			text = x.a.get_text() 
			# splite text in the td tag into a list 
			date_scrape = x.td.text.split()
			# if the length of 'date_scrape' is 1, load 'time' as the only element
			if len(date_scrape) == 1:
				time = date_scrape[0]

			# else load 'date' as the 1st element and 'time' as the second    
			else:
				date = date_scrape[0]
				time = date_scrape[1]
			# Extract the ticker from the file name, get the string up to the 1st '_'  
			ticker = file_name.split('_')[0]
			print(ticker)

			# Append ticker, date, time and headline as a list to the 'parsed_news' list
			parsed_news.append([ticker, date, time, text])
		except Exception as e:
			print(e)

# Perform Sentiment Analysis with Vader
# Instantiate the sentiment intensity analyzer
vader = SentimentIntensityAnalyzer()
# Set column names
columns = ['ticker', 'date', 'time', 'headline']
# Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)

# Iterate through the headlines and get the polarity scores using vader
scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()
# Convert the 'scores' list of dicts into a DataFrame
scores_df = pd.DataFrame(scores)

# Join the DataFrames of the news and the list of dicts
parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')
# Convert the date column from string to datetime
parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date

# Group by each ticker and get the mean of all sentiment scores
mean_scores = parsed_and_scored_news.groupby(['ticker']).mean()

# Get Market Cap, Sector and Industry of each Ticker
sectors = []
industries = []
marketcap = []
for ticker in tickers:
    print(ticker)
    tickerdata = yf.Ticker(ticker)
    marketcap.append(tickerdata.info['marketCap'])
    sectors.append(tickerdata.info['sector'])
    industries.append(tickerdata.info['industry'])
	
# Combine the Information Above and the Corresponding Tickers into a DataFrame
d = {'Symbol': tickers, 'Sector': sectors, 'Industry': industries, 'Market Cap': marketcap}
# create dataframe from 
df_info = pd.DataFrame(data=d)

# Get Names of Companies from the Dow Jones DataFrame obtained Earlier
df_info_name = df_info.merge(df_dow_jones[['Company', 'Symbol']], on = 'Symbol')

# Join Stock Information and Sentiment Information
df = mean_scores.merge(df_info_name, left_on = 'ticker', right_on = 'Symbol')
df = df.rename(columns={"compound": "Sentiment Score", "neg": "Negative", "neu": "Neutral", "pos": "Positive"})

# Generate the Treemap Plot
# group data into sectors at the highest level, breaks it down into industry, and then ticker, specified in the 'path' parameter
# the 'values' parameter uses the value of the column to determine the relative size of each box in the chart
# the color of the chart follows the sentiment score
# when the mouse is hovered over each box in the chart, the negative, neutral, positive and overall sentiment scores will all be shown
# the color is red (#ff0000) for negative sentiment scores, black (#000000) for 0 sentiment score and green (#00FF00) for positive sentiment scores
fig = px.treemap(df, path=[px.Constant("Dow Jones"), 'Sector', 'Industry', 'Symbol'], values='Market Cap',
                  color='Sentiment Score', hover_data=['Company', 'Negative', 'Neutral', 'Positive', 'Sentiment Score'],
                  color_continuous_scale=['#FF0000', "#000000", '#00FF00'],
                  color_continuous_midpoint=0)

fig.data[0].customdata = df[['Company', 'Negative', 'Neutral', 'Positive', 'Sentiment Score']].round(3) # round to 3 decimal places
fig.data[0].texttemplate = "%{label}<br>%{customdata[4]}"

fig.update_traces(textposition="middle center")
fig.update_layout(margin = dict(t=30, l=10, r=10, b=10), font_size=20)

# Get current date, time and timezone to print to the html page
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
timezone_string = datetime.now().astimezone().tzname()

# Generate HTML File with Updated Time and Treemap
with open('dow_jones_live_sentiment.html', 'a') as f:
    f.truncate(0) # clear file if something is already written on it
    title = "<h1>Dow Jones Stock Sentiment Dashboard</h1>"
    updated = "<h2>Last updated: " + dt_string + " (Timezone: " + timezone_string + ")</h2>"
    description = "This dashboard is updated every half an hour with sentiment analysis performed on latest scraped news headlines from the FinViz website.<br><br>"
    code = """<a href="https://medium.com/datadriveninvestor/use-github-actions-to-create-a-live-stock-sentiment-dashboard-online-580a08457650">Explanatory Article</a> | <a href="https://github.com/damianboh/dow_jones_live_stock_sentiment_treemap">Source Code</a>"""
    author = """ | Created by Damian Boh, check out my <a href="https://damianboh.github.io/">GitHub Page</a>"""
    f.write(title + updated + description + code + author)
    f.write(fig.to_html(full_html=False, include_plotlyjs='cdn')) # write the fig created above into the html file

