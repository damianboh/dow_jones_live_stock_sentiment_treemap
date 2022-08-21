# dow_jones_live_stock_sentiment_treemap

This project keeps an updated treemap dashboard of stock sentiments in the Dow Jones Index, deployed here: https://damianboh.github.io/dow_jones_live_sentiment.html

A workflow is configured in GitHub actions to install necessary libraries from requirements.txt, run the Python script "update_sentiment_page.py" that scrapes FinViz for financial headlines, perform sentiment analysis and generate the updated html page.

The html page is then pushed to my Github pages repository.

An accompanying Jupyter notebook "get_dow_jones_stock_sentiment.ipynb" is included for exploration, it has similar code to the .py script and shows the output at every step.
