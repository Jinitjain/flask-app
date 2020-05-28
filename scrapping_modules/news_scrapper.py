# money control news article's scrapper
import requests 
from bs4 import BeautifulSoup

def extract_article (url):
    # request for article web page
    req = requests.get(url) 

    # extract html data using the 'lxml' parser  
    soup = BeautifulSoup(req.content, 'lxml')

    # extract news article
    news_article = ''
    paras = soup.find_all('p')

    for para in paras:
        news_article = news_article + para.text

    print(news_article)
    return news_article

if __name__ == "__main__":
    doc = extract_article("https://www.moneycontrol.com/news/business/markets/reliance-industries-rights-entitlement-jumps-to-rs-209-90-up-16-intraday-5321661.html")