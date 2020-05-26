# CNBC news article's scrapper
import requests 
from bs4 import BeautifulSoup

def extract_article (url):
    # request for article web page
    req = requests.get(url) 

    # extract html data using the 'lxml' parser  
    soup = BeautifulSoup(req.content, 'lxml')

    # extract news article's headline  
    headline = soup.find('h1', class_="ArticleHeader-headline").text

    # extract news description
    description = ''
    news_desc = soup.find_all('div', class_="group")
    for desc in news_desc:
        description = description + desc.text

    return headline + description

# if __name__ == "__main__":
#     extract_article("https://www.cnbc.com/2020/05/22/coronavirus-goldman-sachs-on-india-growth-gdp-forecast.html")