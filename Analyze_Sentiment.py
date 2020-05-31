DEBUG = False

import nltk
nltk.download('punkt')
nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/stopwords.zip')
    #nltk.data.find('tokenizers/punkt.zip')
    nltk.data.find('taggers/averaged_perceptron_tagger.zip')
    #nltk.data.find('chunkers/maxent_ne_chunker.zip')
    nltk.data.find('corpora/words.zip')
    nltk.data.find('stemmers/snowball_data.zip')
except LookupError:
    nltk.download('stopwords')
    #nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    #nltk.download('maxent_ne_chunker')
    nltk.download('words') 
    nltk.download('snowball_data')

from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
import re
import numpy as np
import pandas as pd

# !pip3 install textblob
# !python3 -m textblob.download_corpora

from textblob import TextBlob


# In[2]:


# nltk.download()
from nltk.corpus import stopwords
stop = stopwords.words('english')
# stop = list(map(lambda x: x.upper(),stop))
stop.remove('in')
stop.remove('at')
stop.remove('and')
stop.remove('to')


# ## Get article in document

# ### Jinit Code

# In[3]:


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

    return headline, description

document = extract_article("https://www.cnbc.com/2020/05/22/coronavirus-goldman-sachs-on-india-growth-gdp-forecast.html")


def nltk_process(document):
    document = " ".join([i for i in document.split() if i not in stop])
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    #print(sentences)
    return sentences

def nltk_eval(document):
    chunkGram = r"""
    ADDRESS: {<JJ.?|CD.?>+<CD.?|JJ.?|NNP.?>+<CD|NNP>+}
    TEST : {}
    NP: {<DT|JJ|NN.*>+}
    """
    organization = {}
    location = {}
    chunkParser = nltk.RegexpParser(chunkGram)
    for tagged_sentence in nltk_process(document):
            for chunk in nltk.ne_chunk(tagged_sentence):
                if type(chunk) == nltk.tree.Tree:
                    #print((chunk))
                    if chunk.label() == 'PERSON':
                        ## Organization is found
                        temp = (' '.join([c[0] for c in chunk])).lower()
                        if temp in organization:
                            organization[temp] += 1
                        else:
                            organization[temp] = 1
                    if chunk.label() == 'ORGANIZATION':
                        ## Organization is found
                        temp = (' '.join([c[0] for c in chunk])).lower()
                        if temp in organization:
                            organization[temp] += 1
                        else:
                            organization[temp] = 1
                    if chunk.label() == 'GPE':
                        temp = ' '.join([c[0] for c in chunk])
                        if temp in location:
                            location[temp] += 1
                        else:
                            location[temp] = 1
    organization = {k: v for k, v in sorted(organization.items(), key=lambda item: item[1], reverse=True)}
    location = {k: v for k, v in sorted(location.items(), key=lambda item: item[1], reverse=True)}
    
    
    #print("Location:\n",list(location.keys()))
    
    organization.update(location)
    print("Organizations:\n",list(organization.keys()))
    
    return organization, location
    # return names, organization, location, address, other



def clean_companies(companies):
    cleaned_companies = []
    for c in companies:
        x = c.lower()
        if x.endswith('limited'):
            x = x[:-8]
        elif x.endswith('ltd'):
            x = x[:-4]
        elif x.endswith('limited.'):
            x = x[:-9]
        elif x.endswith('ltd.'):
            x = x[:-5]
        else:
            #print(x)
            pass
        cleaned_companies.append(x)
    return cleaned_companies

def clean_subsectors(subsectors):
    global stemmer
    stemmer = SnowballStemmer("english")
    modified_subsectors = [re.split(' |-',x) for x in subsectors]
    real_subsectors = [re.split(' & ',x) for x in subsectors]

    remove_list = ['&', 'others','services','management', 'application', 'manufacturing', "", 'integrated',
                  'independent', 'producers','manufacturers', 'general', 'inputs','facilities','production'
                  'providers', 'components','products', 'other', 'development','companies','devices','supplies',
                  'equipment', 'parts', '&']
    for msubsectors in modified_subsectors:
        for x in remove_list:
            if x in msubsectors:
                msubsectors.remove(x)
    clean_subsectors = []
    for item in list(zip(real_subsectors, modified_subsectors)):
        temp = [x.strip() for x in item[0]]
        temp.extend(item[1])
        temp = list(set(temp))
        clean_subsectors.append(temp)

    dict_subsectors = {}
    for key, value in dict(zip(subsectors, clean_subsectors)).items():
        value2 = [stemmer.stem(x) for x in value]
        #print(key, " " , value, " ", value2)
        dict_subsectors[key] = value2
        
    return dict_subsectors




def preprocess_data():
    global df, cleaned_companies, cleaned_subsectors, tickers
    df = pd.read_csv('finalBetaDB.csv')
    df.rename(columns={' Sector': 'Sector', ' Sub-sector':'Subsector'}, inplace=True)
    df = df.replace(np.nan, '', regex=True)
    
    cols = ['Sector', 'Subsector']
    for col in cols:
        df[col] = df[col].str.strip()
        
    cleaned_companies = clean_companies(df['Company'])
    cleaned_subsectors = clean_subsectors(df['Subsector'])
    tickers = df['Symbol'].values

    df['cleaned_companies'] = cleaned_companies
    #df['cleaned_subsectors'] = cleaned_subsectors
    
    return df, cleaned_companies, cleaned_subsectors, tickers


# In[48]:
##  preprocess_data()

#df, cleaned_companies, cleaned_subsectors = preprocess_data()


# In[10]:



def find_subsectors(organizations, cleaned_subsectors):
    """Find subsectors in organization"""
    
    organ_to_subsector = {}
    for subsector, subsector_values in cleaned_subsectors.items():
        for organization in organizations.keys():
            if stemmer.stem(organization) in subsector_values:
                if organization not in organ_to_subsector:
                    organ_to_subsector[organization] = list()
                organ_to_subsector[organization].append(subsector)
    if DEBUG: print("Organization to subsector: \n", organ_to_subsector)
    return organ_to_subsector

def find_companies(organizations, cleaned_companies, tickers):
    
    organ_to_company = {}
    for company, symbol in zip(cleaned_companies, tickers):
        for organization in organizations.keys():
            x = organization.lower()+' '
            y = organization.lower()
            if company.startswith(x) or (company == y) or (y == symbol.lower()):
                if organization not in organ_to_company:
                    organ_to_company[organization] = list()
                organ_to_company[organization].append(company)
    if DEBUG: print('Organization to company : \n', organ_to_company)
    return organ_to_company

def find_organ_context(document, organization):
    count_sentences = 0
    organ_to_sentenceid = {}
    for sentence in document.split('.'):
        count_sentences += 1
        for organ in organization.keys():
            if organ in sentence.lower():
                if organ not in organ_to_sentenceid:
                    organ_to_sentenceid[organ] = list()
                organ_to_sentenceid[organ].append(count_sentences)
    return organ_to_sentenceid, count_sentences

def remove_other_organization(organizations, organ_to_subsector, organ_to_company):
    dict_organizations = organizations.copy()
    for organization in organizations.keys():
        if (organization not in organ_to_subsector.keys()) and (organization not in organ_to_company.keys()):
            dict_organizations.pop(organization)
    if DEBUG: print("Required organizations \n ",dict_organizations)
    return dict_organizations



# In[16]:




# In[18]:

# In[19]:


def find_sentiment_of_context(document, organ_to_sentenceid, total_sentences):
    list_of_sentences = document.split('.')
    polarity_of_organ = {}

    for item, value in organ_to_sentenceid.items():
        polarity_of_organ[item] = list()
        for i in value:
            lower_value = i-1
            upper_value = i+2
            if lower_value < 0:
                lower_value = 0
            if upper_value > total_sentences:
                upper_value = total_sentences
            current_context = str(list_of_sentences[lower_value:upper_value+1])
            current_context = TextBlob(current_context)
            polarity_of_organ[item].append(current_context.sentiment.polarity)

    #print(polarity_of_organ)
    return polarity_of_organ

# In[20]:


def find_max_or_min_value(arr):
    maxi = max(arr)
    mini = min(arr)
    
    return maxi if abs(maxi) > abs(mini) else mini

def distribute_polarity(polarity_of_organ, organ_to_subsector, organ_to_company):
    subsector_to_polarity = {}
    company_to_polarity = {}
    if DEBUG: print("Polarity of organization\n",polarity_of_organ)
    if DEBUG: print("Organ to company\n",organ_to_company)
    for organization in polarity_of_organ:
        value = find_max_or_min_value(polarity_of_organ[organization])
        
        if organization in organ_to_subsector.keys():
            for subsector in organ_to_subsector[organization]:
                if subsector in subsector_to_polarity:
                    value = subsector_to_polarity[subsector] if abs(subsector_to_polarity[subsector]) > abs(value) else value
                subsector_to_polarity[subsector] = value
                
        if organization in organ_to_company.keys():
            for company in organ_to_company[organization]:
                if company in company_to_polarity:
                    value = company_to_polarity[company] if abs(company_to_polarity[company]) > abs(value) else value
                company_to_polarity[company] = value
        
    if DEBUG: print("Subsector to polarity\n", subsector_to_polarity)
    if DEBUG: print("Company to polarity\n", company_to_polarity)
    return subsector_to_polarity, company_to_polarity


# ## One stop evaluater

# In[31]:


def find_subsector_company_sentiment_json_format(document):
    #document = " ".join(document)
    #print(document)
    organization, _ = nltk_eval(document)
    #print(organization)
    organ_to_subsector = find_subsectors(organization, cleaned_subsectors)
    organ_to_company = find_companies(organization, cleaned_companies, tickers)
    
    updated_organizatation = remove_other_organization(organization, organ_to_subsector, organ_to_company)
    organ_to_sentenceid, total_sentences = find_organ_context(document, updated_organizatation)
    
    polarity_of_organ = find_sentiment_of_context(document, organ_to_sentenceid, total_sentences)
    subsector_to_polarity, company_to_polarity = distribute_polarity(polarity_of_organ, organ_to_subsector, organ_to_company)
    
    output_format = make_news_output_format(subsector_to_polarity, company_to_polarity, df)
    
    return output_format


# In[32]:


# In[46]:


def make_news_output_format(subsector_to_polarity, company_to_polarity, df):
    news_output = {}
    news_output['Params'] = list()
    for key, value in subsector_to_polarity.items():
        item_type = 'Commodity'
        #item_symbol = df[df['cleaned_subsectors'] == key]['Symbol'].values[0]
        item_symbol = key
        item_sentiment = value
        temp = dict()
        temp['label'] =item_type
        temp['symbol'] = item_symbol
        temp['sentiment'] = item_sentiment
        news_output['Params'].append(temp)
    
    for key, value in company_to_polarity.items():
        item_type = 'Organisation'
        item_symbol = df[df['cleaned_companies'] == key]['Symbol'].values[0] + '.NS'
        item_sentiment = value
        temp = dict()
        temp['label'] =item_type
        temp['symbol'] = item_symbol
        temp['sentiment'] = item_sentiment
        if DEBUG: temp['name'] = df[df['cleaned_companies'] == key]['Company'].values[0]
        #print(temp)
        news_output['Params'].append(temp)
        
    return news_output

#### output_data = find_subsector_company_sentiment_json_format(document)  
#print(output_data)

# In[44]:

# document1 = """Reliance Industries-Rights Entitlement share price traded sharply higher on May 27, with more than 78 lakh shares volume.It touched an intraday high of Rs 209.90 and a low of Rs 163.75 after opening the session at Rs 177 on the National Stock Exchange.At 1439 hours, it was trading at Rs 197, up 8.48 percent over the previous day's close of Rs 181.60.The trading in RIL Rights Entitlement will continue till May 29, so that as per T+2 settlement, the eligibility for partly paid-up rights shares will be decided on the closing data of June 2.The person eligible for those shares on June 2 will have to pay the first instalment of Rs 314.25 on June 3, the closing date for the rights issue.After the finalisation, the partly paid-up rights shares will be allotted and credited to eligible shareholders by June 11 and the same will be listed on bourses on June 12.Mukesh Ambani-owned Reliance Industries plans to raise Rs 53,125 crore through the rights issue at a price of Rs 1,257 per share.The second instalment of Rs 314.25 will be due in May 2021 and the final instalment of Rs 628.50 in November 2021.This is the biggest ever rights issue by an Indian company, and the first by Reliance Industries in 30 years.Ahead of the closing of Rights Entitlement, RIL has already raised Rs 78,562 crore by selling over 17 percent stake in Jio Platforms over the last one month to Facebook, Silver Lake, Vista, General Atlantic and KKR.Disclaimer: Reliance Industries Ltd. is the sole beneficiary of Independent Media Trust which controls Network18 Media & Investments Ltd..reckoner_bx{ background-color: #F0F0F0; padding: 20px; font: 400 16px/22px 'Noto Serif',arial; border-radius: 5px; margin-bottom: 0px;}.reckoner_bx .rek_title{font: 700 18px/25px 'Fira Sans',arial; color: #0155A0; margin-bottom: 7px; text-transform: uppercase;}.reckoner_bx .btn_reck{border-radius: 20px; background-color: #135B9D; display: inline-block; font: 700 14px/19px 'Noto Serif',arial; padding: 8px 25px; color: #fff !important; text-decoration: none !important;}.reckoner_bx .rek_btnbx{ margin-top: 10px; }.reckoner_bx .bldcls{font-weight: bold;}Coronavirus Essential | Lockdown might be extended to June 15 as cases cross 1.5 lakh; India number may peak in July, experts say Copyright \u00a9 e-Eighteen.com Ltd. All rights reserved. Reproduction of news articles, photos, videos or any other content in whole or in part in any form \r\n        or medium without express writtern permission of moneycontrol.com is prohibited. Copyright \u00a9 e-Eighteen.com Ltd All rights resderved. Reproduction of news articles, photos, videos or any other content in whole or in part in any form or medium without express writtern permission of moneycontrol.com is prohibited."""
# document2 = document2
# document3 = """Sun Pharmaceutical Industries on May 27 reported a consolidated profit of Rs 399.8 crore for quarter ended March 2020, declining 37.1 percent YoY due to one-time loss of Rs 260.6 crore.

# As exceptional item, an amount of Rs 104.28 crore, including interest, has been charged in the P&L account with respect to Central excise refund claims case. Also, the company has made provision for Rs 156.36 crore for the settlement with the US Department of Justice to resolve the sales, marketing and promotion of two of its products - Levulan and Blu-u .

# Consolidated revenue during the quarter rose 14.3 percent year-on-year to Rs 8,184.9 crore, which was ahead of the average of estimates of analysts polled by CNBC-TV18 which pegged at Rs 8,015.8 crore.

# On the operating front, consolidated earnings before interest, tax, depreciation and amortisation (EBITDA) grew by 34 percent to Rs 1,363 crore and margin expanded by 250 bps to 16.7 percent, which also missed analysts expectations.

# RELATED NEWS
# Sagar Cements Q4 profit down 94% to Rs 1.18 crore
# Jubilant Life Sciences posts Rs 260.49 crore profit for March quarter
# Muthoot Capital Q4 profit falls 40% to Rs 14 crore on auto sector woes
# A CNBC-TV18 poll estimates for EBITDA were at Rs 1,652.3 crore and margin at 20.6 percent for the quarter.

# Sun Pharma reported tax expenses for the quarter at Rs 83.09 crore against tax write-back of Rs 28.8 crore in same period previous fiscal.

# Other income during Q4FY20 declined sharply to Rs 102.23 crore, compared to Rs 281.53 crore in Q4FY19.

# For the full year (FY20), Sun Pharma has reported a 41.3 percent growth in profit at Rs 3,764.93 crore and 13 percent increase in revenue at Rs 32,837.5 crore compared to previous year.

# Its subsidiary Taro Pharmaceuticals declared profit for the March quarter 2020 at $54.2 million, declining 7.2 percent compared to March quarter 2019.

# Its net sales also declined during the quarter, down by $5 million to $174.9 million YoY.

# "Despite the leading market position of many of our products, we continue to face a challenging US generic market. In the short-term, even as we commercialize recently approved products, we expect operations and profitability to be temporarily impacted as a result of the COVID-19 pandemic," Uday Baldota, Taro's CEO said."""
#preprocess_data()
# print(find_subsector_company_sentiment_json_format(document3))

# %%


# %%
