#!/usr/bin/env python
# coding: utf-8

# # Azure hackathon

# ## Import

# In[1]:


import nltk
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
try:
    nltk.data.find('corpora/stopwords.zip')
    #nltk.data.find('tokenizers/punkt.zip')
    nltk.data.find('taggers/averaged_perceptron_tagger.zip')
    #nltk.data.find('chunkers/maxent_ne_chunker.zip')
    nltk.data.find('corpora/words.zip')
except LookupError:
    nltk.download('stopwords')
    #nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    #nltk.download('maxent_ne_chunker')
    nltk.download('words') 
from nltk.corpus import stopwords
stop = stopwords.words('english')

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


# # CNBC news article's scrapper
# import requests 
# from bs4 import BeautifulSoup

# def extract_article (url):
#     # request for article web page
#     req = requests.get(url) 

#     # extract html data using the 'lxml' parser  
#     soup = BeautifulSoup(req.content, 'lxml')

#     # extract news article's headline  
#     headline = soup.find('h1', class_="ArticleHeader-headline").text

#     # extract news description
#     description = ''
#     news_desc = soup.find_all('div', class_="group")
#     for desc in news_desc:
#         description = description + desc.text

#     return headline, description

# document = extract_article("https://www.cnbc.com/2020/05/22/coronavirus-goldman-sachs-on-india-growth-gdp-forecast.html")


# In[4]:



# ## Process Document

# In[6]:


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
    
    print("Organizations:\n",list(organization.keys()))
    print("Location:\n",list(location.keys()))
    
    organization.update(location)
                            
    return organization, location
    # return names, organization, location, address, other


# ## Setting up one time company and subsector loading

# ### Utilities function

# In[8]:


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

def clean_subsectors(subsector):
    return subsector.str.lower()
    


# ### New

# In[11]:


def preprocess_data():
    global df, cleaned_companies, cleaned_subsectors
    df = pd.read_csv('finalBetaDB.csv')
    df.rename(columns={' Sector': 'Sector', ' Sub-sector':'Subsector'}, inplace=True)
    df = df.replace(np.nan, '', regex=True)
    
    cols = ['Sector', 'Subsector']
    for col in cols:
        df[col] = df[col].str.strip()
        
    cleaned_companies = clean_companies(df['Company'])
    cleaned_subsectors = clean_subsectors(df['Subsector'])

    df['cleaned_companies'] = cleaned_companies
    df['cleaned_subsectors'] = cleaned_subsectors
    
    return df, cleaned_companies, cleaned_subsectors


# ## Finding sentiment

# In[89]:


def find_subsectors(organizations, cleaned_subsectors):
    """Find subsectors in organization"""
    
    organ_to_subsector = {}
    for subsector in cleaned_subsectors:
        for organization in organizations.keys():
            for split_organization in organization.split(' '):
                if split_organization in subsector.split(' '):
                    if split_organization not in organ_to_subsector:
                        organ_to_subsector[split_organization] = list()
                    organ_to_subsector[split_organization].append(subsector)
    return organ_to_subsector

def find_companies(organizations, cleaned_companies):
    
    organ_to_company = {}
    for company in cleaned_companies:
        for organization in organizations.keys():
            #print("############", organization)
            for split_organization in organization.split(' '):
                #print("+++++++++++++" , split_organization)
                if split_organization in company.split(' '):
                    if split_organization not in organ_to_company:
                        organ_to_company[split_organization] = list()
                    organ_to_company[split_organization].append(company)
    #print("organ to company", organ_to_company)
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
        for split_organization in organization.split(' '):
            if split_organization in (organ_to_subsector.keys() or organ_to_company.keys()):
                dict_organizations[organization] = organizations[organization]
    #print("Dict organizations ", dict_organizations)
    return dict_organizations


# In[90]:


def find_sentiment_of_context(document, organ_to_sentenceid, total_sentences):
    list_of_sentences = document.split('.')
    polarity_of_organ = {}
    #print(list_of_sentences)
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

#polarity_of_organ = find_sentiment_of_context(document, organ_to_sentenceid, total_sentences)


# In[91]:


def find_max_or_min_value(arr):
    maxi = max(arr)
    mini = min(arr)
    
    return maxi if abs(maxi) > abs(mini) else mini

def distribute_polarity(polarity_of_organ, organ_to_subsector, organ_to_company):
    subsector_to_polarity = {}
    company_to_polarity = {}
    for organization in polarity_of_organ:
        ### TODO: Make use of organization dict to consider the impact score
        value = find_max_or_min_value(polarity_of_organ[organization])
        for split_organization in organization.split(' '):
            if split_organization in organ_to_subsector.keys():
                for subsector in organ_to_subsector[split_organization]:
                    if subsector in subsector_to_polarity:
                        value = subsector_to_polarity[subsector] if abs(subsector_to_polarity[subsector]) > abs(value) else value
                    subsector_to_polarity[subsector] = value

            elif split_organization in organ_to_company.keys():
                for company in organ_to_company[split_organization]:
                    if company in company_to_polarity:
                        value = company_to_polarity[company] if abs(company_to_polarity[company]) > abs(value) else value
                    company_to_polarity[company] = value
        
    # print(subsector_to_polarity)
    # print(company_to_polarity)
    return subsector_to_polarity, company_to_polarity

#subsector_to_polarity, company_to_polarity = distribute_polarity(polarity_of_organ, organ_to_subsector, organ_to_company)


# In[92]:


def make_news_output_format(subsector_to_polarity, company_to_polarity, df):
    news_output = {}
    news_output['Params'] = list()
    for key, value in subsector_to_polarity.items():
        item_type = 'Commodity'
        item_symbol = df[df['cleaned_subsectors'] == key]['Symbol'].values[0]
        item_sentiment = value
        temp = dict()
        temp['label'] =item_type
        temp['symbol'] = item_symbol
        temp['sentiment'] = item_sentiment
        news_output['Params'].append(temp)
    
    for key, value in company_to_polarity.items():
        item_type = 'Organization'
        item_symbol = df[df['cleaned_companies'] == key]['Symbol'].values[0] + '.NS'
        item_sentiment = value
        temp = dict()
        temp['label'] =item_type
        temp['symbol'] = item_symbol
        temp['sentiment'] = item_sentiment
        news_output['Params'].append(temp)
        
    return news_output


# ## One stop evaluater

# In[94]:


def find_subsector_company_sentiment_json_format(document):
    #document = " ".join(document)
    organization, _ = nltk_eval(document)
    print(organization)
    organ_to_subsector = find_subsectors(organization, cleaned_subsectors)
    organ_to_company = find_companies(organization, cleaned_companies)
    
    #print("Organ subsectors", organ_to_subsector)
    #print("Organ campanies", organ_to_company)
    
    updated_organizatation = remove_other_organization(organization, organ_to_subsector, organ_to_company)
    organ_to_sentenceid, total_sentences = find_organ_context(document, updated_organizatation)
    
    
    
    #print("Sentence id", organ_to_sentenceid)
    #print(organ_to_subsector)
    
    polarity_of_organ = find_sentiment_of_context(document, organ_to_sentenceid, total_sentences)
    
    #print("polarity of organisation", polarity_of_organ)
    subsector_to_polarity, company_to_polarity = distribute_polarity(polarity_of_organ, organ_to_subsector, organ_to_company)
    
    output_format = make_news_output_format(subsector_to_polarity, company_to_polarity, df)
    
    return output_format


# In[95]:


# document = """Reliance Industries-Rights Entitlement share price traded sharply higher on May 27, with more than 78 lakh shares volume.It touched an intraday high of Rs 
# 209.90 and a low of Rs 163.75 after opening the session at Rs 177 on the National Stock Exchange.At 1439 hours, it was trading at Rs 197, up 8.48 percent over the previous day's close of Rs 181.60.The trading in RIL Rights Entitlement will continue till May 29, so that as per T+2 settlement, the eligibility for partly paid-up rights shares will be decided on the closing data of June 2.The person eligible for those shares on June 2 will have to pay the first instalment of Rs 314.25 on June 3, the closing date for the rights issue.After the finalisation, the partly paid-up rights shares will be allotted and credited to eligible shareholders by June 11 and the same will be listed on bourses on June 12.Mukesh Ambani-owned Reliance Industries plans to raise Rs 53,125 crore through the rights issue at a price of Rs 1,257 per share.The second instalment of Rs 314.25 will be due in May 2021 and the final instalment of Rs 628.50 in November 2021.This is the biggest ever rights issue by an Indian company, and the first by Reliance Industries in 30 years.Ahead of the closing of Rights Entitlement, RIL has already raised Rs 78,562 crore by selling over 17 percent stake in Jio Platforms over the last one month to Facebook, Silver Lake, Vista, General Atlantic and KKR.Disclaimer: Reliance Industries Ltd. is the sole beneficiary of Independent Media Trust which controls Network18 Media & Investments Ltd..reckoner_bx{ background-color: #F0F0F0; padding: 20px; font: 400 16px/22px 'Noto Serif',arial; border-radius: 5px; margin-bottom: 0px;}.reckoner_bx .rek_title{font: 700 18px/25px 'Fira Sans',arial; color: #0155A0; margin-bottom: 7px; text-transform: uppercase;}.reckoner_bx .btn_reck{border-radius: 20px; background-color: #135B9D; display: inline-block; font: 700 14px/19px 'Noto Serif',arial; padding: 8px 25px; color: #fff !important; text-decoration: none !important;}.reckoner_bx .rek_btnbx{ margin-top: 10px; }.reckoner_bx .bldcls{font-weight: bold;}Coronavirus Essential | Lockdown might be extended to June 15 as cases cross 1.5 lakh; India number may peak in July, experts say Copyright © e-Eighteen.com Ltd. All rights reserved. Reproduction of news articles, photos, videos or any other content in whole or in part in any form
#         or medium without express writtern permission of moneycontrol.com is prohibited. Copyright © e-Eighteen.com Ltd All rights resderved. Reproduction 
# of news articles, photos, videos or any other content in whole or in part in any form or medium without express writtern permission of moneycontrol.com is 
# prohibited."""
# preprocess_data()
# find_subsector_company_sentiment_json_format(document)

