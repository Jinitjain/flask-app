# using flask_restful
import json
import random
import Analyze_Sentiment as analyze_sentiment
from flask import Flask, jsonify, request 
from flask_restful import Resource, Api
from flask_cors import CORS
from scrapping_modules.news_scrapper import extract_article 

# creating the flask app 
app = Flask(__name__) 
# creating an API object 
api = Api(app)
CORS(app) 
  
# extract relation between different entities 
class extract_relation(Resource): 

    def post(self):
        data = request.get_json()
        response = {"relations" : []}
        analyze_sentiment.preprocess_data()

        for news_article in data["news"]:
            name = news_article["name"]
            url = news_article["url"]
            print(url)

            sentiment = {}
            try:
                article = extract_article(url)
                # print(article)
                sentiment = analyze_sentiment.find_subsector_company_sentiment_json_format(article)
                # print(sentiment)
            except Exception:
                pass
            
            response["relations"].append({"name": name,
                                          "url": url,
                                          "sentiment": sentiment,
                                          "article": article})
            

        return jsonify(response) 
  
# adding the defined resources along with their corresponding urls 
api.add_resource(extract_relation, '/extract-relation') 
  
# driver function 
if __name__ == '__main__': 
    app.run(debug = True)