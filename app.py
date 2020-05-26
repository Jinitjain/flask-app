# using flask_restful
import json
import random 
from flask import Flask, jsonify, request 
from flask_restful import Resource, Api
from scrapping_modules.news_scrapper import extract_article 

# creating the flask app 
app = Flask(__name__) 
# creating an API object 
api = Api(app) 
  
# extract relation between different entities 
class extract_relation(Resource): 

    def post(self):
        data = request.get_json()
        # news_json = json.loads(data)
        response = {"relations" : []}
        for news_article in data["news"]:
            name = news_article["name"]
            url = news_article["url"]
            print(url)

            sentiment = {}
            try:
                article = extract_article(url)
                # add Jayant's function call
                sentiment = { "A" : random.random(),
                              "B" : random.random() }

            except Exception:
                pass

            response["relations"].append({"name": name,
                                          "url": url,
                                          "sentiment": sentiment})
            

        return jsonify(response) 
  
# adding the defined resources along with their corresponding urls 
api.add_resource(extract_relation, '/extract-relation') 
  
# driver function 
if __name__ == '__main__': 
    app.run(debug = True)