from pymongo import MongoClient
import json

mongo_client = MongoClient()
db = mongo_client.yelp
coll = db.checkin

from bson import json_util
data = json_util.loads('Data/dataset/checckin.json')
coll.insert_many(data)
