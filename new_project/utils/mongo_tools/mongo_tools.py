from pymongo import MongoClient
import json

class MongoConnector:

    def __init__(self, host, port, db_name, collection_name):
        self.host = host
        self.port = port
        self.connector = MongoClient(self.host, self.port)
        self.db_name = db_name
        self.db = self.connector.get_database(self.db_name)
        self.collection = self.db.get_collection(collection_name)

    def insert_aggregated(self, aggregated):
        print self.db.collection_names()
        for file_name in aggregated.keys():
            if not self.collection.find_one({'name': file_name}):
                self.collection.insert({'name': file_name})
            for attr in aggregated[file_name].keys():
                #print "ATTR" + attr
                #print self.collection.collection.find_one({'name': file_name})
                self.collection.find_one_and_update({'name': file_name},
                                                               {'$set': {attr: aggregated[file_name][attr]}})

class Doc:
    def __init__(self, doc):
        self.doc = doc
        self.name = doc['name']
        self.id = doc['_id']
