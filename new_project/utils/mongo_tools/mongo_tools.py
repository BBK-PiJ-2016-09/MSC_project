from pymongo import MongoClient


class MongoConnector:

    def __init__(self, config):
        self.host = config.host
        self.port = config.port
        self.connector = MongoClient(self.host, self.port)
        self.db_name = config.db_name
        self.db = self.connector.get_database(self.db_name)
        self.collection = MongoCollection(config.collection_name, self.db)


class Doc:
    def __init__(self, doc):
        self.doc = doc
        self.name = doc['name']
        self.id = doc['_id']


class MongoCollection:
    def __init__(self, name, db):
        self.name = name
        self.db = db

    @property
    def collection(self):
        return self.db.get_collection(self.name)

    @property
    def count(self):
        return self.collection.count()

    @property
    def scan_collection(self):
        for doc in self.collection.find():
            yield Doc(doc)
