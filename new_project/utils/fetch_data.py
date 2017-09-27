from catalog_tools.catalog_tools import *
from mongo_tools.mongo_tools import *
import logging

if __name__ == "__main__":
    logger = logging.getLogger('opencga')
    connector = get_default_opencga_connector('UN38')
    mongo_connector = MongoConnector('localhost', 27017, 'datavisualization', 'files')
    CatalogCrawler(connector, '/home/eserra/PROJECTREPO/new_project/data/raw', logger).summarize()
    CatalogAggregator('/home/eserra/PROJECTREPO/new_project/data/raw/', mongo_connector).aggregate_data()
