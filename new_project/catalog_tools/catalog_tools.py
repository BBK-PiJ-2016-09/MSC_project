import logging
import re
import sys
import os
import json
import numpy

from collections import Counter
from traceback import format_exception
from pyCGA.opencgaconfig import ConfigClient
from pyCGA.opencgarestclients import OpenCGAClient


class CatalogAggregator:
    def __init__(self, raw_data_dir, output_dir, numeric_chunks=50, non_numeric_chunks=10):
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.aggregated = {}
        self.numeric_chunks = numeric_chunks
        self.non_numeric_chunks = non_numeric_chunks
        self.output_dir = output_dir
        self.output_file = output_dir.rstrip('/') + '/aggregated.json'

    def aggregate_data(self):
        for list in os.walk(self.raw_data_dir):
            for file in list[2]:
                self.aggregate_file(self.raw_data_dir + file)
        with open(self.output_file, 'w') as fp:
            json.dump(self.aggregated, fp)

    def is_numeric(self, some_string):
        try:
            float(some_string)
            return True
        except ValueError:
            return False

    def parse_header(self, file_header):
        file_name = '/'.join(file_header.split(';')[0:-1]).replace(
            'by_name/SAMPLE_ID/DELIVERY_ID/Assembly', 'delivery_dir')
        file_attribute = '.'.join(file_header.split(';')[-1].split(':'))
        return file_name, file_attribute

    def aggregate_file(self, file_path):
        filehandler = open(file_path, 'r')
        file_header = filehandler.readline()
        lines = []
        original_file_name, original_attribute = self.parse_header(file_header)
        for line in filehandler:
            if line:
                line = line.rstrip('\n')
                lines.append(line)
        if all(self.is_numeric(line) for line in lines):
            return self.aggregate_numeric_list(original_file_name, original_attribute, map(float, lines))
        return self.aggregate_non_numeric_list(original_file_name, original_attribute, lines)

    def aggregate_numeric_list(self, original_file_name, original_attribute, some_list):
        if original_file_name not in self.aggregated:
            self.aggregated[original_file_name] = {}

        if not some_list:
            self.aggregated[original_file_name][original_attribute] = {'min': 0,
                                                                       'max': 0,
                                                                       'values': []}
        else:
            min_val = min(some_list)
            max_val = max(some_list)
            increment = (max_val - min_val) / self.numeric_chunks
            if increment == 0 or not some_list:
                self.aggregated[original_file_name][original_attribute] = {'min': 0,
                                                                       'max': 0,
                                                                       'values': []}
            else:
                intervals = numpy.arange(min_val, max_val, increment)
                val_list = [0] * self.numeric_chunks
                i = 0
                while i < len(some_list):
                    j = 0
                    while j < len(val_list):
                        if intervals[j] > some_list[i]:
                            val_list[j] += 1
                            j = len(val_list)
                        j += 1
                    i += 1
                self.aggregated[original_file_name][original_attribute] = \
                    {'min': min_val, 'max': max_val, 'values': val_list}

    def aggregate_non_numeric_list(self, original_file_name, original_attribute, some_list):
        counted = Counter(some_list)
        if original_file_name not in self.aggregated:
            self.aggregated[original_file_name] = {}

        self.aggregated[original_file_name][original_attribute] = \
            dict(sorted(counted.iteritems(), key=operator.itemgetter(1), reverse=True)[:10])




class CatalogException(Exception):
    pass


class CatalogCrawler:
    def __init__(self, connector, summary_dir, logger, sample_limit=100000):
        self.connector = connector
        self.aggregated_paths = {}
        self.summary_dir = summary_dir
        self.logger = logger
        self.blacklisted_attributes = ['path', 'uri']
        self.blacklisted_file_endings = ['log']
        self.open_files = {}
        self.files_2_code = {}
        self.last_file = 1
        self.regexs = self.get_regexs()
        self.sample_limit = sample_limit

    def summarize(self):
        self.logger.info("Getting samples")
        self.samples = self.connector.get_all_samples(self.sample_limit)
        self.logger.info("Got {} samples, proceeding...".format(len(self.samples)))
        print("Got {} samples, proceeding...".format(len(self.samples)))

        for sample in self.samples:
            self.logger.info("Getting files for sample {}".format(sample['name']))
            print("Getting files for sample {}".format(sample['name']))
            files = self.connector.file_get_all_associated_with_sample(sample['name'])
            self.logger.info("Got {} Files for sample {}".format(len(files), sample['name']))
            self.process_files(files)

    def process_files(self, files):
        for file in files:
            if self.transform_path(file['path']) not in self.aggregated_paths:
                self.aggregated_paths[self.transform_path(file['path'])] = {}
            self.add_file_to_aggregated(file)

    def should_blacklist_file(self, file):
        for forbiden_ending in self.blacklisted_file_endings:
            if file['path'].endswith(forbiden_ending):
                return True
        return False

    def add_file_to_aggregated(self, file):
        if not self.should_blacklist_file(file):
            for attr in file:
                path = self.join_attr_path(":", self.transform_path(file['path']), attr)
                self.register_attr(attr, file[attr], path)

    @staticmethod
    def join_attr_path(sep, path, attr):
        return sep.join([path, attr])

    def register_attr(self, attr, attr_value, path):
        path = self.transform_path(path)
        if isinstance(attr_value, dict):
            for new_attr in attr_value:
                new_path = self.join_attr_path(":", path, new_attr)
                self.register_attr(new_attr, attr_value[new_attr], new_path)
        elif isinstance(attr_value, list):
            for subval in attr_value:
                self.register_attr(attr, subval, path)
        else:
            if attr not in self.blacklisted_attributes:
                self.send_to_file(path, attr_value)

    def get_physical_file_path(self, path):
        if path not in self.files_2_code:
            self.files_2_code[path] = self.last_file
            self.last_file = self.last_file + 1
            self.open_files[self.files_2_code[path]] = open(self.summary_dir + "/"
                                                            + str(self.files_2_code[path]), 'w')
            self.open_files[self.files_2_code[path]].write(str(path) + "\n\n")
            self.open_files[self.files_2_code[path]].close()
        return self.summary_dir + "/" + str(self.files_2_code[path])

    def send_to_file(self, path, attr_value):
        fh = open(self.get_physical_file_path(str(path)), 'a')
        fh.write(str(attr_value) + "\n")
        fh.close()

    def transform_path(self, path):
        return ";".join(map(self.exchange_regexs, path.split("/")))

    def exchange_regexs(self, string):
        for key in self.regexs.keys():
            string = re.sub(self.regexs[key]['regex'], self.regexs[key]['sub'], string)
        if isinstance(string, int):
            return 'EXECUTION'
        return string

    def get_regexs(self):
        return {
            'sample': {'regex': r'LP\d{7}-DNA_[A-H](0[1-9]|1[0-2])',
                       'sub': 'SAMPLE_ID'},
            'normal_sample': {'regex': r'NormalLP\d{7}-DNA_[A-H](0[1-9]|1[0-2])',
                       'sub': 'NORMAL_SAMPLE_ID'},
            'cancer_sample': {'regex': r'CancerLP\d{7}-DNA_[A-H](0[1-9]|1[0-2])',
                              'sub': 'CANCER_SAMPLE_ID'},
            'delivery': {'regex': r'^((?:RAREP|RARET|CANCP|CANCT)[0-9]{5}|(?:HX|CF|CH|OX|VD|BE|ED)[0-9]{8}|V_V[0-9]{11}|[0-9]{10})$$',
                         'sub': 'DELIVERY_ID'},
            'genotype_array_idat_red': {'regex': r'[0-9]{12}[_]{1}[A-z a-z]{1}[0-9]{2}[A-z a-z]{1}[0-9]{2}[_]RED',
                                        'sub': 'GENOTYPING_IDAT_RED'},
            'genotype_array_idat_green': {'regex': r'[0-9]{12}[_]{1}[A-z a-z]{1}[0-9]{2}[A-z a-z]{1}[0-9]{2}[_]GRN',
                                          'sub': 'GENOTYPING_IDAT_GREEN'},
            'genotype_array_gtc_regex': {'regex': r'[0-9]{12}[_]{1}[A-z a-z]{1}[0-9]{2}[A-z a-z]{1}[0-9]{2}',
                                         'sub': 'GENOTYPING_GTC'}
        }


class CatalogOpenCGA(object):
    """
    Provides functionality for Bertha-OpenCGA integration. This class takes the
    assumptions made by Bertha about OpenCGA and makes the translation. Example:
    An sample ID in Bertha is an LP Number, while in OpenCGA it's a sample name.
    This class provides a reusable set of Bertha-specific methods to interact
    with OpenCGA.
    The usage pattern is::

        with BerthaOpenCGA(...) as connector:
            do_work

    The connector log in with __enter__ and logout in __exit__
    """

    def __init__(self, study_id, instance_url, username, password, paths_root,
                 ignore_dirs, retry_max_attemps=None,
                 min_retry_seconds=None, max_retry_seconds=None,
                 logger=None, exclude_dirs=None):
        """
        :param study_id: integer study id or study alias
        :param instance_url: example 'localhost:8080/opencga'
        :param username: OpenCGA username (string) for login
        :param password: OpenCGA password (string) for login
        :param paths_root: path root to be stripped from OpenCGA paths, e.g. '/genomes'
        :param ignore_dirs: if True, directories will not be registered
        :param retry_max_attemps: optional: a max number of attempts when retrying.
        No retries by default. If specified, then min_retry_seconds and max_retry_seconds
        must also be supplied.
        :param min_retry_seconds: retry wait time starts at min_retry_seconds and doubles
         after each failure, capped at max_retry_seconds
        :param max_retry_seconds: see min_retry_seconds
        :param logger: optional - if supplied, the connector will log messages in this logger
        :param exclude_dirs: a list of strings - any file which path starts with one of the
        strings in this list will not be linked to catalog.
        """
        self.study_id = study_id
        self.instance_url = instance_url
        self.username = username
        self.password = password
        self.paths_root = paths_root
        self.ignore_dirs = ignore_dirs

        if retry_max_attemps:
            assert min_retry_seconds and max_retry_seconds, \
                "If retry_max_attemps is supplied, min_retry_seconds and max_retry_seconds must be supplied"
        self.retry_max_attempts = retry_max_attemps
        self.min_retry_seconds = min_retry_seconds
        self.max_retry_seconds = max_retry_seconds

        self.logger = logger
        self.exclude_dirs = exclude_dirs or []

        # a cache mapping sample lp numbers to OpenCGA sample IDs to reduce number of requests
        self.sample_id_cache = {}
        config = ConfigClient.get_basic_config_dict(self.instance_url)
        config['retry'] = dict(
            max_attempts=self.retry_max_attempts,
            min_retry_seconds=self.min_retry_seconds,
            max_retry_seconds=self.max_retry_seconds
        )
        self.pycga = OpenCGAClient(config, self.username, self.password, on_retry=self._on_pycga_retry)
        self._log("Connected and logged in")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pycga.logout()
        self.pycga = None

    # noinspection PyUnusedLocal
    def _on_pycga_retry(self, pycga, exc_type, exc_val, exc_tb, call):
        exc_str = ''.join(format_exception(exc_type, exc_val, exc_tb))
        msg = """pyCGA call with
                 {call}
                 failed with:
                 {failure}
                 Retrying...""".format(call=call, failure=exc_str)
        self._log(msg, logging.ERROR)

    def _log(self, msg, level=logging.INFO, *args, **kwargs):
        if self.logger:
            self.logger.log(level, msg, *args, **kwargs)
        return None

    def _get_file_id(self, path):
        """
        Gets the id of a file given its path
        :param path: string - absolute or OpenCGA path
        :return: an integer - the internal OpenCGA file ID. Raises CatalogException if not
        found or not unique.
        """
        results = self.pycga.files.search(self.study_id, path=self.to_catalog_path(path),
                                          include='id').get()
        if len(results) == 1:
            return results[0]['id']  # Ok
        elif len(results) > 1:
            raise CatalogException("More than one file found for path {} in OpenCGA. Ids: {}"
                                         .format(path, ','.join([str(result['id']) for result in results])))
        else:
            raise CatalogException("File {} not found in OpenCGA".format(path))

    def get_all_samples(self, limit=1000000):
        samples = self.pycga.samples.search(study=self.study_id, name="~LP", limit=limit)
        return samples[0]

    def to_catalog_path(self, path):
        """
        If path starts with self.paths_root, strips that root and any leading '/' that
        might be left. Otherwise, no changes.
        :return: the transformed path. Examples, assuming self.paths_root == '/genomes':
            /genomes/dir/file.txt --> 'dir/file.txt'
            genomes/dir/file.txt --> 'genomes/dir/file.txt'
            /dir/file.txt --> '/dir/file.txt'
        """
        if path.startswith(self.paths_root):
            return path[len(self.paths_root):].lstrip('/')
            # left-strip paths_root (don't use lstrip for that!) and then any remaining leading '/'
        else:
            return path

    @staticmethod
    def create_tracking_entry(process_name, process_version,
                              process_time, **kwargs):
        entry = {
            'name': process_name,
            'version': process_version,
            'processDate': "{:%Y%m%d%H%M%S}".format(process_time)  # this is the opencga datetime format
        }
        entry.update(kwargs)
        return entry

    @staticmethod
    def contains_tracking_item(processes, item, ignore_lookup=None):
        if ignore_lookup is None:
            ignore_lookup = set()
        else:
            ignore_lookup = set(ignore_lookup)  # ensure it's a set
        ignore_lookup.add('processDate')

        # compose a json path expression, e.g. '$[?name=="Bertha" & version=="1.3.2" & runId==1]'
        filter_parts = ["{}=={}".format(key, jsonpath.expr_str(value))
                        for key, value in item.iteritems()
                        if key not in ignore_lookup]
        pattern = "$[?{filter}]".format(filter=" & ".join(filter_parts))
        return bool(jsonpath.match(pattern, processes))

    def sample_search(self, lp_number, **options):
        """
        Get a sample from the current study in openCGA given its LP number. If not found, then returns None. If
        there is more than 1 sample with that LP number, then raises an CatalogException
        :param lp_number: LP number of the sample, which is store in the 'name' field in OpenCGA. A string.
        :return: A dictionary representing a sample, as returned by OpenCGA, or None if not found.
        """
        results = self.pycga.samples.search(study=self.study_id, name=lp_number, **options).get()
        if len(results) == 1:
            sample = results[0]
            self.sample_id_cache[lp_number] = sample['id']
            return sample  # Ok
        elif len(results) > 1:
            raise CatalogException("More than one sample found for LP Number {}. OpenCGA sample IDs: {}"
                                         .format(lp_number, ','.join([str(result['id']) for result in results])))
        else:
            return None  # Not found

    def sample_get(self, lp_number, **options):
        """Gets a sample by LP number or raises a CatalogException if not found
        :param lp_number: string, sample lp number, matched against "name" field
        :param options: additional query options, e.g. include=stats
        :return: A sample dictionary as returned by OpenCGA, or CatalogException
        if not found
        """
        sample = self.sample_search(lp_number, **options)
        if sample is None:
            raise CatalogException("Sample {} not found in OpenCGA".format(lp_number))
        return sample

    def sample_id_get(self, lp_number):
        """If a sample is in the sample cache, returns its id from cache,
        otherwise, looks it up in OpenCGA. If not found, raises a
        CatalogException
        :param lp_number: string, sample lp number, matched against "name" field
        :return: integer: OpenCGA sample ID, or CatalogException if not found.
        """
        sample_id = self.sample_id_cache.get(lp_number)
        if sample_id is not None:
            return sample_id
        else:
            sample = self.sample_search(lp_number)
            if not sample:
                raise CatalogException("Sample {} not found in OpenCGA".format(lp_number))
            return sample['id']


    def file_get(self, file_id, **options):
        """Gets a file identified by its OpenCGA file ID.
        :return: File dictionary as returned by OpenCGA. If not found, raises CatalogException
        """
        catalog_file = self.pycga.files.search(study=self.study_id, id=file_id, **options).get()
        if not catalog_file:
            raise CatalogException("File {} not found in OpenCGA".format(file_id))
        else:
            return catalog_file[0]

    def file_get_all_associated_with_sample(self, sample):
        files = self.pycga.files.search(study=self.study_id, sampleIds=sample, bioformat='ALIGNMENT')
        return files[0]


def is_opencga_bypassed():
    return False


def get_default_opencga_connector(study_id):
    """
    Creates a BerthaOpenCGA with default bertha configuration values (url, username
    password, genomes root from environment variables)
    :param study_id: Study ID to be used in the connector
    :param logger: optional, if supplied then the connector will log on this logger,
    otherwise the default module logger will be used
    :param exclude_dirs: list of directory paths (strings) to be excluded from registration
    :return:
    """

    return CatalogOpenCGA(study_id,
                         get_default_opencga_url(),
                         get_default_opencga_user(),
                         get_default_opencga_password(),
                         get_default_root_path(),
                         ignore_dirs=True,
                         retry_max_attemps=sys.maxint,
                         min_retry_seconds=1,
                         max_retry_seconds=1,
                         logger=None,
                         exclude_dirs=None)


def get_default_opencga_url():
    return 'http://bio-test-opencga-catalog-05.gel.zone:8080/opencga'


def get_default_opencga_user():
    return 'bertha'


def get_default_opencga_password():
    return 'jYZ1r2sPqMl6'


def get_default_root_path():
    return '/genomes'


if __name__ == "__main__":
    logger = logging.getLogger('opencga')
    connector = get_default_opencga_connector('UN38')
    CatalogCrawler(connector, '/home/eserra/MSC_Project/catalog_summary/new_project/data/raw', logger).summarize()
    aggregate('/home/eserra/MSC_Project/catalog_summary/new_project/data/raw', '/home/eserra/MSC_Project/catalog_summary/new_project/data/aggregated')
