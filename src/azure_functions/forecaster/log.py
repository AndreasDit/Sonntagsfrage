import logging
import sys
from datetime import datetime
from logging import FileHandler

import google.cloud.logging
import yaml
from google.cloud.logging.handlers import CloudLoggingHandler

import code_configs

file = open(code_configs.CONFIG_PATH, 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

FORMATTER = logging.Formatter(cfg['logging']['log_format'])
LOG_LEVEL = cfg['logging']['log_level']

LOG_FILE_PATH = cfg['logging']['log_file_path']
LOG_FILE_NAME = cfg['logging']['log_file_name']
today = datetime.today()
log_file_name = LOG_FILE_NAME.replace('<YYYYmmddHMS>', today.strftime('%Y%m%d%H%M%S'))


def get_console_handler():
    """
    Create a console_handler for logging

    :return: console_handler
    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    """
    Create a file_handler for logging

    :return: file_handler
    """
    file_handler = FileHandler(LOG_FILE_PATH + log_file_name)
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_google_handler():
    """
    Create a Google Cloud Logging handler for logging

    :return: cloud_handler
    """
    client = google.cloud.logging.Client(project="breuninger-playground-adm")
    cloud_handler = CloudLoggingHandler(client, name="prognose2020")
    cloud_handler.setFormatter(FORMATTER)

    return cloud_handler


def get_logger(logger_name):
    """
    Create a logger with the given name including File and Console handlers.

    :param logger_name: the name/identity of the logger you want to create
    :return: logger: the logger containing a file and console handler with the given name
    """
    logger = logging.getLogger(logger_name)

    if LOG_LEVEL == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    elif LOG_LEVEL == 'INFO':
        logger.setLevel(logging.INFO)
    elif LOG_LEVEL == 'WARN':
        logger.setLevel(logging.WARN)
    elif LOG_LEVEL == 'ERROR':
        logger.setLevel(logging.ERROR)
    elif LOG_LEVEL == 'CRITICAL':
        logger.setLevel(logging.CRITICAL)

    logger.addHandler(get_console_handler())
    logger.addHandler(get_google_handler())
    # logger.addHandler(get_file_handler())

    logger.propagate = False

    return logger
