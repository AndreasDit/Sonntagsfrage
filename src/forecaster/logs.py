import logging
import sys
import yaml
import os
from datetime import datetime
from logging import FileHandler

sys.path.append(os.getcwd())
import src.forecaster.configs_for_code as cfg

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)

FILE_PATH_LOGGING = configs['logging']['file_path']
FILE_NAME_LOGGING = configs['logging']['file_name']
FORMATTING = logging.Formatter(configs['logging']['format'])
LOG_LEVEL = configs['logging']['log_level']

today = datetime.today()
FILE_NAME_LOGGING = FILE_NAME_LOGGING.replace('<YYYYmmddHMS>', today.strftime('%Y%m%d%H%M%S'))

def create_file_handler():
    """
    Creates a file_handler for logging

    :return: file_handler
    """

    file_handler = FileHandler(FILE_PATH_LOGGING + FILE_NAME_LOGGING)
    file_handler.setFormatter(FORMATTING)

    return file_handler

def create_logger(logger_name):
    """
    Creates a logger with the given name including File and Console handlers.

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

    logger.addHandler(create_file_handler())

    logger.propagate = False

    return logger

