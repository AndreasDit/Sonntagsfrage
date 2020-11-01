import yaml

import logs
import utils
import configs_for_code as cfg

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

def load_data():
    """
        This function loads all data from the sources and combines them into one dataframe for further processing.

        :return: pandas.DataFrame: Returns one dataframe with the combinated data from all input sources.
    """

    logger.debug("Started load_data()")




