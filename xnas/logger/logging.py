"""Logging."""

import os
import logging
import simplejson
from xnas.core.config import cfg
from xnas.core.utils import float_to_decimal


# Show filename and line number in logs
_FORMAT = "[%(filename)s: %(lineno)3d]: %(message)s"

# Log file name
_LOG_FILE = "stdout.log"

# Data output with dump_log_data(data, data_type) will be tagged w/ this
_TAG = "json_stats: "

# Data output with dump_log_data(data, data_type) will have data[_TYPE]=data_type
_TYPE = "_type"


def setup_logging():
    """Sets up the logging."""
    # Clear the root logger to prevent any existing logging config
    # (e.g. set by another module) from messing with our setup
    logging.root.handlers = []
    # Construct logging configuration
    logging_config = {"level": logging.INFO, "format": _FORMAT}
    logging_config["filename"] = os.path.join(cfg.OUT_DIR, _LOG_FILE)
    # Configure logging
    logging.basicConfig(**logging_config)


def get_logger(name):
    """Retrieves the logger."""
    return logging.getLogger(name)


def dump_log_data(data, data_type, prec=4):
    """Covert data (a dictionary) into tagged json string for logging."""
    data[_TYPE] = data_type
    data = float_to_decimal(data, prec)
    data_json = simplejson.dumps(data, sort_keys=True, use_decimal=True)
    return "{:s}{:s}".format(_TAG, data_json)
