"""
This module contains utility functions for data analysis. 

It includes functions for logging, saving plots, and performing statistical analysis.

Author: Your Name
Date: Current Date
"""



from hydra.experimental import initialize, compose
import churn_library as cl
from utils import get_logger

def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	with initialize(config_path="config"):
		cfg = compose("stable_train.yaml")
	cfg.job_logging.name = "test_ChurnLibrary"
	# import pdb; pdb.set_trace()
	logger = get_logger(cfg=cfg)
	cly_test = cl.ChurnLibrary(cfg=cfg, logger=logger)
	
	try:
		cly_test.import_data()  # import data
		logger.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logger.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert cly_test.df.shape[0] > 0
		assert cly_test.df.shape[1] > 0
	except AssertionError as err:
		logger.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err




