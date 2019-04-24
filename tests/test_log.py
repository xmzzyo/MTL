# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_log
   Description :
   Author :       xmz
   date：          2019/4/22
-------------------------------------------------
"""
import os
import sys

from loguru import logger

logger.remove()
logger.add(sys.stdout, colorize=True,
           format="<yellow>{level}</yellow> | <red>{file}</red> | <red>{module}</red> | " \
                  "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | <level>{message}</level>",
           level="INFO")
logger.add(os.path.join("../../logs", "log_{time}.txt"), enqueue=True, colorize=True,
           format="{level} | {file} | {module} | {time:YYYY-MM-DD at HH:mm:ss} | {message}", level="INFO")

logger.info("sth")
logger.info("sb")
logger.info("If you're using Python {:.2f}, prefer  of course!", 3.6333)
