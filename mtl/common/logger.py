# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     logger
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
logger.add(os.path.join("logs", "log_{time}.txt"), enqueue=True, colorize=True,
           format="{level} | {file} | {module} | {time:YYYY-MM-DD at HH:mm:ss} | {message}", level="INFO")
