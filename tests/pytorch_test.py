# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     pytorch_test
   Description :
   Author :       xmz
   date：          2019/4/25
-------------------------------------------------
"""
import torch


def matmul_test():
    a = torch.randn(2, 20)
    print(a)
    a = a.view(2, 4, -1)
    print(a)


if __name__ == "__main__":
    matmul_test()
