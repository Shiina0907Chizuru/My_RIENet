import os
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import io
import os
from collections import namedtuple
from Bio.PDB import  MMCIFParser, PDBParser
from Bio.PDB.Atom import DisorderedAtom
from os.path import join, isfile, isdir, basename, normpath
from os import listdir
from Bio import PDB
import random
import pickle
import mrcfile

def check_duplicates_and_save(file_path):
    # 读取TSV文件
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 提取第一列的数据
    first_column = [line.split('\t')[0] for line in lines]

    # 检查是否有重复项
    if len(first_column) == len(set(first_column)):
        # 如果没有重复项，输出文件路径到一个新的记事本（追加模式）
        output_file = "/xiangyux/point/RIENet-main/data/out_put.txt"
        with open(output_file, 'a') as out_file:  # 使用'a'模式，追加写入
            out_file.write(f"当前文件路径: {os.path.abspath(file_path)}\n")
        print(f"文件已保存：{file_path}")
    else:
        print("第一列包含重复项")
def traverse_folders(base_path):
    # 遍历文件夹路径
    for root, dirs, files in os.walk(base_path):
        # 遍历每个文件
        for file in files:
            if file.endswith(".tsv"):  # 只处理 .tsv 文件
                file_path = os.path.join(root, file)
                check_duplicates_and_save(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="排除冗余链", epilog="v0.0.1")
    parser.add_argument("input",
                        action="store",
                        help="path to input file根目录")


    args = parser.parse_args()
    input_folder = join(args.input)

    traverse_folders(input_folder)