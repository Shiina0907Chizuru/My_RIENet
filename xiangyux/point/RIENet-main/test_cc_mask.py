import mrcfile
import numpy as np
import torch
import torch.nn as nn
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
import einops
import random
from itertools import product
import pickle
MRCObject = namedtuple("MRCObject", ["grid", "voxel_size", "global_origin", "nstart"])

def load_Map(input_map_path)-> MRCObject:
    # Read MRC file
    mrc = mrcfile.open(input_map_path, permissive=True)
    grid= mrc.data.copy()
    voxel_size_o = float(mrc.voxel_size.x)
    voxel_size = np.asarray(mrc.voxel_size.tolist(), dtype=np.float32)

    origin = np.array(mrc.header.origin.tolist(), dtype=np.float32)
    nstart = np.asarray([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart], dtype=np.float32)

    mapcrs = np.asarray([mrc.header.mapc, mrc.header.mapr, mrc.header.maps], dtype=int)


    mrc.close()

    # Reorder
    sort = np.asarray([0, 1, 2], dtype=np.int64)
    for i in range(3):
        sort[mapcrs[i] - 1] = i
    nstart = np.asarray([nstart[i] for i in sort])

    grid = np.transpose(grid, axes=2 - sort[::-1])
    global_origin = origin
    # Move offsets from nstart to origin if origin is zero (MRC2000)
    if np.all(origin == 0):
        global_origin = origin + nstart * voxel_size




    return MRCObject(grid, voxel_size_o, global_origin,nstart)

def save_mrc(grid, voxel_size, origin,nstart, filename):
    (z, y, x) = grid.shape
    o = mrcfile.new(filename, overwrite=True)
    o.header["cella"].x = x * voxel_size
    o.header["cella"].y = y * voxel_size
    o.header["cella"].z = z * voxel_size
    o.header["origin"].x = origin[0]
    o.header["origin"].y = origin[1]
    o.header["origin"].z = origin[2]
    (o.header.nxstart, o.header.nystart, o.header.nzstart)=nstart
    out_box = np.reshape(grid, (z, y, x))
    o.set_data(out_box.astype(np.float32))
    o.update_header_stats()
    o.flush()
    o.close()















map_all_path = "/defaultShare/FinialPDB/PDB-6f0k-EMD-4165/tgt.map"
map_chain_path = "/defaultShare/FinialPDB/PDB-6f0k-EMD-4165/src.map"

out = "/defaultShare/FinialPDB/PDB-6f0k-EMD-4165"
out_all_map = os.path.join(out, "tgt_s.map")
out_chain_map = os.path.join(out, "src_s.map")

map_all = load_Map(map_all_path)
map_chain = load_Map(map_chain_path)

tgt = map_all.grid
src = map_chain.grid
tgt = tgt * src *10
src = src * src *10
save_mrc(tgt, map_all.voxel_size, map_all.global_origin, map_all.nstart, out_all_map)
save_mrc(src, map_all.voxel_size, map_all.global_origin, map_all.nstart, out_chain_map)