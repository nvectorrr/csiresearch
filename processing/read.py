'''Функции работы с файловой системой и сортировки файлов с CSI'''
from os import DirEntry, scandir
from typing import List
from re import search
import numpy as np
import csiread


def listdirs(rootdir: str) -> List[DirEntry]:
    lst = []

    def func(rootdir):
        for it in scandir(rootdir):
            if it.is_dir():
                func(it)
            else:
                lst.append(it)
    func(rootdir)

    return lst


def categorize(lst: List[DirEntry], categories: List[str]) -> dict[str, List[DirEntry]]:
    catsPathes: dict[str, List[str]] = {}

    for cat in categories:
        for item in lst:
            if search('.*' + cat + '.*', item.name):
                if cat in catsPathes:
                    catsPathes[cat].append(item)
                else:
                    catsPathes[cat] = [item]

    return catsPathes


def extractCSI20MHz(fpath: str) -> np.ndarray:
    data = csiread.Atheros(fpath, nrxnum=2, ntxnum=5,
                           tones=56, if_report=False)
    data.read(endian='big')
    payload_len = np.bincount(data.payload_len).argmax()
    csi = data.csi[(data.payload_len == payload_len)
                   & (data.nc == 2)][:, :, :2, :2]
    return csi, data

def extractCSI40MHz(fpath: str) -> np.ndarray:
    data = csiread.Atheros(fpath, nrxnum=2, ntxnum=5,
                           tones=114, if_report=False)
    data.read(endian='big')
    payload_len = np.bincount(data.payload_len).argmax()
    csi = data.csi[(data.payload_len == payload_len)
                   & (data.nc == 2)][:, :, :2, :2]
    return csi, data


# Также функции множественного считывания файлов по корневому пути

def extractFrom(fdir: List[str]):
    data = []
    for fname in scandir(fdir):
        data.append({'filename': fname, 'csi': extractCSI(fdir + '/' + fname)})

    return data
