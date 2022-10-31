import os
import json
import re
from webbrowser import get
from xmlrpc.client import boolean
import ipdb
from typing import List, Dict, Tuple, Optional, Iterable
from inspect import signature

# import all kinds of libraries for extracting APIs
import pandas
from pandas import *
import numpy
from numpy import *
import sklearn
from sklearn import *
import torch
from torch import *
from tensorflow import *
import tensorflow, tensorflow.keras
from django import *
import django, django.apps, django.conf, django.contrib, django.core, django.db, django.forms, \
    django.http, django.middleware, django.template, django.test, django.utils, django.views
from selenium import *
import selenium, selenium.common, selenium.webdriver
from matplotlib import *
import matplotlib, matplotlib.afm, matplotlib.animation, matplotlib.artist, matplotlib.axes, matplotlib.backend_bases, matplotlib.backend_managers, matplotlib.backend_tools,\
    matplotlib.backends, matplotlib.bezier, matplotlib.blocking_input, matplotlib.category, matplotlib.cbook, matplotlib.cm, matplotlib.collections,\
    matplotlib.colorbar, matplotlib.colors, matplotlib.contour, matplotlib.container, matplotlib.dates, matplotlib.docstring, matplotlib.dviread,\
    matplotlib.figure, matplotlib.font_manager, matplotlib.fontconfig_pattern, matplotlib.ft2font, matplotlib.gridspec, matplotlib.image, matplotlib.legend,\
    matplotlib.legend_handler, matplotlib.lines, matplotlib.markers, matplotlib.mathtext, matplotlib.mlab, matplotlib.offsetbox, matplotlib.patches,\
    matplotlib.path, matplotlib.patheffects, matplotlib.projections, matplotlib.pyplot, matplotlib.quiver, matplotlib.rcsetup, matplotlib.sankey, \
    matplotlib.scale, matplotlib.spines, matplotlib.stackplot, matplotlib.streamplot, matplotlib.style, matplotlib.table,\
    matplotlib.testing, matplotlib.text, matplotlib.textpath, matplotlib.ticker, matplotlib.tight_layout, matplotlib.transforms,\
    matplotlib.tri, matplotlib.type1font, matplotlib.units, matplotlib.widgets
import flask
import scipy, scipy.cluster, scipy.constants, scipy.fft, scipy.fftpack, scipy.integrate, scipy.interpolate, scipy.io, scipy.linalg, scipy.misc,\
    scipy.ndimage, scipy.odr, scipy.optimize, scipy.signal, scipy.sparse, scipy.spatial, scipy.special, scipy.stats
from scipy import *
import seaborn
from seaborn import *
import nltk
from nltk import *
import bs4, bs4.builder, bs4.dammit, bs4.diagnose, bs4.element, bs4.formatter
from bs4 import *
import pygame
from pygame import *
import PIL, PIL.BdfFontFile, PIL.BlpImagePlugin, PIL.BmpImagePlugin, PIL.BufrStubImagePlugin, PIL.ContainerIO, PIL.CurImagePlugin, PIL.DcxImagePlugin, PIL.DdsImagePlugin,\
    PIL.EpsImagePlugin, PIL.ExifTags, PIL.FitsStubImagePlugin, PIL.FliImagePlugin, PIL.FontFile, PIL.FpxImagePlugin, PIL.FtexImagePlugin, PIL.GbrImagePlugin, PIL.GdImageFile,\
    PIL.GifImagePlugin, PIL.GimpGradientFile, PIL.GimpPaletteFile, PIL.GribStubImagePlugin, PIL.Hdf5StubImagePlugin, PIL.IcnsImagePlugin, PIL.IcoImagePlugin, PIL.ImImagePlugin,\
    PIL.Image, PIL.ImageChops, PIL.ImageCms, PIL.ImageColor, PIL.ImageDraw, PIL.ImageDraw2, PIL.ImageEnhance, PIL.ImageFile, PIL.ImageFilter, PIL.ImageFont, PIL.ImageGrab,\
    PIL.ImageMath, PIL.ImageMode, PIL.ImageMorph, PIL.ImageOps, PIL.ImagePalette, PIL.ImagePath, PIL.ImageQt, PIL.ImageSequence, PIL.ImageShow, PIL.ImageStat, PIL.ImageTk,\
    PIL.ImageTransform, PIL.ImageWin, PIL.ImtImagePlugin, PIL.IptcImagePlugin, PIL.Jpeg2KImagePlugin, PIL.JpegImagePlugin, PIL.JpegPresets, PIL.McIdasImagePlugin, PIL.MicImagePlugin,\
    PIL.MpegImagePlugin, PIL.MpoImagePlugin, PIL.MspImagePlugin, PIL.PSDraw, PIL.PaletteFile, PIL.PalmImagePlugin, PIL.PcdImagePlugin, PIL.PcfFontFile, PIL.PcxImagePlugin, PIL.PdfImagePlugin,\
    PIL.PdfParser, PIL.PixarImagePlugin, PIL.PngImagePlugin, PIL.PpmImagePlugin, PIL.PsdImagePlugin, PIL.PyAccess, PIL.SgiImagePlugin, PIL.SpiderImagePlugin, PIL.SunImagePlugin,\
    PIL.TarIO, PIL.TgaImagePlugin, PIL.TiffImagePlugin, PIL.TiffTags, PIL.WalImageFile, PIL.WebPImagePlugin,\
    PIL.WmfImagePlugin, PIL.XVThumbImagePlugin, PIL.XbmImagePlugin, PIL.XpmImagePlugin, PIL.features
from PIL import *
import jieba
from jieba import *
import gensim
from gensim import *
import spacy
from spacy import *
import transformers
from transformers import *
import fairseq
from fairseq import *
import sqlalchemy
from sqlalchemy import *
import scrapy
from scrapy import *
import allennlp, allennlp.commands, allennlp.common, allennlp.confidence_checks, allennlp.data, allennlp.evaluation, allennlp.fairness, \
    allennlp.interpret, allennlp.models, allennlp.modules, allennlp.nn, allennlp.predictors, allennlp.tools, allennlp.training
from allennlp import *
import datasets
from datasets import *
import tokenizers
from tokenizers import *
# affecting the torch module, comment out first
# import mxnet 
# from mxnet import *
import imageio
from imageio import *
import pytest
from pytest import *
import metpy, metpy.constants, metpy.io, metpy.calc, metpy.plots, metpy.plots.ctables, metpy.interpolate
import ansible, ansible.cli, ansible.collections, ansible.compat, ansible.config, ansible.errors, ansible.executor, ansible.galaxy, ansible.inventory, ansible.module_utils,\
    ansible.modules, ansible.parsing, ansible.playbook, ansible.plugins, ansible.template, ansible.utils, ansible.vars, ansible.constants, ansible.context, ansible.release
from metpy import *
import torchdata
from torchdata import *
import torcharrow
from torcharrow import *

import requests
from requests import *

import datetime, zlib, random, math, sys, glob, os, urllib, time, re, json, unittest, collections, subprocess, copy, functools, itertools, six, threading,\
    tempfile, io, pickle, pathlib, socket, struct, hashlib, traceback, csv, uuid, pprint

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

MAX_DEPTH = 3 # default 3

def judge_not_normal_fun(attr):
    """
    input: function name
    output: whether the function is "not very regular"
    """
    return ((attr.startswith('__') and attr.endswith('__')) or (attr.startswith('_') and attr.endswith('_') or (attr.startswith("_")) or (attr.endswith("_")) or (attr.startswith("__")) or (attr.endswith("__"))))

def is_all_line_for_attrs(attrs):
    """
    input: library obj
    output: whether all the attributes are "not very regular" functions
    """
    try:
        dir_attrs = dir(attrs)
        for attr in dir_attrs:
            if not judge_not_normal_fun(attr):
                return False
        return True
    except:
        return True

def is_line_for_attr(attr):
    """
    input: function name
    output: whether the function is "not very regular"
    """
    if judge_not_normal_fun(attr):
        return True
    else:
        return False

def get_apis(
    lib_obj, 
    apis_dict, 
    depth, 
    lib_path,
    get_signature
    ):
    """
    input: library obj
    output: Dict of API：{API_name: API_doc}
    """
    if depth > MAX_DEPTH or is_all_line_for_attrs(lib_obj):
        return

    for attr in dir(lib_obj):
        if is_line_for_attr(attr):
            continue
        try:
            next_lib_path = lib_path + "." + attr
            if apis_dict.get(next_lib_path) is None:
                # format: {API_path: [API_doc, API_attr, API_signature]}
                if get_signature == "True":
                    apis_dict[next_lib_path] = [getattr(lib_obj, attr).__doc__, attr, signature(getattr(lib_obj, attr)).__str__()]
                else:
                    apis_dict[next_lib_path] = [getattr(lib_obj, attr).__doc__, attr]
            else:
                raise Exception("Duplicate API: {}".format(next_lib_path))
        except Exception as e:
            if getattr(lib_obj, attr).__doc__ != "" and getattr(lib_obj, attr).__doc__ != "None":
                apis_dict[next_lib_path] = [getattr(lib_obj, attr).__doc__, attr, ""]
            pass
        try:
            get_apis(getattr(lib_obj, attr), apis_dict, depth+1, lib_path + "." + attr, get_signature)
        except Exception as e:
            continue
    pass
    return apis_dict

def extract_apis_from_one_library(
    library_name: str, 
    output_dir: str, 
    process_num: int,
    get_signature: str
    ):
    """
    input: 库名称
    output: 库名称对应的API字典：{API_name: API_doc}
    """
    if library_name == "pandas":
        apis_dict = get_apis(pandas, dict(), 0, "pandas", get_signature)
    elif library_name == "numpy":
        apis_dict = get_apis(numpy, dict(), 0, "numpy", get_signature)
    elif library_name == "sklearn":
        apis_dict = get_apis(sklearn, dict(), 0, "sklearn", get_signature)
    elif library_name == "torch":
        apis_dict = get_apis(torch, dict(), 0, "torch", get_signature)
    elif library_name == "tensorflow":
        apis_dict = get_apis(tensorflow, dict(), 0, "tensorflow", get_signature)
    elif library_name == "django":
        apis_dict = get_apis(django, dict(), 0, "django", get_signature)
    elif library_name == "selenium":
        apis_dict = get_apis(selenium, dict(), 0, "selenium", get_signature)
    elif library_name == "matplotlib":
        apis_dict = get_apis(matplotlib, dict(), 0, "matplotlib", get_signature)
    elif library_name == "flask":
        apis_dict = get_apis(flask, dict(), 0, "flask", get_signature)
    elif library_name == "scipy":
        # import scipy, scipy.cluster, scipy.constants, scipy.fft, scipy.fftpack, scipy.integrate, scipy.interpolate, scipy.io, scipy.linalg, scipy.misc,\
        # scipy.ndimage, scipy.odr, scipy.optimize, scipy.signal, scipy.sparse, scipy.spatial, scipy.special, scipy.stats
        apis_dict = get_apis(scipy, dict(), 0, "scipy", get_signature)
        apis_dict = dict(get_apis(scipy.cluster, dict(), 0, "scipy.cluster", get_signature), **apis_dict)
        apis_dict = dict(get_apis(scipy.constants, dict(), 0, "scipy.constants", get_signature), **apis_dict)
        apis_dict = dict(get_apis(scipy.fft, dict(), 0, "scipy.fft", get_signature), **apis_dict)
        apis_dict = dict(get_apis(scipy.fftpack, dict(), 0, "scipy.fftpack", get_signature), **apis_dict)
        apis_dict = dict(get_apis(scipy.integrate, dict(), 0, "scipy.integrate", get_signature), **apis_dict)
        apis_dict = dict(get_apis(scipy.interpolate, dict(), 0, "scipy.interpolate", get_signature), **apis_dict)
        apis_dict = dict(get_apis(scipy.io, dict(), 0, "scipy.io", get_signature), **apis_dict)
        apis_dict = dict(get_apis(scipy.linalg, dict(), 0, "scipy.linalg", get_signature), **apis_dict)
        apis_dict = dict(get_apis(scipy.misc, dict(), 0, "scipy.misc", get_signature), **apis_dict)
        apis_dict = dict(get_apis(scipy.ndimage, dict(), 0, "scipy.ndimage", get_signature), **apis_dict)
        apis_dict = dict(get_apis(scipy.odr, dict(), 0, "scipy.odr", get_signature), **apis_dict)
        apis_dict = dict(get_apis(scipy.optimize, dict(), 0, "scipy.optimize", get_signature), **apis_dict)
        apis_dict = dict(get_apis(scipy.signal, dict(), 0, "scipy.signal", get_signature), **apis_dict)
        apis_dict = dict(get_apis(scipy.sparse, dict(), 0, "scipy.sparse", get_signature), **apis_dict)
        apis_dict = dict(get_apis(scipy.spatial, dict(), 0, "scipy.spatial", get_signature), **apis_dict)
        apis_dict = dict(get_apis(scipy.special, dict(), 0, "scipy.special", get_signature), **apis_dict)
        apis_dict = dict(get_apis(scipy.stats, dict(), 0, "scipy.stats", get_signature), **apis_dict)
    elif library_name == "seaborn":
        apis_dict = get_apis(seaborn, dict(), 0, "seaborn", get_signature)
    elif library_name == "nltk":
        apis_dict = get_apis(nltk, dict(), 0, "nltk", get_signature)
    elif library_name == "beautifulsoup":
        apis_dict = get_apis(bs4, dict(), 0, "bs4", get_signature)
    elif library_name == "pygame":
        apis_dict = get_apis(pygame, dict(), 0, "pygame", get_signature)
    elif library_name == "PIL":
        apis_dict = get_apis(PIL, dict(), 0, "PIL", get_signature)
    elif library_name == "jieba":
        apis_dict = get_apis(jieba, dict(), 0, "jieba", get_signature)
    elif library_name == "gensim":
        apis_dict = get_apis(gensim, dict(), 0, "gensim", get_signature)
    elif library_name == "spacy":
        apis_dict = get_apis(spacy, dict(), 0, "spacy", get_signature)
    elif library_name == "transformers":
        apis_dict = get_apis(transformers, dict(), 0, "transformers", get_signature)
    elif library_name == "fairseq":
        apis_dict = get_apis(fairseq, dict(), 0, "fairseq", get_signature)
    elif library_name == "sqlalchemy":
        apis_dict = get_apis(sqlalchemy, dict(), 0, "sqlalchemy", get_signature)
    elif library_name == "scrapy":
        apis_dict = get_apis(scrapy, dict(), 0, "scrapy", get_signature)
    elif library_name == "allennlp":
        apis_dict = get_apis(allennlp, dict(), 0, "allennlp", get_signature)
    elif library_name == "datasets":
        apis_dict = get_apis(datasets, dict(), 0, "datasets", get_signature)
    elif library_name == "tokenizers":
        apis_dict = get_apis(tokenizers, dict(), 0, "tokenizers", get_signature)
    elif library_name == "mxnet":
        apis_dict = get_apis(mxnet, dict(), 0, "mxnet", get_signature)
    elif library_name == "imageio":
        apis_dict = get_apis(imageio, dict(), 0, "imageio", get_signature)
        apis_dict = dict(get_apis(imageio.plugins.bsdf, dict(), 0, "imageio.plugins.bsdf", get_signature), **apis_dict)
        apis_dict = dict(get_apis(imageio.plugins.dicom, dict(), 0, "imageio.plugins.dicom", get_signature), **apis_dict)
        apis_dict = dict(get_apis(imageio.plugins.feisem, dict(), 0, "imageio.plugins.feisem", get_signature), **apis_dict)
        apis_dict = dict(get_apis(imageio.plugins.ffmpeg, dict(), 0, "imageio.plugins.ffmpeg", get_signature), **apis_dict)
        apis_dict = dict(get_apis(imageio.plugins.fits, dict(), 0, "imageio.plugins.fits", get_signature), **apis_dict)
        apis_dict = dict(get_apis(imageio.plugins.freeimage, dict(), 0, "imageio.plugins.freeimage", get_signature), **apis_dict)
        apis_dict = dict(get_apis(imageio.plugins.gdal, dict(), 0, "imageio.plugins.gdal", get_signature), **apis_dict)
        apis_dict = dict(get_apis(imageio.plugins.lytro, dict(), 0, "imageio.plugins.lytro", get_signature), **apis_dict)
        apis_dict = dict(get_apis(imageio.plugins.npz, dict(), 0, "imageio.plugins.npz", get_signature), **apis_dict)
        apis_dict = dict(get_apis(imageio.plugins.pillow, dict(), 0, "imageio.plugins.pillow", get_signature), **apis_dict)
        apis_dict = dict(get_apis(imageio.plugins.pillow_legacy, dict(), 0, "imageio.plugins.pillow_legacy", get_signature), **apis_dict)
        apis_dict = dict(get_apis(imageio.plugins.simpleitk, dict(), 0, "imageio.plugins.simpleitk", get_signature), **apis_dict)
        apis_dict = dict(get_apis(imageio.plugins.spe, dict(), 0, "imageio.plugins.spe", get_signature), **apis_dict)
        apis_dict = dict(get_apis(imageio.plugins.swf, dict(), 0, "imageio.plugins.swf", get_signature), **apis_dict)
        apis_dict = dict(get_apis(imageio.plugins.tifffile, dict(), 0, "imageio.plugins.tifffile", get_signature), **apis_dict)
    elif library_name == "pytest":
        apis_dict = get_apis(pytest, dict(), 0, "pytest", get_signature)
    elif library_name == "metpy":
        apis_dict = get_apis(metpy, dict(), 0, "metpy", get_signature)
    elif library_name == "ansible":
        apis_dict = get_apis(ansible, dict(), 0, "ansible", get_signature)
    elif library_name == "torchdata":
        apis_dict = get_apis(torchdata, dict(), 0, "torchdata", get_signature)
    elif library_name == "torcharrow":
        apis_dict = get_apis(torcharrow, dict(), 0, "torcharrow", get_signature)
    elif library_name == "requests":
        apis_dict = get_apis(requests, dict(), 0, "requests", get_signature)
    elif library_name == "datetime":
        apis_dict = get_apis(datetime, dict(), 0, "datetime", get_signature)
    elif library_name == "zlib":
        apis_dict = get_apis(zlib, dict(), 0, "zlib", get_signature)
    elif library_name == "random":
        apis_dict = get_apis(random, dict(), 0, "random", get_signature)
    elif library_name == "math":
        apis_dict = get_apis(math, dict(), 0, "math", get_signature)
    elif library_name == "sys":
        apis_dict = get_apis(sys, dict(), 0, "sys", get_signature)
    elif library_name == "glob":
        apis_dict = get_apis(glob, dict(), 0, "glob", get_signature)
    elif library_name == "os":
        apis_dict = get_apis(os, dict(), 0, "os", get_signature)
    elif library_name == "urllib":
        apis_dict = get_apis(urllib, dict(), 0, "urllib", get_signature)
    elif library_name == "time":
        apis_dict = get_apis(time, dict(), 0, "time", get_signature)
    elif library_name == "re":
        apis_dict = get_apis(re, dict(), 0, "re", get_signature)
    elif library_name == "json":
        apis_dict = get_apis(json, dict(), 0, "json", get_signature)
    elif library_name == "unittest":
        apis_dict = get_apis(unittest, dict(), 0, "unittest", get_signature)
    elif library_name == "collections":
        apis_dict = get_apis(collections, dict(), 0, "collections", get_signature)
    elif library_name == "subprocess":
        apis_dict = get_apis(subprocess, dict(), 0, "subprocess", get_signature)
    elif library_name == "copy":
        apis_dict = get_apis(copy, dict(), 0, "copy", get_signature)
    elif library_name == "functools":
        apis_dict = get_apis(functools, dict(), 0, "functools", get_signature)
    elif library_name == "itertools":
        apis_dict = get_apis(itertools, dict(), 0, "itertools", get_signature)
    elif library_name == "six":
        apis_dict = get_apis(six, dict(), 0, "six", get_signature)
    elif library_name == "threading":
        apis_dict = get_apis(threading, dict(), 0, "threading", get_signature)
    elif library_name == "tempfile":
        apis_dict = get_apis(tempfile, dict(), 0, "tempfile", get_signature)
    elif library_name == "io":
        apis_dict = get_apis(io, dict(), 0, "io", get_signature)
    elif library_name == "pickle":
        apis_dict = get_apis(pickle, dict(), 0, "pickle", get_signature)
    elif library_name == "pathlib":
        apis_dict = get_apis(pathlib, dict(), 0, "pathlib", get_signature)
    elif library_name == "socket":
        apis_dict = get_apis(socket, dict(), 0, "socket", get_signature)
    elif library_name == "struct":
        apis_dict = get_apis(struct, dict(), 0, "struct", get_signature)
    elif library_name == "hashlib":
        apis_dict = get_apis(hashlib, dict(), 0, "hashlib", get_signature)
    elif library_name == "traceback":
        apis_dict = get_apis(traceback, dict(), 0, "traceback", get_signature)
    elif library_name == "csv":
        apis_dict = get_apis(csv, dict(), 0, "csv", get_signature)
    elif library_name == "uuid":
        apis_dict = get_apis(uuid, dict(), 0, "uuid", get_signature)
    elif library_name == "pprint":
        apis_dict = get_apis(pprint, dict(), 0, "pprint", get_signature)
    else:
        raise NotImplementedError("Not supported library: {}".format(library_name))

    logging.info(f"len of {library_name} apis: {len(apis_dict)}")

    return apis_dict


def extract_apis_from_libraries(
    libraries: str, 
    out_dir: str, 
    process_num: int,
    overwrite: str,
    get_signature: str
    ):

    # cpu_count = min(process_num, multiprocessing.cpu_count())
    cpu_count = process_num
    libraries_list = libraries.split(",")

    libraries_api_number_dict = dict()
    for library in libraries_list:
        output_file_path = os.path.join(out_dir, library, f"{library}_apis_doc_{get_signature}.jsonl")
        if os.path.exists(output_file_path):
            if overwrite == "True":
                os.remove(output_file_path)
            else:
                logging.info(f"{output_file_path} already exists, skip")
                continue
        else:
            logging.info(f"{output_file_path} does not exist, start to create it")
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        writer = open(output_file_path, "w+")

        library_apis = extract_apis_from_one_library(library.strip(), out_dir, cpu_count, get_signature)
        libraries_api_number_dict[library] = len(library_apis)
        
        if get_signature == "True":
            for api_path, [api_doc, api_name, api_signature] in library_apis.items():
                api_doc = api_doc if isinstance(api_doc, str) else str(api_doc)
                writer.write(json.dumps({
                    api_path: [api_doc, api_name, api_signature]
                })+"\n")
        else:
            for api_path, [api_doc, api_name] in library_apis.items():
                api_doc = api_doc if isinstance(api_doc, str) else str(api_doc)
                writer.write(json.dumps({
                    api_path: [api_doc, api_name]
                })+"\n")

        logging.info(f"Successfully {library} APIs extracted!")

    # print the library name and its corresponding number of APIs
    for library, number in libraries_api_number_dict.items():
        logging.info(f"{library} has {number} APIs")

    logging.info(f"Have extracted successfully all libraries ({libraries}) APIs!")
    logging.info(f"the total number of libraries: {len(libraries_list)}")
    pass


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='extract api from all kinds of python libraries')
    parser.add_argument('-ls', '--libraries', type=str, default='pandas, numpy', help='all kinds of libraries', required=True)
    parser.add_argument('-o', '--output_dir', type=str, help='output dir', required=True)
    parser.add_argument('-pn', '--process_num', type=int, default=20, help='process cpu num')
    parser.add_argument('-ow', '--overwrite', type=str, default="False", help='whether overwrite the output file')
    parser.add_argument('-gs', '--get_signature', type=str, default="True", help='whether get signature')

    args = parser.parse_args()

    logging.info(f"Libraries: [{args.libraries}]")
    logging.info(f"whether overwrite the output file: {args.overwrite}")
    
    extract_apis_from_libraries(
        libraries=args.libraries, 
        out_dir=args.output_dir, 
        process_num=args.process_num,
        overwrite=args.overwrite,
        get_signature=args.get_signature
    )
    
    pass

    