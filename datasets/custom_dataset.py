import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch
import copy, os, collections
import json
# from .human_parse_labels import get_label_map, DF_LABEL, YF_LABEL
import pandas as pd
from utils import pose_utils




class SEG:
    labels = {'background': 0, 'hat':1, 'hair': 2,  'face': 3, 'upper-clothes':4,
               'pants': 5, 'arm': 6, 'leg': 7, 'shoes':8, 'skin':9}
    BACKGROUND = 0
    HAT = 1
    HAIR = 2    
    FACE = 3
    UPPER = 4
    PANTS = 5
    ARMS = 6
    LEGS = 7
    SHOES = 8
    SKIN = 9

    CUSTOM_LABELS = {'background': 0, 'hat':1, 'hair': 2,  'face': 3, 'upper-clothes':4,
               'pants': 5, 'arm': 6, 'leg': 7, 'shoes':8, 'skin':9, 'glove':10 }
    
    # PID = [0,4,6,7] # bg, face, arm, leg (the first has to be bg and the second has to be face.)
    # GID = [2,5,1,3] # hair, top, bottom, jacket

    # my params
    PERSON_IDS = [BACKGROUND,FACE,ARMS,LEGS,SKIN] # bg, face, skin, arm, leg (the first has to be bg and the second has to be face.)
    GARMENT_IDS = [5,6,7,8,9] # hair, shoes, top, bottom, hat
