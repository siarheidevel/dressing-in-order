import logging
from typing import List
import requests
import torch
import os, json, random
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from io import StringIO, BytesIO
import shutil, random, json
from pathlib import Path
import numpy as np
import cv2, logging, time
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.chdir(os.path.abspath(os.path.dirname(__file__)))
CUDA_DEVICE= 'cuda:1'
torch.cuda.set_device(CUDA_DEVICE)
logging.basicConfig(level = logging.INFO)
from datasets.custom_dataset import SEG, PairDataset, get_palette, _view_pose
from models.customdior_model import CustomDIORModel
from utils import pose_utils

class Opt:
    def __init__(self):
        pass
# # bg, face, skin, arm, leg (the first has to be bg and the second has to be face.)
# PID  = [SEG.ID['background'], SEG.ID['face'], SEG.ID['skin'], SEG.ID['arm'], SEG.ID['leg']] 
PID = SEG.PERSON_IDS

# # hair, shoes, top, bottom, hat
# GID = [SEG.ID['hair'], SEG.ID['shoes'], SEG.ID['pants'], SEG.ID['upper-clothes'], SEG.ID['hat']]
GID = SEG.GARMENT_IDS

IMG_DIOR_SIZE = (256, 176)
IMG_DEFAULT_SIZE = (512, 352)
IMG_DIOR_SIZE = (512, 352)
IMG_DEFAULT_SIZE = (768, 528)
TEMP_DIR = '/tmp/model/dressing'
    
img_transforms = T.Compose([T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])


# # 256x176 res
# IMG_DIOR_SIZE = (256, 176)
# IMG_DEFAULT_SIZE = (512, 352)
# def get_model_params():
#     opt = Opt()
#     # opt.dataroot = dataroot
#     opt.isTrain = False
#     opt.phase = 'test'
#     opt.n_human_parts = 10; opt.n_kpts = 18; opt.style_nc = 64
#     opt.n_style_blocks = 6; opt.netG = 'dior'; opt.netE = 'adgan'; opt.ngf = 64    
#     opt.norm_type = 'instance'; opt.relu_type = 'leakyrelu'
#     opt.init_type = 'orthogonal'; opt.init_gain = 0.02; 
#     opt.random_rate = 1; opt.perturb = False; opt.warmup=False; opt.verbose = False
#     opt.name = "dior_custom_64_256_full"
#     opt.checkpoints_dir = 'checkpoints'; opt.vgg_path = ''; opt.flownet_path = ''; 
#     opt.frozen_enc = True; opt.frozen_flownet = True;
#     opt.load_iter = 0; opt.epoch = 'latest'
#     opt.gpu_ids = [0] 
#     return opt

# 512x352 res
IMG_DIOR_SIZE = (512, 352)
IMG_DEFAULT_SIZE = (768, 528)
def get_model_params():
    opt = Opt()
    # opt.dataroot = dataroot
    opt.isTrain = False
    opt.phase = 'test'
    opt.n_human_parts = 10; opt.n_kpts = 18; opt.style_nc = 32
    opt.n_style_blocks = 4; opt.netG = 'dior'; opt.netE = 'adgan'; opt.ngf = 32    
    opt.norm_type = 'instance'; opt.relu_type = 'leakyrelu'
    opt.init_type = 'orthogonal'; opt.init_gain = 0.02; 
    opt.random_rate = 1; opt.perturb = False; opt.warmup=False; opt.verbose = False
    # opt.name = "dior_custom_512_deepfashion_1"
    opt.name = "dior_custom_512_deepfashion_2_aug_nobg"
    opt.checkpoints_dir = 'checkpoints'; opt.vgg_path = ''; opt.flownet_path = ''; 
    opt.frozen_enc = True; opt.frozen_flownet = True;
    opt.load_iter = 0; opt.epoch = 'latest'
    opt.gpu_ids = [1]
    return opt

# # 256x176 res
# IMG_DIOR_SIZE = (256, 176)
# IMG_DEFAULT_SIZE = (512, 352)
# def get_model_params():
#     opt = Opt()
#     # opt.dataroot = dataroot
#     opt.isTrain = False
#     opt.phase = 'test'
#     opt.n_human_parts = 8; opt.n_kpts = 18; opt.style_nc = 64
#     opt.n_style_blocks = 4; opt.netG = 'dior'; opt.netE = 'adgan'; opt.ngf = 64    
#     opt.norm_type = 'instance'; opt.relu_type = 'leakyrelu'
#     opt.init_type = 'orthogonal'; opt.init_gain = 0.02; 
#     opt.random_rate = 1; opt.perturb = False; opt.warmup=False; opt.verbose = False
#     opt.name = "DIORv1_64"
#     opt.checkpoints_dir = 'checkpoints'; opt.vgg_path = ''; opt.flownet_path = ''; 
#     opt.frozen_enc = True; opt.frozen_flownet = True;
#     opt.load_iter = 0; opt.epoch = 'latest'
#     opt.gpu_ids = [0] 
#     return opt

def load_model():
    opt = get_model_params()
    model = CustomDIORModel(opt)
    # from models.dior_model import DIORModel
    # model = DIORModel(opt)
    # freeze model
    for model_name in model.model_names:
        inner_model = 'net' + model_name
        # for m in model.split('.'):
        net = getattr(model, inner_model)
        for param in net.parameters():
            param.requires_grad = False
        net.eval()
    model.eval()
    model.load_networks(opt.epoch)
    model.print_networks(opt.verbose)
    print("[init] frozen net %s." % model)
    # model.setup(opt)
    return model


@torch.no_grad()
def swap_garment(source_img, source_pose, source_parse,
    garment_img, garment_pose, garment_parse, swap_garment_ids:List[int]):
    source_img = source_img.to(model.device)[None,...]
    source_pose = source_pose.float().to(model.device)[None,...]
    source_parse = source_parse.float().to(model.device)[None,...]

    garment_img = garment_img.to(model.device)[None,...]
    garment_pose = garment_pose.float().to(model.device)[None,...]
    garment_parse = garment_parse.float().to(model.device)[None,...]

    gsegs = model.encode_attr(source_img,source_parse,
                                source_pose, source_pose, GID)
    psegs = model.encode_attr(source_img,source_parse,
                                source_pose, source_pose, PID)
    for garment_id in swap_garment_ids:
        garment_seg = model.encode_single_attr(garment_img, garment_parse, 
            garment_pose, source_pose, garment_id)        
        gsegs[GID.index(garment_id)] = garment_seg

    img_tensor = model.decode(source_pose, psegs, gsegs)
    new_img = Image.fromarray((img_tensor[0].cpu().permute([1,2,0]).numpy()*127 + 127).astype(np.uint8))
    new_img.save('img_out.png')
    # TODO swap face
    return new_img

def resize2box(im, box=(256,176), fill_color=[0,0,0], interpolation = Image.LANCZOS):
    '''
    Resize image to box, fill fill_color
    '''
    if isinstance(im, str):
        im = np.array(Image.open(im))
    elif isinstance(im, Path):
        im = np.array(Image.open(im))
    elif isinstance(im, Image.Image):
        im = np.array(im)
    ratio = min(box[0]/im.shape[0], box[1]/im.shape[1])
    im = np.array(Image.fromarray(im).resize((int(ratio*im.shape[1]),int(ratio*im.shape[0])),interpolation))
    h_start = (box[0]-im.shape[0]) // 2 
    w_start = (box[1]-im.shape[1]) // 2
    if len(im.shape) == 2:
        image_array = np.ones((*box,),dtype=np.uint8) * fill_color
    else:
        image_array = np.ones((*box,im.shape[-1]),dtype=np.uint8) * fill_color    

    image_array[h_start:h_start+im.shape[0],
            w_start:w_start+im.shape[1],...] = np.array(im)
    return Image.fromarray(image_array.astype(np.uint8))

def crop2box(im, box=(256,176)):
    '''
    Resize and crop image to box
    '''
    if isinstance(im, str):
        im = np.array(Image.open(im))
    elif isinstance(im, Path):
        im = np.array(Image.open(im))
    elif isinstance(im, Image.Image):
        im = np.array(im)
    ratio = max(box[0]/im.shape[0], box[1]/im.shape[1])
    im = np.array(Image.fromarray(im).resize((int(ratio*im.shape[1]),int(ratio*im.shape[0])),Image.LANCZOS))
    return im



def preprocess_image(image_file):
    image_array = np.zeros((*IMG_DEFAULT_SIZE,3),dtype=np.uint8)
    image_rgb = cv2.imread(image_file)[:,:,[2,1,0]]

    # get person bbox
    #     curl -X 'POST' \
    #   'http://localhost:8010/boxes' \
    #   -H 'accept: application/json' \
    #   -H 'Content-Type: multipart/form-data' \
    #   -F 'upload_file=@000057_0.jpg;type=image/jpeg'
    with open(image_file, 'rb') as f:
        files = {'upload_file': f}
        response = requests.post('http://localhost:8010/boxes', files=files)
        if response.status_code == 200:
            json = response.json()
            bboxes = json['bboxes']
    
    assert len(bboxes)>0
    box=bboxes[0]
    width, height = image_rgb.shape[1], image_rgb.shape[0]
    # Image.fromarray(image_rgb[int(pred_boxes[0][0][1]):int(pred_boxes[0][1][1]),int(pred_boxes[0][0][0]):int(pred_boxes[0][1][0]),...]).save("box_img.png")
    # 384, 512
    margin_y, margin_x = (box[2] - box[0])*0.05, (box[3] - box[1])*0.05
    box_img_array = image_rgb[int(max(box[1]-margin_y, 0)):int(min(box[3]+margin_y,height)),
        int(max(box[0]-margin_x, 0)):int(max(box[2]+margin_x,width)),...]


    im = resize2box(box_img_array, IMG_DEFAULT_SIZE, [0,0,0])
    
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    new_image_file = os.path.join(TEMP_DIR, 
        ''.join(random.choice('abcdefgh') for i in range(4))+'_'+ Path(image_file).name)
    
    preprocess_image_file=new_image_file + ".preprocess.jpg"
    # dior_image_file = new_image_file + ".dior.thumb.jpg"
    im.save(preprocess_image_file, "JPEG")
    # Image.fromarray(image_array).resize(list(reversed(IMG_DIOR_SIZE)), resample=Image.LANCZOS).save(
    #         dior_image_file, "JPEG")
    return new_image_file

def get_data_from_image(image_file):
    new_image_file = preprocess_image(image_file)
    preprocess_image_file = new_image_file + ".preprocess.jpg"
    # get parsing result
    #     curl -X 'POST' \
    #   'http://localhost:8010/segment/?render=true' \
    #   -H 'accept: application/json' \
    #   -H 'Content-Type: multipart/form-data' \
    #   -F 'upload_file=@02_1_front.jpg;type=image/jpeg'
    with open(preprocess_image_file, 'rb') as f:
        files = {'upload_file': f}
        response = requests.post('http://localhost:8010/segment/?render=true', files=files)
        if response.status_code == 200:
            parse_img = Image.open(BytesIO(response.content))
            parse_img_array = np.array(parse_img)
            # 
    
    #
    non_background = np.where(parse_img_array>0)
    margin_h, margin_w = 0, 10
    bbox = min(non_background[0]),min(non_background[1]),max(non_background[0]),max(non_background[1])
    bbox = max(0,bbox[0] - 10),max(0,bbox[1] - margin_w),min(bbox[2]+margin_h, parse_img_array.shape[0]), min(bbox[3]+margin_w, parse_img_array.shape[1])
    logging.info(f'bbox {bbox}')
    parse_img_array = parse_img_array[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    im = cv2.imread(new_image_file + ".preprocess.jpg")[:,:,[2,1,0]]
    im = im[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    
    im = resize2box(im, box = IMG_DIOR_SIZE)
    im.save(new_image_file + ".thumb.jpg")
    parse_img_array = np.array(resize2box(parse_img_array, box = IMG_DIOR_SIZE, fill_color=[0],interpolation=Image.NEAREST))
    # resize2box(parse_img_array, box = IMG_DIOR_SIZE, fill_color=[0]).convert("P", palette=Image.ADAPTIVE, colors=10).save('img.png')
    

    print('')

    
    # get pose estimation
    # curl -X 'POST' \
    # 'http://localhost:8020/pose/?render=true' \
    # -H 'accept: application/json' \
    # -H 'Content-Type: multipart/form-data' \
    # -F 'upload_file=@02_1_front.jpg;type=image/jpeg'
    with open(new_image_file + ".thumb.jpg", 'rb') as f:
        files = {'upload_file': f}
        response = requests.post('http://localhost:8020/pose/?render=true', files=files)
        if response.status_code == 200:
            json = response.json()
            x_cords, y_cords = json['x'], json['y']
    

    

    seg = parse_img_array
    redused_seg = PairDataset._reduce_segmentation(seg)

    pose_array = np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)
    pose = pose_utils.cords_to_map(pose_array, IMG_DIOR_SIZE, affine_matrix=None, sigma=5)
    pose = np.transpose(pose,(2, 0, 1))

    img = np.array(im)
    #remove bg ####################################################################
    # img[redused_seg==SEG.ID['background'],:] = [234,230,224]
    # Image.fromarray((cv2.blur((redused_seg==SEG.ID['background']).astype(np.float),(5,5),0)*254).astype(np.uint8)).save('mask.png')
    bg_kernel = (3, 3)
    # bg_color = [234,230,224]
    bg_color = [255,249,249]
    bg_mask = (redused_seg==SEG.ID['background']).astype(np.uint8) #Image.fromarray((bg_mask*255).astype(np.uint8)).save('mask.png')
    eroded_mask = cv2.dilate(bg_mask,np.ones(bg_kernel),iterations=1) #Image.fromarray((eroded_mask*255).astype(np.uint8)).save('mask.png')
    eroded_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_OPEN, np.ones(bg_kernel, np.uint8))
    blurred_mask = cv2.GaussianBlur(eroded_mask.astype(np.float),bg_kernel,0) #Image.fromarray((blurred_mask*255).astype(np.uint8)).save('mask.png')
    # blurred_img = cv2.blur(img.astype(np.float),(5,5),0)  #Image.fromarray((blurred_img).astype(np.uint8)).save('mask.png')
    bg_img = np.ones_like(img)*bg_color
    final_img = img * (1-blurred_mask[...,None]) + blurred_mask[...,None] * bg_img
    Image.fromarray((final_img ).astype(np.uint8)).save('mask.png')
    img = final_img.astype(np.uint8)
    ###############################################################################
    
    img_tensor = img_transforms(img)
    mask_image=Image.fromarray(redused_seg);mask_image.putpalette(get_palette(len(SEG.ID)));mask_image.save('seg.png')
    image_image=Image.fromarray((img_tensor.permute([1,2,0]).numpy()*127 + 127).astype(np.uint8));image_image.save('img.png')
    pose_image=Image.fromarray(_view_pose(torch.Tensor(pose)));pose_image.save('pose.png')
    Image.blend(Image.blend(mask_image.convert("RGBA"),image_image.convert("RGBA"),0.5), pose_image.convert("RGBA"),0.5).save('input.png')
    return img_tensor, torch.Tensor(pose), torch.Tensor(redused_seg)



def dress_garment(image_file, garment_file, GARMENT_IDS:List[int]):
    source_img, source_pose, source_parse = get_data_from_image(image_file)
    garment_img, garment_pose, garment_parse = get_data_from_image(garment_file)

    new_image = swap_garment(source_img, source_pose, source_parse,
        garment_img, garment_pose, garment_parse, GARMENT_IDS)
    
    #draw face
    s_img = source_img.permute([1,2,0]).numpy()*127 + 127
    g_img = garment_img.permute([1,2,0]).numpy()*127 + 127
    new_image_arr = np.array(new_image)

    ##############################################################################################################
    # new_image_arr[source_parse == SEG.ID['face'],:]=s_img[source_parse == SEG.ID['face']]
    bg_kernel = (5, 5)
    # bg_color = [234,230,224]
    bg_mask = (source_parse.numpy() == SEG.ID['face']).astype(np.uint8) #Image.fromarray((bg_mask*255).astype(np.uint8)).save('mask.png')
    eroded_mask = cv2.erode(bg_mask,np.ones(bg_kernel),iterations=1) #Image.fromarray((eroded_mask*255).astype(np.uint8)).save('mask.png')
    # eroded_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_OPEN, np.ones(bg_kernel, np.uint8))
    blurred_mask = cv2.blur(eroded_mask.astype(np.float),bg_kernel,0) #Image.fromarray((blurred_mask*255).astype(np.uint8)).save('mask.png')
    # blurred_img = cv2.blur(img.astype(np.float),(5,5),0)  #Image.fromarray((blurred_img).astype(np.uint8)).save('mask.png')
    final_img = new_image_arr * (1-blurred_mask[...,None]) + blurred_mask[...,None] * s_img
    Image.fromarray((final_img ).astype(np.uint8)).save('mask.png')
    new_image_arr = final_img.astype(np.uint8)


    # new_image_arr[source_parse == SEG.ID['hair'],:]=s_img[source_parse == SEG.ID['hair']]

    Image.fromarray(np.concatenate((s_img,g_img,new_image_arr),1).astype(np.uint8)).save('img.jpg')
    print('ss')
    return Image.fromarray(new_image_arr)



import uvicorn
from fastapi import FastAPI, Query, Cookie, Header, File, UploadFile, Depends, Request, Response
from fastapi.responses import FileResponse
from enum import Enum, IntEnum

app = FastAPI(debug=False)

@app.on_event("startup")
async def startup_event():
    logging.info('Loading model')
    global model; model = load_model()
    logging.info('Loaded model')


class GarmentType(IntEnum , Enum):
    '''
    [SEG.ID['hair'], SEG.ID['shoes'], SEG.ID['pants'], SEG.ID['upper-clothes'], SEG.ID['hat']]
    '''
    top = SEG.ID['upper-clothes']
    bottom = SEG.ID['pants']
    hair = SEG.ID['hair']
    hat = SEG.ID['hat']
    shoes = SEG.ID['shoes']


@app.post("/dress/")
def segment(response: Response, garment_id:GarmentType = GarmentType.top,
    image_file: UploadFile = File(...),
    garment_file: UploadFile = File(...)):
    """
    Segment image for one person
    **['hat':1, 'hair': 2, 'upper-clothes':4, 'pants': 5, 'shoes':8]
    """
    #TODO max dim size
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    new_image_file = os.path.join(TEMP_DIR, 
        ''.join(random.choice('abcdefgh') for i in range(4))+'_'+ image_file.filename)
    # https://stackoverflow.com/questions/63580229/how-to-save-uploadfile-in-fastapi
    with open(new_image_file, "wb+") as file_object:
        shutil.copyfileobj(image_file.file, file_object)

    new_garment_file = os.path.join(TEMP_DIR, 
        ''.join(random.choice('abcdefgh') for i in range(4))+'_'+ garment_file.filename)
    # https://stackoverflow.com/questions/63580229/how-to-save-uploadfile-in-fastapi
    with open(new_garment_file, "wb+") as file_object:
        shutil.copyfileobj(garment_file.file, file_object)
    
    torch.cuda.set_device(CUDA_DEVICE)
    dress_garment(new_image_file, new_garment_file, GARMENT_IDS=[garment_id])

    # Path(new_image_file).unlink(True)
    img_file = Path('img.jpg')
    return FileResponse(img_file,headers=response.headers, media_type='image/jpeg')

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

if __name__=='__main__':
    uvicorn.run('dressing_processor:app', host="0.0.0.0", port=8000, reload=True, workers =1)
    # global model
    # model = load_model()
    # # print(model)
    # img_file = '/home/deeplab/datasets/custom_fashion/demo_/offi/office_3/20210818_115242.jpg'
    # garment_file = '/home/deeplab/datasets/custom_fashion/demo_/offi/office_4/20210818_115143.jpg'
    # # img_file = '/home/deeplab/datasets/custom_fashion/demo_/2955/29552185/29552185-3.jpg'
    # garment_file = '/home/deeplab/datasets/custom_fashion/demo_/1544/15446515/15446515-1.jpg'
    # dress_garment(img_file, garment_file, [SEG.ID['upper-clothes']])
    # print('end')

