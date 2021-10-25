from pathlib import Path
from posixpath import join
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch
import copy, os, collections
import json
import math, random
# from .human_parse_labels import get_label_map, DF_LABEL, YF_LABEL
import pandas as pd
from utils import pose_utils


class SEG:
    ID = {'background': 0, 'hat':1, 'hair': 2,  'face': 3, 'upper-clothes':4,
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

    seg3_original = {'background': 0, 'hat':1, 'hair': 2,  'face': 3, 'upper-clothes':4,
               'pants': 5, 'arm': 6, 'leg': 7, 'shoes':8, 'skin':9, 'glove':10 }
    
    seg3_to_labels = {'background':'background','hat':'hat','hair':'hair','face':'face',
                      'upper-clothes':'upper-clothes',
                      'pants':'pants','arm':'arm','leg':'leg','shoes':'shoes',
                      'skin':'skin','glove':'background'}


    # PID = [0,4,6,7] # bg, face, arm, leg (the first has to be bg and the second has to be face.)
    # GID = [2,5,1,3] # hair, top, bottom, jacket
    # my params

    # bg, face, skin, arm, leg (the first has to be bg and the second has to be face.)
    PERSON_IDS = [ID['background'], ID['face'], ID['skin'], ID['arm'], ID['leg']] 

    # hair, shoes, top, bottom, hat
    GARMENT_IDS = [ID['hair'], ID['shoes'], ID['pants'], ID['upper-clothes'], ID['hat']]


class PairDataset(data.Dataset):

    def __init__(self, opt) -> None:
        super().__init__()
        self.dataroot = opt.dataroot
        self.crop_size = opt.crop_size
        annotation_index_csv = os.path.join(self.dataroot, 'annotation_index_qanet.csv')
        annotation_pairs_csv = os.path.join(self.dataroot, 'annotation_pairs_qanet.csv')
        # annotation_index_csv = os.path.join(self.dataroot, 'annotation_index.csv')
        # annotation_pairs_csv = os.path.join(self.dataroot, 'annotation_pairs.csv')
        self.do_flip=True

        # group;pair1;pair2
        self.pairs_df = pd.read_csv(annotation_pairs_csv, sep=';')
        # 'image_file', 'image_group', 'keypoints_y', 'keypoints_x', 'img_height', 'img_width', 'gender', 'category'
        self.index_df = pd.read_csv(annotation_index_csv, sep=';')
        self.index_df=self.index_df.set_index('image_file')
        self.index_df['image_file']=self.index_df.index

        transform_list=[]
        # transform_list.append(transforms.Resize(size=self.load_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
        self.transforms = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.pairs_df)
    
    def __getitem__(self, index):
        from_name, to_name = self.pairs_df.iloc[index][['from','to']].to_list()
        from_img, from_kpt, from_parse = self.load_data(from_name, do_augm=True)
        to_img, to_kpt, to_parse = self.load_data(to_name, do_augm=True)
        if self.do_flip:
            make_flip = random.random() > 0.5
            if make_flip:
                from_img = torch.flip(from_img, [2])
                from_parse = torch.flip(from_parse, [1])
                from_kpt = torch.flip(from_kpt, [2])[[0,1,5,6,7,2,3,4,11,12,13,8,9,10,15,14,17,16],...]

                to_img = torch.flip(to_img, [2])
                to_parse = torch.flip(to_parse, [1])
                to_kpt = torch.flip(to_kpt, [2])[[0,1,5,6,7,2,3,4,11,12,13,8,9,10,15,14,17,16],...]
        # Image.fromarray((np.transpose(from_img,(1,2,0)).numpy()*127+127).astype(np.uint8),'RGB').save('image.jpg');Image.fromarray(from_parse.numpy()*20).save('maskk.png')
        # Image.fromarray((from_parse.numpy()*20).astype(np.uint8)).save('seg.png')
        # Image.fromarray(_view_pose(from_kpt)).save('pose.png')
        # Image.fromarray((from_img.permute([1,2,0]).numpy()*127 + 127).astype(np.uint8)).save('img.png')

        return from_img, from_kpt, from_parse, to_img, to_kpt, to_parse, index #torch.Tensor([0])


    def load_data(self, name, do_augm=False):
        seg = np.array(Image.open(name + ".seg_qanet.render.png"))
        img = cv2.imread(name)[:,:,[2,1,0]]
        pose_y_str,pose_x_str = self.index_df.loc[name][['keypoints_y', 'keypoints_x']].to_list()
        pose_array = pose_utils.load_pose_cords_from_strings(pose_y_str, pose_x_str)
        affine_transform = PairDataset._get_affine_stransform(self.crop_size, img.shape[0],img.shape[1], do_augm)

        dstSize = (self.crop_size[1],self.crop_size[0])
        resized_img = cv2.bilateralFilter(cv2.warpAffine(img, affine_transform[:2],dstSize, 
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)),d=5,sigmaColor=15, sigmaSpace=15).astype(np.uint8)
        Image.fromarray(resized_img).save('img.png')

        # Image.fromarray(cv2.bilateralFilter(cv2.warpAffine(img, affine_transform[:2],dstSize, 
        #     flags=cv2.INTER_AREA,
        #     borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)),d=5,sigmaColor=11, sigmaSpace=24).astype(np.uint8),'RGB').save('image3.jpg')
        # Image.fromarray(img).resize((352,512), Image.LANCZOS).save('image2.jpg')
        # Image.fromarray(cv2.warpAffine(img, affine_transform[:2],dstSize, 
        #         flags=cv2.INTER_CUBIC,
        #         borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))).save('image.jpg')
        
        #Image.fromarray(img).transform((176,256), Image.AFFINE, affine_transform[:2].reshape(-1).tolist(),resample=Image.BILINEAR).save('img.png')
        # Image.fromarray(img).resize((176,256), Image.ANTIALIAS).save('img.png') 
        # Image.fromarray(cv2.warpAffine(cv2.blur(img,(2,3)), affine_transform[:2],dstSize, 
        #     flags=cv2.INTER_AREA,
        #     borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))).save('img.png')
        #TODO remove white noise in segm: erode->dilate
        resized_seg = cv2.warpAffine(seg, affine_transform[:2], dstSize,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue = SEG.ID['background'])
        redused_seg = PairDataset._reduce_segmentation(resized_seg)     
        # mask_image=Image.fromarray(resized_seg);mask_image.putpalette(get_palette(len(SEG.ID)));mask_image.save('seg.png')
        

        pose = pose_utils.cords_to_map(pose_array, (self.crop_size[0], self.crop_size[1]), affine_matrix=affine_transform, sigma=5)
        # resized_pose = cv2.warpAffine(pose, affine_transform[:2], dstSize,
        #     flags=cv2.INTER_AREA,
        #     borderMode=cv2.BORDER_CONSTANT, borderValue = 0)
        # cv2.imwrite('pose.jpg',np.sum(pose, axis=-1)*127)
        pose = np.transpose(pose,(2, 0, 1))
        # resized_img = np.transpose(resized_img,(2, 0, 1))


        # augmenation remove bg 
        resized_img =_remove_bg(resized_img, (redused_seg==SEG.ID['background']).astype(np.uint8))

        img_tensor, pose_tensor, seg_tensor = self.transforms(resized_img), torch.Tensor(pose), torch.Tensor(redused_seg)
        # Image.fromarray((img_tensor.permute([1,2,0]).numpy()*127 + 127).astype(np.uint8)).save('img.png')
        # # Image.fromarray((torch.flip(img_tensor, [2]).permute([1,2,0]).numpy()*127 + 127).astype(np.uint8)).save('img_flip.png')
        # Image.fromarray((seg_tensor.numpy()*20).astype(np.uint8)).save('seg.png')
        # # Image.fromarray(( torch.flip(seg_tensor, [1]).numpy()*20).astype(np.uint8)).save('seg_flip.png')
        # Image.fromarray(_view_pose(pose_tensor)).save('pose.png')
        # # Image.fromarray(_view_pose(torch.flip(pose_tensor, [2])[[0,1,5,6,7,2,3,4,11,12,13,8,9,10,15,14,17,16],...])).save('pose_flip.png')

        return img_tensor, pose_tensor, seg_tensor

    @staticmethod
    def _reduce_segmentation(seg):
        texture_mask = copy.deepcopy(seg)
        for label,id in SEG.seg3_original.items():
            new_id = SEG.ID.get(SEG.seg3_to_labels.get(label,''),0)
            texture_mask[seg == id] = new_id
        return texture_mask
        
    @staticmethod
    def _get_affine_stransform(crop_size, img_height, img_width, do_augm = False):
        fit_height, fit_width = crop_size[0], crop_size[1]
        center = img_height * 0.5 + 0.5, img_width * 0.5 + 0.5
        do_flip, angle, shift, scale = 0, 0, [0,0], 1
        # heavy augmentation
        if do_augm:
            do_flip, angle, shift, scale = 0, 0, [np.random.uniform(low=-0.05, high=0.05),0], np.random.uniform(low=0.8, high=1.2) 
            # do_flip, angle, shift, scale = 0, 0, [0.1,0], 1.2       
        affine_matrix = PairDataset.get_affine_matrix(center=center, fit=(fit_height, fit_width), angle=angle, translate=shift,
                scale=scale, flip=do_flip)
        return affine_matrix


    @staticmethod
    def get_affine_matrix(center, fit, angle, translate, scale,  flip):
        if flip:
            # https://stackoverflow.com/questions/57863376/why-i-cannot-flip-an-image-with-opencv-affine-transform
            M_x_flip=np.float32([[-1, 0, 2*center[1]-1], [0, 1, 0]])
        else:
            M_x_flip=np.float32([[1, 0, 0], [0, 1, 0]])

        fit_scale = max(fit[0]/(2*center[0]), fit[1]/(2*center[1]))
        FIT_scale = np.float32([[fit_scale, 0, 0], [0, fit_scale, 0]])
        
        M_scale = np.float32([[scale, 0, 0], [0, scale, 0]])
        # 0.02 fix coef for bad segmentation
        translate[1] = (1-2* fit_scale* center[0]*scale / fit[0])+0.02
        M_translate = np.float32([[1, 0, translate[0]*fit[1]], [0, 1, translate[1]*fit[0]]])
        # M_translate = np.float32([[1, 0, -(center[1]*2*fit_scale - fit[1])*scale], [0, 1, (1-scale) * fit[0]]])

        rads = math.radians(angle)
        cos, sin = math.cos(rads), math.sin(rads)
        M_rotate = np.float32([[cos, -sin, fit[0]*(1-cos) - fit[1]*sin], 
            [sin, cos, fit[1]*(1-cos) + fit[0]*sin]])
        M_rotate = cv2.getRotationMatrix2D((fit[0]/2,fit[1]/2), angle, 1.0)
        M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)

        M_x_flip = np.vstack((M_x_flip,[0,0,1]))
        M_rotate = np.vstack((M_rotate,[0,0,1]))
        M_translate = np.vstack((M_translate,[0,0,1]))
        M_scale = np.vstack((M_scale,[0,0,1]))
        FIT_scale = np.vstack((FIT_scale,[0,0,1]))

        return M_translate @ M_scale  @ FIT_scale @ M_rotate @ M_x_flip


class VisualDataset(data.Dataset):

    def __init__(self, eval_file: Path, dim=(256,176)):
        super().__init__()
        self.dataroot = eval_file.parent
        self._load_visual_anns(eval_file)
        self.selected_keys = [ "gfla", "jacket", "lace", "pattern", "plaid", "plain", "print", "strip", "flower"]
        # DFPairDataset.__init__(self, dataroot, dim, isTrain, n_human_part=n_human_part)
        # load standard pose
        self.crop_size = dim
        transform_list=[]
        # transform_list.append(transforms.Resize(size=self.load_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
        self.transforms = transforms.Compose(transform_list)
        self._load_standard_pose()
        
        #load standard patches
        #patch_root = "/".join(dataroot.split("/")[:-1])
        #self.standard_patches = [self._load_img(os.path.join(patch_root, "dtd/images", fn)).unsqueeze(0) for fn in TEST_PATCHES]
        #self.standard_patches = torch.cat(self.standard_patches)
        
        

    def _load_standard_pose(self):
        self.standard_poses = []
        for key in self.pose_keys:
            curr_from, curr_from_kpt, curr_parse = self.load_data(str(self.dataroot/key))
            self.standard_poses.append(curr_from_kpt.unsqueeze(0))
        # self.standard_poses = [
        #    self._load_kpt(key).unsqueeze(0) for key in self.pose_keys
        #   ]
        self.standard_poses = torch.cat(self.standard_poses)

    def get_patches(self):
        return self.standard_patches

    def __len__(self):
        return sum([len(self.attr_keys[i]) for i in self.attr_keys])

    
    def _load_visual_anns(self, eval_anns_path):
        with open(eval_anns_path) as f:
            raw_anns = f.readlines()
        pose_cnt = 1
        self.pose_keys = []
        for line in raw_anns[1:]:
            if line.startswith("attr"):
                break
            self.pose_keys.append(line[:-1])
            pose_cnt += 1
        self.attr_keys = collections.defaultdict(list)
        #import pdb; pdb.set_trace()
        for line in raw_anns[pose_cnt+1:]:
            category, key = line[:-1].split(", ")
            self.attr_keys[category].append(key)
        mixed = []
        for category in ['flower','plaid','print','strip']:
            mixed.append(self.attr_keys[category][0])
        self.attr_keys["mixed"] = mixed

    def get_patch_input(self):
        return torch.cat(self.standard_patches)

    def get_all_pose(self, key, std_pose=True):
        if std_pose:
            return self.standard_poses
        # folder_path = os.path.join(self.kpt_dir,key).split("/")
        # prefix = folder_path[-1]
        # folder_path = "/".join(folder_path[:-1])
        # ret = []
        
        # for fn in os.listdir(folder_path):
        #     if fn.startswith(prefix) and fn.endswith('_kpt.npy'):
        #         curr = self._load_kpt(os.path.join(folder_path, fn[:-8]))   
        #         ret.append(curr[None])
            
        # if len(ret) < 2:
        #     return self.standard_poses
        # return torch.cat(ret)

    
    
        
        
    def get_pose_visual_input(self, subset="plain", std_pose=True):
        keys = self.attr_keys[subset]
        keys = keys[:min(len(keys), 8)]
        all_froms, all_kpts, all_parses = [], [], []
        all_from_kpts = []
        for key in keys:
            curr_key = key# + view_postfix
            curr_from, curr_from_kpt, curr_parse = self.load_data(str(self.dataroot/curr_key))
            all_from_kpts += [curr_from_kpt.unsqueeze(0)]
            curr_kpt = self.get_all_pose(key, std_pose=std_pose)
            all_kpts += [curr_kpt]
            all_froms += [curr_from.unsqueeze(0)]
            all_parses += [curr_parse.unsqueeze(0)]
        all_froms = torch.cat(all_froms)
        all_parses = torch.cat(all_parses)
        all_from_kpts = torch.cat(all_from_kpts)
        return all_froms, all_parses, all_from_kpts, all_kpts #self.standard_poses

    def get_attr_visual_input(self, subset="plain",view_postfix="_1_front"):
        keys = self.attr_keys[subset]
        keys = keys[:min(len(keys), 4)]
        all_froms, all_parses, all_kpts = [], [], []
        for key in keys:
            curr_key = key# + view_postfix
            curr_from, to_kpt, curr_parse = self.load_data(str(self.dataroot/curr_key))
            all_froms += [curr_from.unsqueeze(0)]
            all_parses += [curr_parse.unsqueeze(0)]
            all_kpts += [to_kpt.unsqueeze(0)]
        all_froms = torch.cat(all_froms)
        all_parses = torch.cat(all_parses)
        all_kpts = torch.cat(all_kpts)
        return all_froms, all_parses, all_kpts 

    def get_inputs_by_key(self, key):
        #keys = self.attr_keys[subset]
        #keys = keys[:min(len(keys), 4)]
        curr_from, to_kpt, curr_parse = self.load_data(key)
        return curr_from, curr_parse, to_kpt


    def load_data(self, name):
        seg = np.array(Image.open(name + ".seg_qanet.render.png"))
        img = cv2.imread(name)[:,:,[2,1,0]]
        with open(name + ".pose2.txt",'r') as f:
            line = f.read().split('\t')
            pose_y_str,pose_x_str = line[0],line[1]
        # pose_y_str,pose_x_str = self.index_df.loc[name][['keypoints_y', 'keypoints_x']].to_list()
        # pose_y_str,pose_x_str = self.index_df.loc[name][['keypoints_y', 'keypoints_x']].to_list()
        pose_array = pose_utils.load_pose_cords_from_strings(pose_y_str, pose_x_str)
        affine_transform = PairDataset._get_affine_stransform(self.crop_size, img.shape[0],img.shape[1])

        dstSize = (self.crop_size[1],self.crop_size[0])
        resized_img = cv2.bilateralFilter(cv2.warpAffine(img, affine_transform[:2],dstSize, 
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)),d=5,sigmaColor=15, sigmaSpace=15).astype(np.uint8)
        
        # resized_img = cv2.warpAffine(cv2.blur(img,(3,3)), affine_transform[:2],dstSize, 
        #     flags=cv2.INTER_LINEAR,
        #     borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        #Image.fromarray(img).transform((176,256), Image.AFFINE, affine_transform[:2].reshape(-1).tolist(),resample=Image.BILINEAR).save('img.png')
        # Image.fromarray(img).resize((176,256), Image.ANTIALIAS).save('img.png') 
        # Image.fromarray(cv2.warpAffine(cv2.blur(img,(2,3)), affine_transform[:2],dstSize, 
        #     flags=cv2.INTER_AREA,
        #     borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))).save('img.png')
        #TODO remove white noise in segm: erode->dilate
        resized_seg = cv2.warpAffine(seg, affine_transform[:2], dstSize,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue = SEG.ID['background'])
        redused_seg = PairDataset._reduce_segmentation(resized_seg)     
        # mask_image=Image.fromarray(resized_seg);mask_image.putpalette(get_palette(len(SEG.ID)));mask_image.save('seg.png')
        

        pose = pose_utils.cords_to_map(pose_array, (self.crop_size[0], self.crop_size[1]), affine_matrix=affine_transform, sigma=5)
        # resized_pose = cv2.warpAffine(pose, affine_transform[:2], dstSize,
        #     flags=cv2.INTER_AREA,
        #     borderMode=cv2.BORDER_CONSTANT, borderValue = 0)
        # cv2.imwrite('pose.jpg',np.sum(pose, axis=-1)*127)
        pose = np.transpose(pose,(2, 0, 1))
        # resized_img = np.transpose(resized_img,(2, 0, 1))

        # augmenation remove bg 
        resized_img =_remove_bg(resized_img, (redused_seg==SEG.ID['background']).astype(np.uint8))

        img_tensor, pose_tensor, seg_tensor = self.transforms(resized_img), torch.Tensor(pose), torch.Tensor(redused_seg)
        # Image.fromarray((img_tensor.permute([1,2,0]).numpy()*127 + 127).astype(np.uint8)).save('img.png')
        # Image.fromarray((torch.flip(img_tensor, [2]).permute([1,2,0]).numpy()*127 + 127).astype(np.uint8)).save('img_flip.png')
        # Image.fromarray((seg_tensor.numpy()*20).astype(np.uint8)).save('seg.png')
        # Image.fromarray(( torch.flip(seg_tensor, [1]).numpy()*20).astype(np.uint8)).save('seg_flip.png')
        # Image.fromarray(_view_pose(pose_tensor)).save('pose.png')
        # Image.fromarray(_view_pose(torch.flip(pose_tensor, [2])[[0,1,5,6,7,2,3,4,11,12,13,8,9,10,15,14,17,16],...])).save('pose_flip.png')
        return img_tensor, pose_tensor, seg_tensor


    
def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def _view_pose(pose):
    from utils.pose_utils import draw_pose_from_cords,map_to_cord
    C,H,W = pose.size()
    coords = map_to_cord(pose.cpu().permute(1,2,0).numpy())
    pose_img = draw_pose_from_cords(coords,(H,W))[0]
    return pose_img


def _remove_bg(img, bg_mask, bg_color = [255,249,249]):
    bg_kernel = (3, 3)
    # bg_color = [234,230,224]    
    eroded_mask = cv2.dilate(bg_mask,np.ones(bg_kernel),iterations=1) #Image.fromarray((eroded_mask*255).astype(np.uint8)).save('mask.png')
    eroded_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_OPEN, np.ones(bg_kernel, np.uint8))
    blurred_mask = cv2.GaussianBlur(eroded_mask.astype(np.float),bg_kernel,0) #Image.fromarray((blurred_mask*255).astype(np.uint8)).save('mask.png')
    # blurred_img = cv2.blur(img.astype(np.float),(5,5),0)  #Image.fromarray((blurred_img).astype(np.uint8)).save('mask.png')
    bg_img = np.ones_like(img)*bg_color
    final_img = img * (1-blurred_mask[...,None]) + blurred_mask[...,None] * bg_img
    # Image.fromarray((final_img ).astype(np.uint8)).save('mask.png')
    return final_img.astype(np.uint8)
