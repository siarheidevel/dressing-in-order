import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import imageio, os
from datasets.custom_dataset import SEG, get_palette
import utils.util as functions
GID = [2,5,1,3]
PID = [0,4,6,7]
# bg, face, skin, arm, leg (the first has to be bg and the second has to be face.)
PID  = [SEG.ID['background'], SEG.ID['face'], SEG.ID['skin'], SEG.ID['arm'], SEG.ID['leg']] 

# hair, shoes, top, bottom, hat
GID = [SEG.ID['hair'], SEG.ID['shoes'], SEG.ID['pants'], SEG.ID['upper-clothes'], SEG.ID['hat']]


def define_visualizer(model_name):
    return FlowVisualizer 

    
class FlowVisualizer:
        
    @staticmethod
    def swap_garment(data, model, gid=5, display_mask=False, display_body = False, step=-1, prefix=""):
        model.eval()
        #display_mask = True # display_mask and 'seg' in model.visual_names
        imgs, parses, poses = data
        imgs = imgs.to(model.device)
        parses = parses.float().to(model.device)
        poses = poses.float().to(model.device)
        
        # all_mask = [torch.zeros(imgs.size())[None].to(model.device)]
        all_mask = [(parses.unsqueeze(1) == gid).float().expand_as(imgs).to(model.device).unsqueeze(0)]
        all_fake = [imgs[None]]
        all_body = [torch.zeros_like(imgs[None])]
        # import pdb; pdb.set_trace()
        for img, parse, pose in zip(imgs, parses, poses):
            curr_to_pose = pose[None].expand_as(poses)
            seg = model.encode_single_attr(imgs, parses, poses, curr_to_pose, gid)
            gsegs = model.encode_attr(img[None].expand_as(imgs),
                                            parse[None].expand_as(parses),
                                            curr_to_pose,
                                            curr_to_pose,
                                            GID)
            psegs = model.encode_attr(img[None].expand_as(imgs),
                                            parse[None].expand_as(parses),
                                            curr_to_pose,
                                            curr_to_pose,
                                            PID)
            
            gsegs[GID.index(gid)] = seg
            
            fake_img = model.decode(curr_to_pose, psegs, gsegs) #, attns)
            all_fake += [fake_img[None]]
            if display_mask:
                # import pdb; pdb.set_trace()
                N,C,H,W = imgs.size()
                segm_m = [functions.upsampling(attn, H, W) for attn in [b for a,b in gsegs] + [b for a,b in psegs]]
                segm_max = torch.argmax(torch.cat(segm_m,1), 1)
                segm_color =functions.assign_color(segm_max, model.n_human_parts)
                all_mask += [segm_color[None]]
                # all_mask += [model.get_seg_visual(gid).expand(N,3,H,W)[None]]
            
            if display_body:
                body_img = model.netG.module.to_rgb(model.netG.module.base.detach()).detach()
                all_body += [body_img[None]]


        # display
        all_fake = torch.cat(all_fake)
        _,_,H,W = fake_img.size()
        all_fake[0] = F.interpolate(all_fake[0], (H,W))
        #all_mask[0] = F.interpolate(all_mask[0], (H,W))
        if display_mask:
            all_mask = torch.cat(all_mask)
        if display_body:
            all_body = torch.cat(all_body)
        ret = []
        for i in range(all_fake.size(1)):
            if display_mask and display_body:
                print_img = torch.cat([all_fake[:,i], all_body[:,i], all_mask[:,i]],2)
            elif display_mask:
                print_img = torch.cat([all_fake[:,i], all_mask[:,i]],2)
            elif display_body:
                print_img = torch.cat([all_fake[:,i], all_body[:,i]],2)
            else:
                print_img = all_fake[:, i]
            ret.append(print_img)
        print_img = torch.cat(ret, 2)
        #print_img = (all_fake[:,i] + 1) / 2

        print_img = (print_img + 1) / 2
        print_img = print_img.float().cpu().detach()
        curr_step = step if step >= 0 else i
        if step >= 0:
            curr_step = step
            model.writer.add_images("swap garment %d %s" % (i, prefix), print_img, curr_step)
        else:
            model.writer.add_images("swap garment %s" % (prefix), print_img, i)
        model.train()
        
    @staticmethod
    def swap_pose(data, model, step=-1, prefix="", display_mask=False):
        model.eval()
        imgs, parses, from_poses, poses = data
        imgs = imgs.to(model.device)
        #_,_,H,W = imgs.size()
        parses = parses.float().to(model.device)
        poses = [pose.float().to(model.device) for pose in poses]
        from_poses = from_poses.float().to(model.device)
        N = poses[0].size(0)
        _, H, W = parses.size()
        
        for i, (img, parse, from_pose, pose) in enumerate(zip(imgs, parses, from_poses, poses)):
            # import pdb; pdb.set_trace()
            curr_img = img.expand(N,3,H,W)
            curr_parse = parse.expand(N,H,W)
            curr_from_pose = from_pose.expand(N, 18, H, W)
            psegs = model.encode_attr(curr_img, curr_parse, curr_from_pose, pose, PID)
            gsegs = model.encode_attr(curr_img, curr_parse, curr_from_pose, pose, GID)
            
            fake = model.decode(pose, psegs, gsegs) #, attns)
            _,_,h,w = fake.size()
            pivot_img = F.interpolate(img[None], (h,w))
            print_img = (torch.cat([pivot_img, fake]) + 1) / 2
            print_img = print_img.float().cpu().detach()
            if display_mask:
                # import pdb; pdb.set_trace()
                N,C,H,W = fake.size()
                segm_m = [functions.upsampling(attn, H, W) for attn in [b for a,b in gsegs] + [b for a,b in psegs]]
                segm_max = torch.argmax(torch.cat(segm_m,1), 1)
                segm_max=torch.cat((parse[None],segm_max),0)
                segm_color =functions.assign_color(segm_max, model.n_human_parts).cpu()
                print_img = torch.cat((print_img,segm_color),2)
                
            if step >= 0:
                curr_step = step
                model.writer.add_images("swap pose %d %s" % (i, prefix), print_img, curr_step)
            else:
                model.writer.add_images("swap pose %s" % (prefix), print_img, i)
        model.train()
        
    
        
            