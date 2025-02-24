"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').
"""
from pathlib import Path
import time
from options.train_options import TrainOptions
from datasets import create_dataset, create_visual_ds
from utils.visualizers import define_visualizer
from utils.train_utils import *
from models import create_model
import os, torch
from datasets.custom_dataset import PairDataset, SEG, VisualDataset

def generate_val_img(visual_ds, model, opt, step=0,garment_id=SEG.ID['upper-clothes'],  display_mask =False):
    model.eval()
    Visualizer = define_visualizer(opt.model)
    with torch.no_grad():
        # patches = visual_ds.get_patches()
        for cata in visual_ds.selected_keys:
            data = visual_ds.get_attr_visual_input(cata)
            # Visualizer.swap_garment(data, model,  prefix=cata, step=step, gid=5)
            Visualizer.swap_garment(data, model,  prefix=cata, step=step, gid=garment_id,display_mask=display_mask,
                display_body=True)
            print("[visualize] swap garments - %s" % cata)
            #Visualizer.swap_texture(data, patches, model,  prefix=cata, step=step)
            #print("[visualize] swap textures - %s" % cata)
        
        data = visual_ds.get_pose_visual_input("mixed")
        #import pdb; pdb.set_trace()
        Visualizer.swap_pose(data, model,  prefix="mixed", step=step,display_mask=display_mask)
        print("[visualize] swap poses")
    model.train()
        
def main():
    opt = TrainOptions().parse()   # get training options
    if not opt.square: #opt.crop_size >= 250:
        opt.crop_size = (opt.crop_size, max(1,int(opt.crop_size*1.0/256*176)) )
    else:
        opt.crop_size = (opt.crop_size, opt.crop_size)
    opt.square = False
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset = PairDataset(opt)

    # dataset.load_data('/home/deeplab/datasets/deepfashion/diordataset_custom/img_highres/WOMEN/Tees_Tanks/id_00003241/02_2_side.jpg', do_augm=True)
    # '/home/deeplab/datasets/custom_fashion/data/wildberries_ru_/1743/17430312/17430312-2.jpg'
    # dataset.load_data('/home/deeplab/datasets/deepfashion/diordataset_custom/img_highres/WOMEN/Dresses/id_00000621/02_2_side.jpg', do_augm=True)

    dataset = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=(opt.phase == 'train'),
        num_workers=opt.n_cpus, pin_memory=True, prefetch_factor=10)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    # dataset = torch.utils.data.DataLoader(PairDataset(opt), batch_size=opt.batch_size, shuffle=False,
    #     num_workers=opt.n_cpus, pin_memory=True, prefetch_factor=10)
    # for i,data in enumerate(dataset):
    #     # data = next(dataset_iter)
    #     print(i)
    #     if i>10:break
    # visual_ds = create_visual_ds('PairDataset')

    visual_ds = VisualDataset(Path('/home/deeplab/datasets/deepfashion/diordataset_custom/standard_test_anns.txt'), dim = opt.crop_size)
    
    # set up model
    model = create_model(opt)      # create a model given opt.model and other options
    load_iter = model.setup(opt)               # regular setup: load and print networks; create schedulers
    if load_iter != -1:
        opt.epoch_count = load_iter
    total_iters = opt.epoch_count
    print("[init] start from iter %d" % total_iters)
    opt.run_test = not opt.no_trial_test
        
    progressive_steps = {}
    if opt.progressive:
        progressive_steps = get_progressive_training_policy(opt)
        curr_step = max(0, len([i for i in progressive_steps if i<total_iters]) - 1)
        if curr_step < len(progressive_steps):
            keys = list(progressive_steps.keys())
            bs, cs, coe = progressive_steps[keys[curr_step]]
            print("[progressive] init - iter %d, bs: %d, crop: %d" % (total_iters, bs, cs))
            model, dataset, visual_ds = progressive_adjust(model, opt, bs, cs, coe, square=opt.square) 
    
    # generate_val_img(visual_ds, model, opt, step=total_iters,garment_id=SEG.ID['upper-clothes'], display_mask=True)    
                
    # total_iters =-1
    # train
    epoch_start_time = time.time()  # timer for entire epoch
    while total_iters < opt.n_epochs + opt.n_epochs_decay + 1: 
        for data in dataset:  # inner loop within one epoch
            total_iters += 1
            if total_iters > opt.n_epochs + opt.n_epochs_decay + 1:
                break
                
            
            # model update
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            # generate_val_img(visual_ds, model, opt, step=total_iters) 
            # progressive adjust
            if opt.progressive and (total_iters - 1) in progressive_steps:
                bs, cs, coe = progressive_steps[total_iters - 1]
                print("at total_iter %d, bs: %d, crop: %d" % (total_iters, bs, cs))
                model, dataset, visual_ds = progressive_adjust(model, opt, bs, cs, coe, square=opt.square) 
                break
                
            # log
            if total_iters % opt.print_freq == 0:
                losses = model.get_cum_losses()
                #t_comp = (time.time() - epoch_start_time) / opt.batch_size
                out_string = "[%s][iter-%d]" % (opt.name,  total_iters)
                for loss_name in losses:
                    out_string += "%s: %.4f, " % (loss_name, losses[loss_name])
                print(out_string)

            
                        # save latest ckpt  
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (total_iters %d)' % (total_iters))
                model.save_networks('latest', total_iters)
                print('End of iter %d / %d \t Time Taken: %d sec' % (total_iters, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
            
            # save periodic ckpt
            if total_iters % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of iters %d' % (total_iters))
                save_suffix = 'iter_%d' % total_iters 
                model.save_networks(save_suffix)

            
            # tensorboard
            if total_iters % opt.display_freq == 0:   #
                model.compute_visuals(total_iters, loss_only=False)
                if opt.run_test:
                    generate_val_img(visual_ds, model, opt, step=total_iters, display_mask =True)
                print("at %d, compute visuals" % total_iters)                
           

            
            
                
            # update learning rate
            if total_iters % opt.lr_update_unit == 0:
                print(total_iters)
                model.update_learning_rate()                     # update learning rates at the end of every iteration.
                

    model.save_networks('latest', total_iters)


if __name__ == "__main__":
    main()
            

        
           
            
