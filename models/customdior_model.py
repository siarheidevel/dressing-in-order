from .dior_base_model import *
import utils.util as functions
from utils.mask import Masks
from models import external_functions
from datasets.custom_dataset import SEG, get_palette
from utils.pose_utils import draw_pose_from_cords,map_to_cord
import copy

PID = [0,4,6,7] # bg, face, arm, leg (the first has to be bg and the second has to be face.)
GID = [2,5,1,3] # hair, top, bottom, jacket

# bg, face, skin, arm, leg (the first has to be bg and the second has to be face.)
PID  = [SEG.ID['background'], SEG.ID['face'], SEG.ID['skin'], SEG.ID['arm'], SEG.ID['leg']] 

# hair, shoes,  bottom, top, hat
GID = [SEG.ID['hair'], SEG.ID['shoes'], SEG.ID['pants'], SEG.ID['upper-clothes'], SEG.ID['hat']]
        
class CustomDIORModel(DIORBaseModel):
    def __init__(self, opt):
        opt.n_human_parts = len(SEG.ID)
        DIORBaseModel.__init__(self, opt)
        self.netE_opt = opt.netE
        self.frozen_flownet = opt.frozen_flownet
        self.random_rate = opt.random_rate
        self.perturb = opt.perturb
       
        if opt.frozen_enc: 
            self.frozen_models += ['E_attr']
            
        if opt.netG in ['adseq2']:
            self.netE_attr.module.reduced = True
        self.warmup = opt.warmup
            
    def modify_commandline_options(parser, is_train):
        DIORBaseModel.modify_commandline_options(parser, is_train)
        parser.add_argument('--flownet_path', type=str, default="", help='')
        parser.add_argument('--loss_coe_seg', type=float, default=0.1, help='coefficient of cls loss')
        parser.add_argument('--loss_coe_context', type=float, default=0, help='coefficient of contextual loss')
        parser.add_argument('--loss_coe_flow_reg', type=float, default=0.001, help='coefficient of cls loss')
        parser.add_argument('--loss_coe_flow_cor', type=float, default=2, help='coefficient of cls loss')
        parser.add_argument('--frozen_flownet', action="store_true", help='coefficient of cls loss')
        parser.add_argument('--frozen_enc', action="store_true", help='coefficient of cls loss')
        parser.add_argument('--perturb', action="store_true", help='coefficient of cls loss')
        parser.add_argument('--warmup', action="store_true", help='coefficient of cls loss')
        parser.set_defaults(n_style_blocks=4)
        parser.set_defaults(random_rate=1)
        return parser

    def _init_loss(self, opt):
        super()._init_loss(opt)
        if self.isTrain:
            self.criterionVGG = external_functions.VGGLoss().to(self.device)
            self.contextSimlilarity =external_functions.ContextSimilarityLoss().to(self.device)
            self.ganFeaturesLoss = external_functions.FeatureMappingsLoss().to(self.device)
            
            self.loss_coe['seg'] = opt.loss_coe_seg
            self.loss_coe['flow_reg'] = 0
            self.loss_coe['flow_cor'] = 0
            if opt.loss_coe_context > 0:
                self.loss_coe['context'] = opt.loss_coe_context
                self.loss_names += ['context']

            if opt.loss_coe_seg > 0:
                self.loss_names += ['seg']
                self.visual_names += ['seg']
                self.criterionCE = nn.BCELoss()

            self.loss_names += ['dpose_features']
            self.loss_names += ['dcontent_features']

            if not opt.frozen_flownet:
                self.loss_coe['flow_reg'] = opt.loss_coe_flow_reg
                self.loss_coe['flow_cor'] = opt.loss_coe_flow_cor
                self.loss_names += ['flow_reg', 'flow_cor']
                self.Correctness = external_functions.PerceptualCorrectness().to(self.device)
                self.Regularization = external_functions.MultiAffineRegularizationLoss(kz_dic={2:5,3:3}).to(self.device)


    def _init_models(self, opt):
        self.model_names = ["E_attr", "G", "VGG", "Flow"]
        self.frozen_models = ["VGG"]
        self.visual_names = ['from_img', 'fake_B', 'to_img','from_pose_view','from_parse_view','to_pose_view','to_parse_view']
        if opt.frozen_flownet:
            self.frozen_models += ["Flow"]
        if opt.frozen_enc:
            self.frozen_models += ["E_attr"]
        self.netVGG = networks.define_tool_networks(tool='vgg', load_ckpt_path="", gpu_ids=opt.gpu_ids)
        # netG
        self.netG = networks.define_G(input_nc=opt.n_kpts, output_nc=3, ngf=opt.ngf, latent_nc=opt.ngf * (2 ** 2), 
                                      style_nc=opt.style_nc,
                                      n_style_blocks=opt.n_style_blocks, n_human_parts=opt.n_human_parts, netG=opt.netG, 
                                      norm=opt.norm_type, relu_type=opt.relu_type,
                                      init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        self.netFlow = networks.define_tool_networks(tool='flownet', load_ckpt_path=opt.flownet_path, gpu_ids=opt.gpu_ids)
        self.netE_attr = networks.define_E(input_nc=3, output_nc=opt.style_nc, netE=opt.netE, ngf=opt.ngf, n_downsample=2,
                                           norm_type=opt.norm_type, relu_type=opt.relu_type, frozen_flownet=opt.frozen_flownet,
                                           init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
     
        if self.isTrain:
            self.model_names += ["D_pose", "D_content"]
            self.netD_pose = networks.define_D(opt.n_kpts+3, opt.ndf, opt.netD,
                                          opt.n_layers_D, norm='none', use_dropout=not opt.no_dropout, 
                                          init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids
                                              )
            self.netD_content = networks.define_D(3+self.n_human_parts, opt.ndf, opt.netD,
                                          n_layers_D=opt.n_layers_D, norm='none', use_dropout=not opt.no_dropout, 
                                          init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

  

    def encode_single_attr(self, img, parse, from_pose, to_pose, i=0, full=False):
        flow, attn = self.netFlow(img, from_pose, to_pose)
        mask = (parse == i).float().unsqueeze(1)
        crop = img * mask
        return self.netE_attr.module.enc_seg(crop, flow[-1], self.netVGG) 

    def encode_attr(self, img, parse, from_pose, to_pose, sid=[]):
        # encode
        self.flow_fields, attn = self.netFlow(img, from_pose, to_pose)
        ret = []
        if not sid:
            sid = range(self.n_human_parts)
        for i in sid:
            mask =  (parse == i).float().unsqueeze(1)
            crop = img * mask#from PIL import Image; Image.fromarray((127*np.transpose(crop[0].cpu().numpy(),(1,2,0)) + 127 ).astype(np.uint8)).save('masked.png')
            fmap, fmask = self.netE_attr.module.enc_seg(crop, self.flow_fields[-1], self.netVGG)           
            ret.append((fmap, fmask))#from PIL import Image; Image.fromarray((254*fmask[0,0].detach().cpu().numpy() ).astype(np.uint8)).save('masked_shape.png')
        return ret

    def perturb_images(self, img):
        _,_,H,W = img.size()
        imgs = []
        for im in img:
            mask =  Masks.get_ff_mask(H,W,BRUSH_WIDTH=0.05, num_v=20, SEGMENT_LENGTH=0.3)#from PIL import Image; Image.fromarray((255*mask).astype(np.uint8)).save('mask.png')
            mask = torch.from_numpy(mask).unsqueeze(0).to(img.device).float()
            imgs += [(im * (1 - mask)).unsqueeze(0)]#from PIL import Image; Image.fromarray((127*np.transpose((im * (1 - mask)).cpu().numpy(),(1,2,0)) + 127 ).astype(np.uint8)).save('masked.png')
        img = torch.cat(imgs)
        return img
    
    def forward(self):
        self.reduce = random.random() > self.random_rate
        if  not self.reduce:
            psegs = self.encode_attr(self.from_img, self.from_parse, self.from_kpt, self.to_kpt, PID)
            gsegs = self.encode_attr(self.from_img, self.from_parse, self.from_kpt, self.to_kpt, GID)
            self.attn = [b for a,b in gsegs] + [b for a,b in psegs] #from PIL import Image; Image.fromarray((254*torch.cat(([a for a in self.attn]),-1)[0,0].detach().cpu().numpy() ).astype(np.uint8)).save('masked_shape.png')
            self.fake_B = self.netG(self.to_kpt, psegs, gsegs)

        else:
            img = self.to_img
            if self.perturb:
                img = self.perturb_images(img)
            self.from_img = img
            self.from_pase = self.to_parse
            self.from_kpt = self.to_kpt
            if self.warmup:
                z = self.netE_attr(img, self.netVGG)
                self.fake_B = self.netG.module.to_rgb(z)
                return 
            else:
                psegs = self.encode_attr(img, self.to_parse, self.to_kpt, self.to_kpt, PID)
                gsegs = self.encode_attr(img, self.to_parse, self.to_kpt, self.to_kpt, GID)
                self.attn = [b for a,b in gsegs] + [b for a,b in psegs]
                self.fake_B = self.netG(self.to_kpt, psegs, gsegs)
                
        self.attn = [functions.upsampling(attn, self.to_kpt.size(2), self.to_kpt.size(3)) for attn in self.attn]
    
    def decode(self, pose, psegs, gsegs):
        return self.netG(pose, psegs, gsegs)

    
    def compute_visuals(self, step, loss_only=False):
        if 'seg' in self.visual_names:
            self.seg = torch.argmax(torch.cat(self.attn,1), 1).detach()
            self.seg = functions.assign_color(self.seg, self.n_human_parts)

        if 'from_pose_view' in self.visual_names:
            self.from_pose_view = self._view_pose(self.from_kpt)
        
        if 'to_pose_view' in self.visual_names:
            self.to_pose_view = self._view_pose(self.to_kpt)

        if 'from_parse_view' in self.visual_names:
            self.from_parse_view = self._view_parse(self.from_parse)

        if 'to_parse_view' in self.visual_names:
            self.to_parse_view = self._view_parse(self.to_parse)        

        super().compute_visuals(step, loss_only)
    
    def _view_pose(self, pose):
        N,C,H,W = pose.size()
        pose_imgs=[]
        for i in range(N):
            coords = map_to_cord(pose[i].cpu().permute(1,2,0).numpy())
            pose_imgs.append(draw_pose_from_cords(coords,(H,W))[0])
        poses= torch.tensor(pose_imgs).div(255.).float().permute(0,3,1,2)
        return poses
    
    def _view_parse(self,parse):
        N,H,W = parse.size()
        parse_view = np.zeros((N,H,W,3))
        colors = np.array(get_palette(len(SEG.ID))).reshape(-1,3)
        for i in range(len(SEG.ID)):
            parse_view[(parse==i).cpu().numpy()]=colors[i]
        parse = torch.tensor(parse_view/255.).float().permute(0,3,1,2)
        return parse
    
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # pose 
        self.loss_D = self.compute_D_pose_loss()
        self.loss_D = self.loss_D + self.compute_D_content_loss()
            

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # GAN loss 
        fake_AB = torch.cat((self.to_kpt, self.fake_B), 1) 
        pred_fake, d_layers_fake = self.netD_pose(fake_AB)
        self.loss_G_GAN_pose = self.criterionGAN(pred_fake, True) *  self.loss_coe['GAN']
        self.loss_G = self.loss_G_GAN_pose

        fake_AB = torch.cat((self.to_parse2, self.fake_B), 1)
        pred_fake, d_layers = self.netD_content(fake_AB)
        self.loss_G_GAN_content = self.criterionGAN(pred_fake, True) * self.loss_coe['GAN']
        self.loss_G = self.loss_G + self.loss_G_GAN_content

        with( torch.no_grad()):
            _, d_layers_real = self.netD_content(torch.cat((self.to_parse2, self.to_img), 1))
        features_loss = self.ganFeaturesLoss(d_layers_real, d_layers_fake)
        self.loss_dcontent_features=features_loss * 0.1

        with( torch.no_grad()):
            _, d_layers_real = self.netD_pose(torch.cat((self.to_kpt, self.to_img), 1))
        features_loss = self.ganFeaturesLoss(d_layers_real, d_layers_fake)
        self.loss_dpose_features=features_loss * 0.1
        self.loss_G = self.loss_G + self.loss_dpose_features + self.loss_dcontent_features
        
        fake_B = self.fake_B
        real_B = self.to_img
        # rec, per, style loss
        self.loss_G = self.loss_G + self.compute_rec_loss(fake_B, real_B)
        self.loss_per, self.loss_sty = self.criterionVGG(real_B, fake_B)
        self.loss_per = self.loss_per * self.loss_coe['per']
        self.loss_sty = self.loss_sty * self.loss_coe['sty']
        self.loss_G = self.loss_G + self.loss_per + self.loss_sty

        #context loss
        if 'context' in self.loss_coe:
            self.loss_context = self.contextSimlilarity(real_B, fake_B) * self.loss_coe['context']
            self.loss_G = self.loss_G + self.loss_context

        # additional loss
        self.loss_G = self.loss_G + self.compute_seg_loss()
        self.loss_G = self.loss_G + self.compute_flow_field_loss()

    def compute_seg_loss(self, GARMENTS=GID):
        
        if not self.loss_coe['seg']:
            return 0.0
        
        self.loss_seg = 0.0
        if len(self.attn) == 4:        
            for i in range(len(GARMENTS)):
                target =  (self.to_parse == GARMENTS[i]).unsqueeze(1).float()
                self.loss_seg = self.loss_seg + self.criterionCE(self.attn[i], target) * self.loss_coe['seg']
        else:
            mylist = GID + PID
            for i in range(len(mylist)):
                target =  (self.to_parse == mylist[i]).unsqueeze(1).float()#from PIL import Image; Image.fromarray((254*torch.cat((self.attn[i], target),-1)[0,0].detach().cpu().numpy() ).astype(np.uint8)).save('masked_shape.png')
                self.loss_seg = self.loss_seg + self.criterionCE(self.attn[i], target) * self.loss_coe['seg']
        return self.loss_seg
    
    def compute_flow_field_loss(self):
        loss = 0.0
        if not self.frozen_flownet:
            self.loss_flow_cor = 0.0
            if self.loss_coe['flow_cor'] > 0:
                # self.loss_flow_cor = self.Correctness(self.to_img, self.from_img, self.flow_fields, [2,3])  * self.loss_coe['flow_cor']
                self.loss_flow_cor = self.Correctness(self.to_img, self.from_img, self.flow_fields, [2,3])  * self.loss_coe['flow_cor']
                loss = loss + self.loss_flow_cor

            self.loss_flow_reg = 0.0
            if self.loss_coe['flow_reg'] > 0:
                self.loss_flow_reg = self.Regularization(self.flow_fields) * self.loss_coe['flow_reg']
                loss = loss + self.loss_flow_reg
        return loss

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        self.loss_D = self.compute_D_pose_loss() + self.compute_D_content_loss()
        
    def compute_D_pose_loss(self):
         # pose 
        fake_AB = torch.cat((self.to_kpt, self.fake_B), 1)  
        pred_fake, _ = self.netD_pose(fake_AB.detach())
        self.loss_D_fake_pose = self.criterionGAN(pred_fake, False) * self.loss_coe['GAN']
        # Real
        real_AB = torch.cat((self.to_kpt, self.to_img), 1)
        pred_real, _ = self.netD_pose(real_AB)
        self.loss_D_real_pose = self.criterionGAN(pred_real, True) * self.loss_coe['GAN']
        return (self.loss_D_fake_pose + self.loss_D_real_pose) / 0.5

    def compute_D_content_loss(self):
        # content
        fake_AB = torch.cat((self.to_parse2, self.fake_B), 1)  
        pred_fake, _ = self.netD_content(fake_AB.detach())
        self.loss_D_fake_content = self.criterionGAN(pred_fake, False) * self.loss_coe['GAN']
        
        real_AB = torch.cat((self.to_parse2, self.to_img), 1)
        pred_real, _ = self.netD_content(real_AB)
        self.loss_D_real_content = self.criterionGAN(pred_real, True)  * self.loss_coe['GAN']
        return (self.loss_D_fake_content + self.loss_D_real_content) / 0.5

    
            
      
        
    