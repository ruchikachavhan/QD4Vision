import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
from voc_utils import nms, selective_search
import numpy as np
import cv2
import torch.optim as optim
from torchmetrics import Accuracy
from voc_utils import ComputeMAP, PlotSamples
import os
from voc_utils import voc_2012_classes as voc_classes

def RCNN(cfg, load_path=None):
  if load_path:
    try:
      print('[*] Attempting to load model from:', load_path)
      loaded = torch.load(load_path)
    except: 
      print('[*] Model does not exist or is corrupted. Creating new model...')
      return _RCNN(cfg)

    # check whether `loaded` is an RCNN instance
    if loaded.__class__.__name__ == '_RCNN':
      return loaded
    else:
      raise ValueError('The loaded tensor is not an instance of _RCNN.')
  else:
    print('[*] Creating model...')
    return _RCNN(cfg)
    
class _RCNN(nn.Module):
  def __init__(self, cfg, device='cuda'):
    super(_RCNN, self).__init__()
    self.num_classes = cfg['n_classes'] 
    self.dobbox_reg = cfg['bbox_reg']
    self.max_proposals = cfg['max_proposals'] # maximum number of regions to extract from given image at inference
    self.image_size = cfg['image_size'] # efficientnet-b0: 224
    self.device = device

    if cfg['network']=='efficientnet-b0':
      self.initialize_weights()

  def inference(self, images, rgb=False, batch_size = 128, apply_nms=True, nms_threshold=0.2): 
    # when given single image
    if type(images) == np.ndarray and len(images.shape)==3: 
      return self.inference_single(images, rgb, batch_size, apply_nms)

    bboxes = []
    for image in tqdm(images, position=0): 
      pred_bboxes = self.inference_single(image, rgb, batch_size, apply_nms, silent_mode=True)
      bboxes.append(pred_bboxes)
    return bboxes

  def inference_single(self, image, rgb=False, batch_size = 128, apply_nms=True, nms_threshold=0.2, silent_mode = False): 
    # image must be loaded in BGR format(cv2.imread) or else rgb must be set to True
    # https://www.pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/
    self.eval()

    if rgb==True: # convert rgb to bgr for selective search
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    preprocess = transforms.Compose([ # preprocess image
      transforms.ToPILImage(),
      transforms.Resize((self.image_size, self.image_size)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # perferm selective search to find region proposals
    print("ss", image.shape)
    rects=selective_search(image)

    proposals=[]
    boxes=[]
    
    for (x, y, w, h) in rects[:self.max_proposals]:
      roi = cv2.cvtColor(image[y:y+h,x:x+w, :], cv2.COLOR_BGR2RGB)
      roi = preprocess(roi)

      proposals.append(roi)
      boxes.append({'x1':x, 'y1':y, 'x2':x + w, 'y2':y + h})

    # convert to DataLoader for batching
    proposals = torch.stack(proposals)
    proposals = torch.Tensor(proposals)
    proposals = torch.utils.data.TensorDataset(proposals)
    proposals = torch.utils.data.DataLoader(proposals, batch_size=batch_size)

    # predict probability of each box
    cnt = 0
    useful_bboxes = []
    for proposal_batch in tqdm(proposals, position= 0, disable = silent_mode):
      patches = proposal_batch[0].to(self.device)

      with torch.no_grad():
        features = self.convnet(patches)
        features = self.flatten(features)
        pred = self.classifier(features)
        pred = torch.nn.functional.softmax(pred, dim=1)

        # if self.dobbox_reg: 
        bbox_refine = self.bbox_reg(features)
        print(bbox_refine.shape)
      useful_idx = torch.where(pred.argmax(1)>0)  # patches which are not classified bg(0)
      print("useful_idx", useful_idx, pred.argmax(1))
      # for idx in useful_idx[0]: # loop through each image
      #   idx = idx.cpu().detach().numpy()
      #   estimate = {}

      #   class_prob = pred[idx].cpu().detach().numpy()
      #   estimate['class'] = class_prob.argmax(0)
      #   estimate['conf'] = class_prob.max(0)

      #   original_bbox = boxes[cnt * batch_size + idx]
      #   if self.dobbox_reg == False:
      #     estimate['bbox'] = original_bbox
      #   else: 
      #     estimate['bbox'] = self.refine_bbox(original_bbox, bbox_refine[idx])
      #     print(estimate)
      #   useful_bboxes.append(estimate)
      # cnt += 1

      # map = mean_average_precision(estimate, )
    
    # apply non-max suppression to remove duplicate boxes
    if apply_nms: 
      useful_bboxes = nms(useful_bboxes, )

    return useful_bboxes

  def refine_bbox(self, bbox, pred): 
      # refine bbox in list of [ {'x1', 'x2', 'y1', 'y2'}, ... ]
      # pred is array of predicted refinements of shape (batch size, 4)
      x, y = (bbox['x1'] + bbox['x2']) / 2, (bbox['y1'] + bbox['y2']) / 2
      w, h = bbox['x2'] - bbox['x1'], bbox['y2'] - bbox['y1']

      newx = x + w * pred[0]
      newy = y + h * pred[1]
      neww = w * torch.exp(pred[2])
      newh = h * torch.exp(pred[3])

      return {'x1': newx - neww/2, 'x2': newx + neww / 2, 'y1': newy - newh/2, 'y2': newy + newh / 2}

  def initialize_weights(self):
      print('[*] Initializing new network...')
      self.effnet = EfficientNet.from_pretrained('efficientnet-b0', num_classes=21).to(self.device)

      self.convnet = self.effnet.extract_features
      self.flatten = nn.Sequential(nn.AvgPool2d(7), nn.Flatten()).to(self.device)
      
      self.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=False), nn.Linear(1280, 21)).to(self.device)

      if self.dobbox_reg:
        self.bbox_reg = nn.Sequential(nn.Dropout(p=0.2, inplace=False), nn.Linear(1280, 4)).to(self.device)

class RCNN_Trainer():
  def __init__(self, model, loader, classifier_dataloader, device, validator=None):
    self.model=model
    self.loader=loader  
    self.classifier_dataloader = classifier_dataloader
    self.validator=validator
    self.device = device

  def classifier_training(self, train_cfg): 
    # if train_cfg['log_wandb']:
    #   wandb.init(project='rcnn_classifier', entity='krenerd77')
    #   wandb.watch(self.model.classifier, log_freq=100)

    cce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    # Gradients for final classifier only
    optimizer = optim.Adam(self.model.classifier.parameters(), lr = train_cfg['lr'], weight_decay=train_cfg['l2_reg'])

    # initialize metrics
    if self.validator is not None: 
      self.validator.initialize(self.model, train_cfg['logging'])
    accuracy_counter = Accuracy(task='multilabel', num_labels=len(voc_classes))


    for epoch in range(train_cfg['epochs']): 
      self.model.train()  # set model to train mode
      print('[*] Training epoch',epoch + 1, '/',train_cfg['epochs'])
      pbar = tqdm(self.classifier_dataloader, position=0, leave=True)
      for step, data in enumerate(pbar):
        optimizer.zero_grad()
        # implement training step -------------------------
        # inference
        features = self.model.convnet(data['image'].to(self.device))
        features = self.model.flatten(features)
        output = self.model.classifier(features)
          
        # backprop
        clf_loss = cce(output, data['label'].to(self.device))
        loss = clf_loss

        loss.backward()
        optimizer.step()
        # logging ------------------------------------------
        acc = accuracy_counter(output.cpu(), data['label'])

        pbar.set_description(f"Loss: {str(loss.cpu().detach().numpy())[:5]}  Accuracy: {str(acc.numpy())[:5]}")

        if train_cfg['log_wandb'] and (step + 1) % 100==0:
          logdict = {}
          logdict['clf_loss'] = clf_loss
          logdict['accuracy'] = acc

        #   wandb.log(logdict)

      # save checkpoints and log
      torch.save(self.model, 'RCNN_checkpoint.pt')
      if self.validator is not None: 
        self.validator.validate() 

  def fine_tuning(self, train_cfg):
    # if train_cfg['log_wandb']:
    #   wandb.init(project='rcnn', entity='krenerd77')
    #   wandb.watch(self.model, log_freq=100)

    cce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    optimizer = optim.Adam(self.model.parameters(), lr = train_cfg['lr'], weight_decay=train_cfg['l2_reg'])

    # lr schedule
    if 'lr_decay' in train_cfg:
      lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_cfg['lr_decay'])
    else: # constant lr
      lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)

    # initialize metrics
    if self.validator is not None: 
      self.validator.initialize(self.model, train_cfg['logging'])
    
    print("NUM CLASSES", len(voc_classes))
    accuracy_counter = Accuracy(task='multilabel', num_labels=len(voc_classes))


    for epoch in range(train_cfg['epochs']): 
      self.model.train()  # set model to train mode
      print('[*] Training epoch',epoch + 1, '/',train_cfg['epochs'])
      pbar = tqdm(self.loader, position=0, leave=True)
      for step, data in enumerate(pbar):
        if step < 10:
          optimizer.zero_grad()
          # implement training step -------------------------
          # inference
          features = self.model.convnet(data['image'].to(self.device))
          features = self.model.flatten(features)
          output = self.model.classifier(features)
          # backprop
          clf_loss = cce(output, data['label'].to(self.device))
          loss = clf_loss
          # bbox regression loss
          # if self.model.dobbox_reg == True:
          bbox_est = self.model.bbox_reg(features)
          # regression targets are described in Appendix C. 
          p_x, p_y, p_w, p_h = data['est_bbox'][0], data['est_bbox'][1], data['est_bbox'][2], data['est_bbox'][3]
          g_x, g_y, g_w, g_h = data['gt_bbox'][0], data['gt_bbox'][1], data['gt_bbox'][2], data['gt_bbox'][3]

          bbox_ans = torch.stack([(g_x - p_x) / p_w, (g_y - p_y) / p_h, torch.log(g_w) / p_w, torch.log(g_h) / p_h], axis = 1)
          bbox_ans = bbox_ans.float().to(self.device)

          # count only images that are not background
          not_bg = (data['label']>0).reshape(len(data['label']), 1).to(self.device)  # mask about whether each image is a background
          bbox_est = bbox_est * not_bg
          bbox_ans = bbox_ans * not_bg

          # add to loss    
          bbox_loss = mse(bbox_est, bbox_ans)
          loss += bbox_loss
            
          loss.backward()
          optimizer.step()
          # logging ------------------------------------------
          # print(output.shape, len(voc_classes), data['label'].shape)
          # acc = accuracy_counter(output.max(1)[1].cpu(), data['label'])
          acc = (output.cpu().argmax(1) == data['label'].cpu()).sum().float() / len(data['label'])
          print(acc)
          pbar.set_description(f"Loss: {str(loss.cpu().detach().numpy())[:5]}  Accuracy: {str(acc)[:5]}")

          # if train_cfg['log_wandb'] and (step + 1) % 100==0:
          logdict = {}
          logdict['clf_loss'] = clf_loss
          logdict['accuracy'] = acc

          #   if self.model.dobbox_reg==True:
          logdict['bbox_loss'] = bbox_loss
        else:
          break
          
        #   wandb.log(logdict)

      # update lr
      lr_schedule.step()
      # save checkpoints and log
      torch.save(self.model, 'RCNN_checkpoint.pt')
      if self.validator is not None: 
        self.validator.validate() 


class Validator():
  def __init__(self, model, val_dataset):
    self.model = model
    self.val_dataset = val_dataset
    self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
  def initialize(self, model, log_keys): 
    self.model = model 

    self.loggers = {}
    self.log_keys = log_keys
    computemap = ComputeMAP()
    for iter, data in enumerate(self.val_loader):
      image = data['image'].squeeze(0).numpy()
      print(image.shape)
      label = data['label']
      bboxes = model.inference(image)
      # print(bboxes)

    # log_funcs = {'map': ComputeMAP, 'plot': PlotSamples}
    # for logtype in log_keys: 
    #   self.loggers[logtype] = log_funcs[logtype]()

  def validate(self):
    self.model.eval()

    # for key in self.log_keys:
    #   self.loggers[key].log(self.model, self.val_dataset)

class ValDataset(torch.utils.data.Dataset):
  # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
  def __init__(self, dataset):
    self.base_dir = dataset.test_dir
    self.dataset = dataset
    self.images = sorted(os.listdir(dataset.test_dir+'/JPEGImages/'))

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    image_path = self.base_dir + '/JPEGImages/' + self.images[idx]
    image = cv2.imread(image_path)

    xml_path = self.base_dir + '/Annotations/' + self.images[idx][:-4]+'.xml'
    gt_bboxes = self.dataset.read_xml(xml_path) # load bboxes in list of dicts {x1, x2, y1, y2, class}

    return  {'image': image, 'label': gt_bboxes}
