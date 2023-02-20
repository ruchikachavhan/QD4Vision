import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.models as models
import torch.optim as optim
import numpy as np
import torchvision.datasets as datasets
import random
import os
import urllib.request
# from skimage import io
import shutil
from tqdm import tqdm
import tarfile
import requests
import xml.etree.ElementTree as ET
import cv2
import wandb
import pickle
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image


voc_2012_classes=['background','Aeroplane',"Bicycle",'Bird',"Boat","Bottle","Bus","Car","Cat","Chair",'Cow',"Dining table","Dog","Horse","Motorbike",'Person', "Potted plant",'Sheep',"Sofa","Train","TV/monitor"]
invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],std = [ 1., 1., 1. ]), 
                                transforms.ToPILImage() ])

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

class VOCDataset():
  def __init__(self):
    self.train_dir=None
    self.test_dir=None
    self.trainDataLink=None
    self.testDataLink=None

    self.common_init()

  def common_init(self):
    # init that must be shared among all subclasses of this method
    self.label_type=['none','aeroplane',"Bicycle",'bird',"Boat","Bottle","Bus","Car","Cat","Chair",'cow',"Diningtable","Dog","Horse","Motorbike",'person', "Pottedplant",'sheep',"Sofa","Train","TVmonitor"]
    self.convert_id=['background','Aeroplane',"Bicycle",'Bird',"Boat","Bottle","Bus","Car","Cat","Chair",'Cow',"Dining table","Dog","Horse","Motorbike",'Person', "Potted plant",'Sheep',"Sofa","Train","TV/monitor"]
    self.convert_labels={}
    for idx, x in enumerate(self.label_type):
      self.convert_labels[x.lower()]=idx

    self.num_classes=len(self.label_type) # 20 + 1(none)

  def download_dataset(self, validation_size=5000):
    # download voc train dataset
    print('[*] Downloading dataset...')
    print(self.trainDataLink)
    urllib.request.urlretrieve(self.trainDataLink, 'voctrain.tar')

    print('[*] Extracting dataset...')
    tar = tarfile.open('voctrain.tar', "r:")
    tar.extractall('/content/VOCtrain')
    tar.close()
    os.remove('/content/voctrain.tar')

    if self.testDataLink is None: 
      # move 5K images to validation set
      print('[*] Moving validation data...')
      ensure_dir(self.test_dir+'/Annotations/')
      ensure_dir(self.test_dir+'/JPEGImages/')

      random.seed(42)
      val_images = random.sample(sorted(os.listdir(self.train_dir + '/JPEGImages')), validation_size)

      for path in val_images:
        img_name = path.split('/')[-1].split('.')[0]
        # move image
        os.rename(self.train_dir+'/JPEGImages/'+img_name+'.jpg', self.test_dir+'/JPEGImages/'+img_name+'.jpg')
        # move annotation
        os.rename(self.train_dir+'/Annotations/'+img_name+'.xml', self.test_dir+'/Annotations/'+img_name+'.xml')
    else: 
      # Load from val data
      print('[*] Downloading validation dataset...')
      urllib.request.urlretrieve(self.testDataLink, 'voctest.tar')

      print('[*] Extracting validation dataset...')
      tar = tarfile.open('voctest.tar', "r:")
      tar.extractall('/content/VOCtest')
      tar.close()
      os.remove('/content/voctest.tar')

  def read_xml(self, xml_path): 
    object_list=[]

    tree = ET.parse(open(xml_path, 'r'))
    root=tree.getroot()
  
    objects = root.findall("object")
    for _object in objects:
      name = _object.find("name").text
      bndbox = _object.find("bndbox")
      xmin = int(float(bndbox.find("xmin").text))
      ymin = int(float(bndbox.find("ymin").text))
      xmax = int(float(bndbox.find("xmax").text))
      ymax = int(float(bndbox.find("ymax").text))
      class_name = _object.find('name').text
      object_list.append({'x1':xmin, 'x2':xmax, 'y1':ymin, 'y2':ymax, 'class': self.convert_labels[class_name]})

    return object_list

class VOC2007(VOCDataset):
  def __init__(self):
    self.train_dir='../../TestDatasets/voc/VOCdevkit/VOC2007'
    self.test_dir='../../TestDatasets/voc/VOCdevkit/VOC2007'
    self.trainDataLink='http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar'
    self.testDataLink='http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar'
    self.common_init()#mandatory
    
class VOC2012(VOCDataset):
  def __init__(self):
    self.train_dir='../TestDatasets/voc/VOCdevkit/VOC2012'
    self.test_dir='../TestDatasets/voc/VOCdevkit/VOC2012'
    # original site goes down frequently, so we use a link to the clone alternatively
    # self.trainDataLink='http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar' 
    self.trainDataLink = 'http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar'
    self.testDataLink=None
    self.common_init()#mandatory

def calculate_IoU(bb1, bb2):
  # calculate IoU(Intersection over Union) of 2 boxes 
  # **IoU = Area of Overlap / Area of Union
  # https://github.com/Hulkido/RCNN/blob/master/RCNN.ipynb

  x_left = max(bb1['x1'], bb2['x1'])
  y_top = max(bb1['y1'], bb2['y1'])
  x_right = min(bb1['x2'], bb2['x2'])
  y_bottom = min(bb1['y2'], bb2['y2'])
  # if there is no overlap output 0 as intersection area is zero.
  if x_right < x_left or y_bottom < y_top:
    return 0.0
  # calculate Overlapping area
  intersection_area = (x_right - x_left) * (y_bottom - y_top)
  bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
  bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
  union_area = bb1_area + bb2_area - intersection_area

  return intersection_area / union_area

def selective_search(image):
  # return region proposals of selective searh over an image
  ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
  ss.setBaseImage(image)
  ss.switchToSelectiveSearchFast()
  return ss.process()

def plot_results(image, bboxes, voc_dataset, color = (0, 69, 255)):
  plot_cfg = {'bbox_color':color, 'bbox_thickness':2, 
                  'font':cv2.FONT_HERSHEY_SIMPLEX, 'fontScale':0.5, 'fontColor':color, 'lineThickness':1}
  img_ss = image.copy()
  for box in bboxes:
    bbox = box['bbox']
    cv2.rectangle(img_ss, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), plot_cfg['bbox_color'], plot_cfg['bbox_thickness'])
    cv2.putText(img_ss, f"{voc_dataset.label_type[box['class']]}, {str(box['conf'])[:5]}",  (bbox['x1'], bbox['y1'] - 5), plot_cfg['font'], 
                plot_cfg['fontScale'], plot_cfg['fontColor'], plot_cfg['lineThickness'])
  return img_ss

def mean_average_precision(pred, truth, iou_threshold = 0.5, num_classes = 21, per_class = False):
    # compute mAP https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/metrics/mean_avg_precision.py
    # `pred` is given in [[{'bbox':{'x1', 'x2', 'y1', 'y2'}, 'class'(int), 'conf'}, ...], ...]
    # `truth` is given in [[{'x1', 'x2', 'y1', 'y2', 'class'(int)}, more boxes...], ...]
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(1, num_classes): # class '0' is background

        TP = 0
        FP = 0
        total_true_bboxes = 0

        # list detected(predicted) objects of class 'c'
        detections = []

        for idx, prs in enumerate(pred):
          for pr in prs:
            if pr['class'] == c:
                detections.append((pr['conf'], idx, pr['bbox']))

        # make checkbox for checking whether gt object was detected
        total_true_bboxes = 0
        is_detected = []
        for gts in pred:
          is_detected.append([False for _ in gts])
          total_true_bboxes += sum([gt['class']==c for gt in gts])

        detections.sort(reverse=True)

        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            num_gts = len(truth[detection[1]])

            # find most closest g.t box to pred as best_gt_idx
            best_iou = 0
            for idx, gt in enumerate(truth[detection[1]]):
                #print(gt, detection[2])
                
                iou = calculate_IoU(gt, detection[2])

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # if considered found
            #try:
            #  print(best_iou, truth[detection[1]][best_gt_idx], detection[2])
            #except:
            #  pass
            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if is_detected[detection[1]][best_gt_idx] == False:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    is_detected[detection[1]][best_gt_idx] = True
                else: # duplicate is FP
                    FP[detection_idx] = 1
            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1
        
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        #if len(TP)>0 and len(FP)>0:
        #  print(TP_cumsum[-1], FP_cumsum[-1])
        #print(total_true_bboxes)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)  # ratio of detected objects!
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)         # ratio of predictions that are true objects!

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        #print(precisions, recalls, torch.trapz(precisions, recalls))
        average_precisions.append(torch.trapz(precisions, recalls))
        #print('----------')
    if per_class: 
        return average_precisions
    else:
        return sum(average_precisions) / len(average_precisions)

#https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/
def nms(P, iou_threshold = 0.5):
  # P: list of dicts {'bbox':(x1,y1,x2,y2), 'conf':float, 'class':int}
  conf_list = np.array([x['conf'] for x in P])
  conf_order = (-conf_list).argsort() # apply minus to reverse order !!
  isremoved = [False for _ in range(len(P))]
  keep = []

  for idx in range(len(P)):
    to_keep = conf_order[idx]
    if isremoved[to_keep]:
      continue
    
    # append to keep list
    keep.append(P[to_keep])
    isremoved[to_keep] = True
    # remove overlapping bboxes
    for order in range(idx + 1, len(P)):
      bbox_idx = conf_order[order]
      if isremoved[bbox_idx]==False:  # if not removed yet
        # check overlapping
        iou = calculate_IoU(P[to_keep]['bbox'], P[bbox_idx]['bbox'])
        if iou > iou_threshold:
          isremoved[bbox_idx] = True
  return keep

class RCNN_Dataset(torch.utils.data.Dataset):
  # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
  def __init__(self, dataset, cfg, data_path, IoU_threshold={'positive':0.5, 'partial':0.3}, sample_ratio=(32, 96)):
    """
    Args:
        label_file (list of tuple(im path, label)): Path to image + annotations.
        im_root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    self.data_path = data_path
    self.dataset = dataset
    self.transform = transforms.Compose([ # preprocess image
        transforms.Resize((cfg['image_size'], cfg['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if self.dataset_exists()==False:
      self.generate_dataset(sample_ratio, IoU_threshold)
    else: 
      print('[*] Loading dataset from', self.data_path)
      with open(self.data_path + 'train_images.pkl', 'rb') as f:
        self.train_images=pickle.load(f)
      with open(self.data_path + 'train_labels.pkl', 'rb') as f:
        self.train_labels=pickle.load(f)

      # check if both files are complete, flawless
      if not len(self.train_images)==len(self.train_labels):
        raise ValueError('The loaded dataset is invalid (of different size).')

  def __len__(self):
    return len(self.train_labels)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()
    image=Image.fromarray(cv2.cvtColor(self.train_images[idx], cv2.COLOR_BGR2RGB))
    return  {'image': self.transform(image), 'label': self.train_labels[idx][0], 
             'est_bbox': self.train_labels[idx][1], 'gt_bbox': self.train_labels[idx][2]}
  '''
  # not working when interact too much w/ drive :(
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    img_name = os.path.join(self.im_root_dir, self.label_file[idx][0])
    image = io.imread(img_name)
    if self.transform:
        image = self.transform(image)
    return  {'image': image, 'label': self.label_file[idx][1]}
  '''
  def dataset_exists(self):
    if os.path.exists(self.data_path+'train_images.pkl')==False:
      return False
    if os.path.exists(self.data_path + 'train_labels.pkl')==False:
      return False
    
    return True    

  def generate_dataset(self, sample_ratio, IoU_threshold, padding=16):
    #https://github.com/Hulkido/RCNN/blob/master/RCNN.ipynb

    image_dir=self.dataset.train_dir+'/JPEGImages/'
    annot_dir=self.dataset.train_dir+'/Annotations/'
    obj_counter = 0
    bg_counter = 0
    self.train_images=[]
    self.train_labels=[]

    print('[*] Generating dataset for R-CNN.')

    pbar = tqdm(sorted(os.listdir(image_dir))[:2000], position=0, leave=True) # only 2000 images :( <-------------
    
    for img_name in pbar:   
      pbar.set_description(f"Data size: {len(self.train_labels)}")
      
      # load image & gt bounding boxes 
      image = cv2.imread(image_dir + img_name)
      xml_path=annot_dir+img_name[:-4]+'.xml'
      gt_bboxes = self.dataset.read_xml(xml_path)
      # generete bbox proposals via selective search
      rects = selective_search(image)[:2000]  # parse first 2000 boxes
      random.shuffle(rects)
      # loop through all ss bounding box proposals
      for (x, y, w, h) in rects:
        # apply padding
        x1, x2 = np.clip([x-padding, x+w+padding], 0, image.shape[1])
        y1, y2 = np.clip([y-padding, y+h+padding], 0, image.shape[0])
        bbox_est = {'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}
        
        # check the proposal with every elements of the gt boxes
        is_object = False # define flag
        for gt_bbox in gt_bboxes:
          iou = calculate_IoU(gt_bbox, bbox_est)

          if iou >= IoU_threshold['positive']: # if object(RoI > 0.5)
            obj_counter+=1
            cropped = image[bbox_est['y1']:bbox_est['y2'], bbox_est['x1']:bbox_est['x2'], :]
            self.train_images.append(cropped)
            #cv2.imwrite(self.data_path+'train_images/'+str(image_count)+'.jpg', cropped) <-- too much drive I/O, timed out
            #image_count+=1
            est_bbox_xywh=((bbox_est['x1'] + bbox_est['x2']) / 2, (bbox_est['y1'] + bbox_est['y2']) / 2,
                            bbox_est['x2']-bbox_est['x1'], bbox_est['y2']-bbox_est['y1'])
            gt_bbox_xywh=((gt_bbox['x1'] + gt_bbox['x2']) / 2, (gt_bbox['y1'] + gt_bbox['y2']) / 2,
                           gt_bbox['x2']-gt_bbox['x1'], gt_bbox['y2']-gt_bbox['y1'])
            self.train_labels.append([gt_bbox['class'], est_bbox_xywh, gt_bbox_xywh])

            is_object = True
            break
        # if the object is not close to any g.t bbox
        if bg_counter < sample_ratio[1] and is_object==False: 
          bg_counter+=1
          cropped = image[bbox_est['y1']:bbox_est['y2'], bbox_est['x1']:bbox_est['x2'], :]
          self.train_images.append(cropped)
          #cv2.imwrite(self.data_path+'train_images/'+str(image_count)+'.jpg', cropped) <-- too much drive I/O, timed out
          #image_count+=1
          est_bbox_xywh=(1.0, 1.0, 1.0, 1.0)
          gt_bbox_xywh=(1.0, 1.0, 1.0, 1.0)
          self.train_labels.append([0, est_bbox_xywh, gt_bbox_xywh])

        if obj_counter >= sample_ratio[0] and bg_counter==sample_ratio[1]:  # control the ratio between 2 types
          obj_counter -= sample_ratio[0]
          bg_counter = 0

        
    print('[*] Dataset generated! Saving labels to', self.data_path)
    with open(self.data_path + 'train_labels.pkl', 'wb') as f:
      pickle.dump(self.train_labels, f)
    with open(self.data_path + 'train_images.pkl', 'wb') as f:
      pickle.dump(self.train_images, f)

def RCNN_DatasetLoader(voc_dataset, cfg, training_cfg, shuffle=True):
  ds = RCNN_Dataset(voc_dataset, cfg, data_path = '../TestDatasets/voc/')
  return torch.utils.data.DataLoader(ds, batch_size=training_cfg['batch_size'], shuffle=shuffle, num_workers=2)

class RCNN_classifier_Dataset(torch.utils.data.Dataset):
  # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
  def __init__(self, dataset, cfg, IoU_threshold={'positive':0.5, 'partial':0.3}, sample_ratio=(32, 96),
              data_path='../TestDatasets/voc/'):
    """
    Args:
        label_file (list of tuple(im path, label)): Path to image + annotations.
        im_root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    self.data_path = data_path
    self.dataset = dataset
    self.transform = transforms.Compose([ # preprocess image
        transforms.Resize((cfg['image_size'], cfg['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if self.dataset_exists()==False:
      self.generate_dataset(sample_ratio, IoU_threshold)
    else: 
      print('[*] Loading dataset from', self.data_path)
      with open(self.data_path + 'train_images_classifier.pkl', 'rb') as f:
        self.train_images=pickle.load(f)
      with open(self.data_path + 'train_labels_classifier.pkl', 'rb') as f:
        self.train_labels=pickle.load(f)

      # check if both files are complete, flawless
      if not len(self.train_images)==len(self.train_labels):
        raise ValueError('The loaded dataset is invalid (of different size).')

  def __len__(self):
    return len(self.train_labels)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()
    image=Image.fromarray(cv2.cvtColor(self.train_images[idx], cv2.COLOR_BGR2RGB))
    return  {'image': self.transform(image), 'label': self.train_labels[idx][0], 
             'est_bbox': self.train_labels[idx][1], 'gt_bbox': self.train_labels[idx][2]}
  '''
  # not working when interact too much w/ drive :(
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    img_name = os.path.join(self.im_root_dir, self.label_file[idx][0])
    image = io.imread(img_name)
    if self.transform:
        image = self.transform(image)
    return  {'image': image, 'label': self.label_file[idx][1]}
  '''
  def dataset_exists(self):
    if os.path.exists(self.data_path+'train_images.pkl')==False:
      return False
    if os.path.exists(self.data_path + 'train_labels.pkl')==False:
      return False
    
    return True    

  def generate_dataset(self, sample_ratio, IoU_threshold, padding=16):
    #https://github.com/Hulkido/RCNN/blob/master/RCNN.ipynb

    image_dir=self.dataset.train_dir+'/JPEGImages/'
    annot_dir=self.dataset.train_dir+'/Annotations/'
    obj_counter = 0
    self.train_images=[]
    self.train_labels=[]

    print('[*] Generating dataset for R-CNN.')

    pbar = tqdm(sorted(os.listdir(image_dir)), position=0, leave=True) # only 2000 images :( <-------------
    
    for img_name in pbar:   
      pbar.set_description(f"Data size: {len(self.train_labels)}")
      
      # load image & gt bounding boxes 
      image = cv2.imread(image_dir + img_name)
      xml_path=annot_dir+img_name[:-4]+'.xml'
      gt_bboxes = self.dataset.read_xml(xml_path)

      # directly use gt bboxes as positive samples
      for gt_bbox in gt_bboxes:
        cropped = image[gt_bbox['y1']:gt_bbox['y2'], gt_bbox['x1']:gt_bbox['x2'], :]
        self.train_images.append(cropped)

        gt_bbox_xywh=((gt_bbox['x1'] + gt_bbox['x2']) / 2, (gt_bbox['y1'] + gt_bbox['y2']) / 2,
                           gt_bbox['x2']-gt_bbox['x1'], gt_bbox['y2']-gt_bbox['y1'])
        est_bbox_xywh = gt_bbox_xywh
        self.train_labels.append([gt_bbox['class'], est_bbox_xywh, gt_bbox_xywh])
      obj_counter += len(gt_bboxes)

      # time to collect background :)
      if obj_counter >= sample_ratio[0]:
        obj_counter -= sample_ratio[0]
        bg_counter = 0
        # generete bbox proposals via selective search
        rects = selective_search(image)[:2000]  # parse first 2000 boxes
        random.shuffle(rects)
        # loop through all ss bounding box proposals
        for (x, y, w, h) in rects:
          # apply padding
          x1, x2 = np.clip([x-padding, x+w+padding], 0, image.shape[1])
          y1, y2 = np.clip([y-padding, y+h+padding], 0, image.shape[0])
          bbox_est = {'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}
          is_object = False
          
          # check the proposal with every elements of the gt boxes
          for gt_bbox in gt_bboxes:
            iou = calculate_IoU(gt_bbox, bbox_est)
              
            if iou > IoU_threshold['partial']: # if object
              is_object=True
              break
          # save image
          if is_object==False:
            bg_counter+=1
            cropped = image[bbox_est['y1']:bbox_est['y2'], bbox_est['x1']:bbox_est['x2'], :]
            self.train_images.append(cropped)
            #cv2.imwrite(self.data_path+'train_images/'+str(image_count)+'.jpg', cropped) <-- too much drive I/O, timed out
            #image_count+=1
            est_bbox_xywh=(1.0, 1.0, 1.0, 1.0)
            gt_bbox_xywh=(1.0, 1.0, 1.0, 1.0)
            self.train_labels.append([0, est_bbox_xywh, gt_bbox_xywh])

          if bg_counter==sample_ratio[1]:  # control the ratio between 2 types
            break

        
    print('[*] Dataset generated! Saving labels to', self.data_path)
    with open(self.data_path + 'train_labels_classifier.pkl', 'wb') as f:
      pickle.dump(self.train_labels, f)
    with open(self.data_path + 'train_images_classifier.pkl', 'wb') as f:
      pickle.dump(self.train_images, f)

def RCNN_classifier_DatasetLoader(voc_dataset, cfg, training_cfg, shuffle=True):
  ds = RCNN_classifier_Dataset(voc_dataset, cfg)
  return torch.utils.data.DataLoader(ds, batch_size=training_cfg['batch_size'], shuffle=shuffle, num_workers=2)


class PlotSamples():
  def __init__(self, to_sample=10):
    # create table for plotting 
    self.to_sample = to_sample
    self.images = []

  def log(self, model, val_dataset):
    print('[*] Plotting samples to wandb board...')
    sampled = []
    for x in range(self.to_sample):
      img=val_dataset[x]['image']
      bboxes=model.inference(img)

      result_image = plot_results(img, bboxes, val_dataset)

      result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
      sampled.append(wandb.Image(result_image))

    self.images.append(sampled)
    # write to table
    sample_columns = ['sample '+str(x+1) for x in range(self.to_sample)]
    sample_table = wandb.Table(columns=sample_columns)

    for step_image in self.images:
      sample_table.add_data(*step_image)
    wandb.run.log({"Samples_table" : sample_table})

class ComputeMAP():
  def __init__(self, iou_threshold=0.5, to_sample=10):
    # create table for plotting 
    self.iou_threshold = iou_threshold
    self.sample_table = wandb.Table(columns=voc_2012_classes)
    self.to_sample = to_sample

  def log(self, model, val_dataset):
    # return 0
    val_images = sorted(os.listdir(image_path))
    
    sampled = []
    for x in range(to_sample):
      img=cv2.imread(image_path+val_images[x]+'.jpg')
      bboxes=model.inference(img)

      result_image = plot_results(img, bboxes, val_dataset)
      sampled.append(result_image)
    return sampled