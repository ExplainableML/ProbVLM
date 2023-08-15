import os
from os.path import join as ospj
from os.path import expanduser
import csv
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision as tv
from torchvision.models.detection import maskrcnn_resnet50_fpn as maskrcnn

class UnNormalize(object):
    def __init__(self, 
#                  mean=[0.485, 0.456, 0.406], 
#                  std=[0.229, 0.224, 0.225]):
                 mean=(0.48145466, 0.4578275, 0.40821073), 
                 std=(0.26862954, 0.26130258, 0.27577711)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        unnormed_tensor = torch.zeros_like(tensor)
        for i, (t, m, s) in enumerate(zip(tensor, self.mean, self.std)):
            unnormed_tensor[i] = t.mul(s).add(m)
            # The normalize code -> t.sub_(m).div_(s)
        return unnormed_tensor

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class FlickrCap(Dataset):
    def __init__(self, data_root, image_ids_path=None, transform=None, target_transform=None):
        self.root = expanduser(data_root)
        self.transform = transform
        self.target_transform = target_transform

        # load ids
#         image_ids_path = './datasets/annotations/flickr/train.txt'
        with open(image_ids_path) as f:
            lines = f.readlines()
        image_files = [line.strip() + '.jpg' for line in lines]

        # load data
        self.datas = []
        data_path = ospj(os.path.dirname(self.root), 'results.csv')
        reader = csv.reader(open(data_path))
        for i, row in enumerate(reader):
            if i == 0:
                continue
            data = [val.strip() for val in row[0].split('|')] # ex: ['1001465944.jpg', '0', 'A woman is walking .']
            if data[0] in image_files:
                self.datas.append(data)

    def __getitem__(self, index, get_caption=False):
        image_file, _, caption = self.datas[index]
        img = Image.open(ospj(self.root, image_file)).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        is_img_masked = False
        img_masked = img
        if self.target_transform is not None:
    #             target = self.target_transform(target)
            target = self.target_transform(caption)
            target = target.squeeze(0)
        if get_caption:
            return img, target, caption, img_masked, is_img_masked
        else:
            return img, target, img_masked, is_img_masked

    def __len__(self):
        return len(self.datas)


class FlickrBboxes(FlickrCap):
    def __init__(self, data_root, device, image_ids_path=None, transform=None, target_transform=None):
        super().__init__(data_root, image_ids_path, transform, target_transform)
        self.device = device
        self.detector = maskrcnn(pretrained=True)
        self.detector = self.detector.to(self. device); self.detector.eval()
        self.unnorm = UnNormalize()
        self.norm = tv.transforms.Compose([tv.transforms.ToTensor(),])

    def __getitem__(self, index, get_caption=False):
        image_file, _, caption = self.datas[index]
        img = Image.open(ospj(self.root, image_file)).convert('RGB')
        
        if self.transform is not None:
            img, img_masked, is_img_masked = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(caption)
            target = target.squeeze(0)

        # bbox
        input_for_bbox = tv.transforms.ToPILImage()(self.unnorm(img))
        input_for_bbox = self.norm(input_for_bbox)
        input_for_bbox = input_for_bbox.to(self.device)
        with torch.no_grad():
            p = self.detector([input_for_bbox])
            bboxes = p[0]['boxes'].cpu().numpy()
            bboxes[:,2] = bboxes[:,2] - bboxes[:,0]
            bboxes[:,3] = bboxes[:,3] - bboxes[:,1]
            cats = p[0]['labels'].cpu().numpy()
            scores = p[0]['scores'].cpu().numpy()
            bboxes = [bbox for i, bbox in enumerate(bboxes) if scores[i] >= 0.5]
            bboxes = torch.tensor(np.array(bboxes))
            bbox_cats = [cat for i, cat in enumerate(cats) if scores[i] >= 0.5]
            bbox_cats = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in bbox_cats]
        if len(bboxes) == 0:
            bboxes = torch.tensor([[0., 0., 0., 0.]])
            bbox_cats = ['none']
            
        if get_caption:
            return img, target, caption, bboxes, bbox_cats
        else:
            return img, target, bboxes