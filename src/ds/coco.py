"""MS-COCO image-to-caption retrieval dataset code

reference codes:
https://github.com/pytorch/vision/blob/v0.2.2_branch/torchvision/datasets/coco.py
https://github.com/yalesong/pvse/blob/master/data.py
"""

import os
from os.path import join as ospj
try:
    import ujson as json
except ImportError:
    import json

from PIL import Image
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset


class CocoCaptionsCap(Dataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        ids (list, optional): list of target caption ids
        extra_annFile (string, optional): Path to extra json annotation file (for training)
        extra_ids (list, optional): list of extra target caption ids (for training)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        instance_annFile (str, optional): Path to instance annotation json (for PMRP computation)

    Example:
        .. code:: python
            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root='dir where images are',
                                    annFile='json annotation file',
                                    transform=transforms.ToTensor())
            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample
            print("Image Size: ", img.size())
            print(target)
        Output: ::
            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']
    """
    def __init__(self, root, annFile, ids=None,
                 extra_annFile=None, extra_ids=None,
                 transform=None, target_transform=None,
                 instance_annFile=None):
        self.root = os.path.expanduser(root)
        if extra_annFile:
            self.coco = COCO()
            with open(annFile, 'r') as fin1, open(extra_annFile, 'r') as fin2:
                dataset = json.load(fin1)
                extra_dataset = json.load(fin2)
                if not isinstance(dataset, dict) or not isinstance(extra_dataset, dict):
                    raise TypeError('invalid type {} {}'.format(type(dataset),
                                                                type(extra_dataset)))
                if set(dataset.keys()) != set(extra_dataset.keys()):
                    raise KeyError('key mismatch {} != {}'.format(list(dataset.keys()),
                                                                  list(extra_dataset.keys())))
                for key in ['images', 'annotations']:
                    dataset[key].extend(extra_dataset[key])
            self.coco.dataset = dataset
            self.coco.createIndex()
        else:
            self.coco = COCO(annFile)
        self.ids = list(self.coco.anns.keys()) if ids is None else list(ids)
        if extra_ids is not None:
            self.ids += list(extra_ids)
        self.ids = [int(id_) for id_ in self.ids]
        self.transform = transform
        self.target_transform = target_transform

        self.all_image_ids = set([self.coco.loadAnns(annotation_id)[0]['image_id'] for annotation_id in self.ids])

        iid_to_cls = {}
        if instance_annFile:
            with open(instance_annFile) as fin:
                instance_ann = json.load(fin)
            for ann in instance_ann['annotations']:
                image_id = int(ann['image_id'])
                code = iid_to_cls.get(image_id, [0] * 90)
                code[int(ann['category_id']) - 1] = 1
                iid_to_cls[image_id] = code

            seen_classes = {}
            new_iid_to_cls = {}
            idx = 0
            for k, v in iid_to_cls.items():
                v = ''.join([str(s) for s in v])
                if v in seen_classes:
                    new_iid_to_cls[k] = seen_classes[v]
                else:
                    new_iid_to_cls[k] = idx
                    seen_classes[v] = idx
                    idx += 1
            iid_to_cls = new_iid_to_cls

            if self.all_image_ids - set(iid_to_cls.keys()):
                print(f'Found mismatched! {self.all_image_ids - set(iid_to_cls.keys())}')

        self.iid_to_cls = iid_to_cls
        self.n_images = len(self.all_image_ids)

    def __getitem__(self, index, get_caption=False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a caption for the annotation.
        """
        coco = self.coco
        annotation_id = self.ids[index]
        annotation = coco.loadAnns(annotation_id)[0]
        image_id = annotation['image_id']
#         target = annotation['caption']
        caption = annotation['caption']

        path = coco.loadImgs(image_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
#             target = self.target_transform(target)
            target = self.target_transform(caption)
            target = target.squeeze(0)
        img_masked = img 
        is_img_masked = False
        if get_caption:
            return img, target, caption, img_masked, is_img_masked
        else:
            return img, target, img_masked, is_img_masked
#         if get_caption:
#             return img, target, caption, annotation_id, image_id
#         else:
#             return img, target, annotation_id, image_id

    def __len__(self):
        return len(self.ids)


class CocoBboxes(CocoCaptionsCap):
    def __init__(self, root, annFile, ids, extra_ids=None, extra_annFile=None, transform=None, target_transform=None, instanceFile=None):
        super().__init__(root, annFile, ids, extra_ids=extra_ids, extra_annFile=extra_annFile, transform=transform, target_transform=target_transform)
        dirname = os.path.dirname(annFile)
        self.coco_for_instance = COCO(instanceFile)
        
        categories_info = self.coco_for_instance.loadCats(self.coco_for_instance.getCatIds())
        self.category_id2name = {info['id']: info['name'] for info in categories_info}
        
    def __getitem__(self, index, get_caption=False):
        """
        Returns:
            bboxes (torch.tensor, size=(#bboxes, 4)): (x_left, y_top, width, height)
        """
        coco = self.coco
        annotation_id = self.ids[index]
        annotation = coco.loadAnns(annotation_id)[0]
        image_id = annotation['image_id']
        caption = annotation['caption']

        path = coco.loadImgs(image_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        W, H = img.size
        if self.transform is not None:
            img, img_masked, is_img_masked = self.transform(img)

        if self.target_transform is not None:
#             target = self.target_transform(target)
            target = self.target_transform(caption)
            target = target.squeeze(0)
        
        # get bboxes
        bbox_ann_ids = self.coco_for_instance.getAnnIds(imgIds=[image_id])
        bbox_anns = self.coco_for_instance.loadAnns(bbox_ann_ids)
        bboxes = torch.tensor([ann['bbox'] for ann in bbox_anns])
        bbox_cats = [self.category_id2name[ann['category_id']] for ann in bbox_anns]
        if len(bboxes) == 0:
            bboxes = torch.tensor([[0., 0., 0., 0.]])
            bbox_cats = ['none']
        else:
            # bbox transform
            length_ratio = 224 / H if W > H else 224 / W
            bboxes *= length_ratio
            if W > H:
                bboxes[:, 0] -= ((W * length_ratio) - 224) / 2 
            else:
                bboxes[:, 1] -= ((H * length_ratio) - 224) / 2 
            x_right = torch.clamp(bboxes[:, 0] + bboxes[:,2], 0, 224)
            y_bottom = torch.clamp(bboxes[:, 1] + bboxes[:,3], 0, 224)
            bboxes[:, 0] = torch.clamp(bboxes[:, 0], 0, 224)
            bboxes[:, 1] = torch.clamp(bboxes[:, 1], 0, 224)
            bboxes[:, 2] = x_right - bboxes[:, 0]
            bboxes[:, 3] = y_bottom - bboxes[:, 1]
            is_object = (bboxes[:,2] > 0).logical_and(bboxes[:,3] > 0)
            bboxes = bboxes[is_object]
            bbox_cats = [cat for i, cat in enumerate(bbox_cats) if is_object[i].item()]
            
        if get_caption:
            return img, target, caption, bboxes, bbox_cats
        else:
            return img, target, bboxes
