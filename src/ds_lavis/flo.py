"""CUB Caption image-to-caption retrieval dataset code

PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import scipy.io
import glob

def pad_text(num):
    if num<10:
        return '0000'+str(num)
    if num<100:
        return '000'+str(num)
           
    if num<1000:
        return '00'+str(num)


class FLOCaption(Dataset):
    """CUB Captions Dataset.

    Args:
        image_root (string): Root directory where images are downloaded to.
        caption_root (string): Root directory where captions are downloaded to.
        target_classes (str or list): target class ids
            - if str, it is the name of the file with target classes (line by line)
            - if list, it is directly used to get classes
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        omit_ids (str, optional): Path of file with the list of image ids to omit,
            if not specified, use all images in the target classes.
        ids (str, optional): Path of file with the list of target image ids,
            if not specified, use all images in the target classes.
    """
    def __init__(self, image_root, caption_root,
                 target_classes,
                 transform=None, target_transform=None,
                 ):

        self.image_root = os.path.expanduser(image_root)
        self.caption_root = os.path.expanduser(caption_root)

        if isinstance(target_classes, str):
            with open(target_classes) as fin:
                _classes = [int(line.strip().split('_')[1]) - 1 for line in fin]
            target_classes = _classes

        target_classes = set(list(target_classes))
        if (target_classes - set(range(102))):
            raise ValueError(f'target classes should be an integer array between 0-102, but {target_classes}')
        print(f'prepare flo dataset with {len(target_classes)} classes')

        targets = []
        index_to_class = {}
        class_to_indices = {}
        class_to_img_indices = {}
        idx = 0
        n_images = 0
        label_path = image_root+'/imagelabels.mat'
        jpg_path = image_root+'/jpg/'
        class_labels = np.array(scipy.io.loadmat(label_path)['labels'])[0]
        images = glob.glob(jpg_path+'*')
        images.sort()
        n_images=0
        for i in range(len(images)):
            img_name = images[i]
            cls_num = class_labels[i] - 1
            if cls_num in target_classes:
                _target = []

                class_txt = 'class_'+pad_text(cls_num+1)
                #print(caption_root,class_txt,img_name)
                caption_img = img_name.split('/')[-1]
                txt_fname = os.path.join(caption_root, class_txt, caption_img.replace('jpg', 'txt'))
                with open(txt_fname) as fin:
                    captions = [line.strip() for line in fin]

                for caption in captions:
                    _target.append(
                        (os.path.join(img_name), caption)
                    )
                    index_to_class[idx] = cls_num
                    class_to_indices.setdefault(cls_num, []).append(idx)
                    idx += 1
                targets.extend(_target)
                n_images+=1
        self.targets = targets
        self.target_classes = target_classes
        self.index_to_class = index_to_class
        self.class_to_indices = class_to_indices
        self.class_to_img_indices = class_to_img_indices

        self.n_images = n_images

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, target = self.targets[index]

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            target = target.squeeze(0)

        return img, target, self.index_to_class[index], index

    def __len__(self):
        return len(self.targets)


class FLOSampler(Sampler):
    """ Sampler for CUB Captions training.

    Args:
        dataset (CUBCaption object): dataset object to apply the sampler.
        batch_size (int): batch size.
        adjust_epoch (bool): if true, the iterations for one epoch is re-calculated.
    """
    def __init__(self, dataset, batch_size, adjust_epoch=True):
        self.dataset = dataset
        self.batch_size = batch_size
        print("Batch:",self.batch_size)
        self.target_classes = dataset.target_classes
        if batch_size != len(self.target_classes):
            raise ValueError(f'{batch_size} != {len(self.target_classes)}')
        self.index_to_class = dataset.index_to_class
        self.class_to_indices = dataset.class_to_indices
        self.n_items = len(self.index_to_class)

        if adjust_epoch:
            self.n_iters = int(self.n_items / len(self.target_classes))
        else:
            self.n_iters = self.n_items

    def __iter__(self):
        batch = []
        indices = list(range(self.n_items))

        np.random.shuffle(indices)
        for cur_iter, idx in enumerate(indices):
            batch = [idx]
            pos_cls = self.index_to_class[idx]
            for cls_num, _indices in self.class_to_indices.items():
                if cls_num == pos_cls:
                    continue
                else:
                    batch.append(np.random.choice(_indices))
            np.random.shuffle(batch)
            if cur_iter > self.n_iters:
                return
            yield batch


    def __len__(self):
        return self.n_iters
