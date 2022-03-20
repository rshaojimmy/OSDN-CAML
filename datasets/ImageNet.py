from __future__ import print_function
import torch
from torchvision import datasets, transforms
import random
from .rotation import RotateImageFolder
from pdb import set_trace as st


from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data


class ImageNetDownSample(data.Dataset):
    """`DownsampleImageNet`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    train_list = [
        ['train_data_batch_1'],
        ['train_data_batch_2'],
        ['train_data_batch_3'],
        ['train_data_batch_4'],
        ['train_data_batch_5'],
        ['train_data_batch_6'],
        ['train_data_batch_7'],
        ['train_data_batch_8'],
        ['train_data_batch_9'],
        ['train_data_batch_10']
    ]
    test_list = [
        ['val_data'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()
            # resize label range from [1,1000] to [0,1000),
            # This is required by CrossEntropyLoss
            self.train_labels[:] = [x - 1 for x in self.train_labels]

            self.train_data = np.concatenate(self.train_data)
            [picnum, pixel] = self.train_data.shape
            pixel = int(np.sqrt(pixel / 3))
            self.train_data = self.train_data.reshape((picnum, 3, pixel, pixel))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            [picnum,pixel]= self.test_data.shape
            pixel = int(np.sqrt(pixel/3))

            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()

            # resize label range from [1,1000] to [0,1000),
            # This is required by CrossEntropyLoss
            self.test_labels[:] = [x - 1 for x in self.test_labels]
            self.test_data = self.test_data.reshape((picnum, 3, pixel, pixel))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        # img.save('/mnt/lustre/shaorui/data/opensetadv/tst/1.jpg', "JPEG")
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)




def get_imagenet(train, split, batch_size, image_size):


    if train:

        transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        dataset = ImageNetDownSample(root='../datasets/imagenet64',
                            train=True, 
                            transform=transform)


    else: 
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        dataset = ImageNetDownSample(root='../datasets/imagenet64',
                    train=False, 
                    transform=transform)

    if split is '0':
      rand_allclass = np.random.RandomState(seed=20210823).permutation(1000).tolist()
    elif split is '1':
      rand_allclass = np.random.RandomState(seed=20210826).permutation(1000).tolist()
    elif split is '2':
      rand_allclass = np.random.RandomState(seed=20210829).permutation(1000).tolist()
      
    knownclass = rand_allclass[:100]
    unknownclass = rand_allclass[100:]


    kmask = [i for i,e  in enumerate(dataset) if e[1] in knownclass]
    unkmask = [i for i,e  in enumerate(dataset) if e[1] in unknownclass]

    if train:

        random.shuffle(kmask)
        validationportion = int(0.1*len(kmask))

        kmask_rand_val = kmask[:validationportion]
        kmask_rand_train= kmask[validationportion:]

        known_set_train = torch.utils.data.Subset(dataset, kmask_rand_train)
        known_set_val = torch.utils.data.Subset(dataset, kmask_rand_val)

        known_set_train = RotateImageFolder(known_set_train)

        known_data_loader_train = torch.utils.data.DataLoader(
            dataset=known_set_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4)

        known_data_loader_val = torch.utils.data.DataLoader(
            dataset=known_set_val,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4)

        return known_data_loader_train, known_data_loader_val, knownclass

    else:

        known_set = torch.utils.data.Subset(dataset, kmask)
        unknown_set = torch.utils.data.Subset(dataset, unkmask)

        known_data_loader = torch.utils.data.DataLoader(
            dataset=known_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4)

        unknown_data_loader = torch.utils.data.DataLoader(
            dataset=unknown_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4)

        return known_data_loader, unknown_data_loader, knownclass