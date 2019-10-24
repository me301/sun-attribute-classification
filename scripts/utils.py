import os
import abc
import six
import scipy.io
from torchvision import transforms
from PIL import Image


@six.add_metaclass(abc.ABCMeta)
class Dataset(object):

    @abc.abstractmethod
    def __init__(self, ims, labels):
        """Initialise the dataset class with images,
        path to your directory and labels."""
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get an item from the whole list of files index by index."""
        raise NotImplementedError

    @abc.abstractproperty
    def __len__(self):
        """Length of the dataset for the Dataloader
        to iterate through all indices"""
        raise NotImplementedError


class SunDataset(Dataset):
    def __init__(self, im_label_path, im_path, label_path):
        # load all image files, sorting them to
        # ensure that they are aligned
        images = scipy.io.loadmat(im_label_path)
        im_list = [list(images['images'][i][0])[0] for i in range(len(images['images']))]# noqa
        self.imgs = [os.path.join(im_path, i) for i in im_list]

        attributes = scipy.io.loadmat(label_path)
        labels = attributes['labels_cv']
        self.labels = (labels > 0).astype(int)

    def __getitem__(self, idx):
        # load images
        img = Image.open(self.imgs[idx])
        img = img.convert(mode='RGB')
        img = transforms.functional.resize(img, (400, 400))
        img = transforms.functional.to_tensor(img)
        img = img.view(3, 400, 400)
        img = img/255
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.imgs)
