'''
USAGE:
from dataset.celeba import celeba_imgs
imgs = celeba_imgs() #shape should be: ()

> PREREQUISITES: CelebA images in img_align_celeba.zip should be in this path relative to this file:
    .\celeba\img_align_celeba
    (e.g.: .\celeba\img_align_celeba\000001.jpg should exist)
'''


import os
import numpy as np
import joblib
import skimage
from skimage import transform, filters
import random
import matplotlib.pyplot as plot
from PIL import Image

cachedir = os.getenv('CACHE_HOME', './cache')
mem = joblib.Memory(cachedir=os.path.join(cachedir, 'celeba'))

dataset_home = os.path.abspath(os.path.dirname(__file__))


class CelebA(object):
    '''
    Large-scale CelebFaces Attributes (CelebA) Dataset [1].
    http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    References:
    [1]: Ziwei Liu, Ping Luo, Xiaogang Wang and Xiaoou Tang.
         Deep Learning Face Attributes in the Wild. Proceedings of
         International Conference on Computer Vision (ICCV), December, 2015.
    '''

    def __init__(self):
        self.name = 'celeba'
        self.n_imgs = 202599
        self.data_dir = os.path.join(dataset_home, self.name)
        self.img_dir = os.path.join(self.data_dir, 'img_align_celeba')
        self._install()

    def img(self, idx):
        img_path = os.path.join(self.img_dir, '%.6d.jpg' % (idx+1))
        #print('\t\tLoading image: ', img_path)
        return np.array(Image.open(img_path))

    def imgs(self):
        for i in range(self.n_imgs):
            yield self.img(i)

    def _install(self):
        partition_list_path = os.path.join(dataset_home, "list_eval_partition.txt")
        partitions = [[], [], []]
        with open(partition_list_path, 'r') as f:
            for i, line in enumerate(f):
                img_name, partition = line.strip().split(' ')
                if int(img_name[:6]) != i + 1:
                    raise ValueError('Parse error.')
                partition = int(partition)
                partitions[partition].append(i)
        self.train_idxs, self.val_idxs, self.test_idxs = map(np.array, partitions)

def img_augment(img, translation=0.0, scale=1.0, rotation=0.0, gamma=1.0,
                contrast=1.0, hue=0.0, border_mode='constant'):
    if not (np.all(np.isclose(translation, [0.0, 0.0])) and
            np.isclose(scale, 1.0) and
            np.isclose(rotation, 0.0)):
        img_center = np.array(img.shape[:2]) / 2.0
        scale = (scale, scale)
        transf = transform.SimilarityTransform(translation=-img_center)
        transf += transform.SimilarityTransform(scale=scale, rotation=rotation)
        translation = img_center + translation
        transf += transform.SimilarityTransform(translation=translation)
        img = transform.warp(img, transf, order=3, mode='edge')
    if not np.isclose(gamma, 1.0):
        img **= gamma
    colorspace = 'rgb'
    if not np.isclose(contrast, 1.0):
        img = color.convert_colorspace(img, colorspace, 'hsv')
        colorspace = 'hsv'
        img[..., 1:] **= contrast
    if not np.isclose(hue, 0.0):
        img = color.convert_colorspace(img, colorspace, 'hsv')
        colorspace = 'hsv'
        img[..., 0] += hue
        img[img[..., 0] > 1.0, 0] -= 1.0
        img[img[..., 0] < 0.0, 0] += 1.0
    img = color.convert_colorspace(img, colorspace, 'rgb')
    if np.min(img) < 0.0 or np.max(img) > 1.0:
        raise ValueError('Invalid values in output image.')
    return img

def _resize(args):
    img, rescale_size, bbox = args
    img = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    # Smooth image before resize to avoid moire patterns
    scale = img.shape[0] / float(rescale_size)
    sigma = np.sqrt(scale) / 2.0
    img = skimage.filters.gaussian(img, sigma=sigma, multichannel=True)
    img = transform.resize(img, (rescale_size, rescale_size, 3), order=3)
    img = (img*255).astype(np.uint8)
    return img

def sample_img_augment_params(translation_sigma=1.0, scale_sigma=0.01,
                              rotation_sigma=0.01, gamma_sigma=0.07,
                              contrast_sigma=0.07, hue_sigma=0.0125):
    translation = np.random.normal(scale=translation_sigma, size=2)
    scale = np.random.normal(loc=1.0, scale=scale_sigma)
    rotation = np.random.normal(scale=rotation_sigma)
    mu = gamma_sigma**2
    gamma = np.random.normal(loc=mu, scale=gamma_sigma)
    gamma = np.exp(gamma/np.log(2))
    mu = contrast_sigma**2
    contrast = np.random.normal(loc=mu, scale=contrast_sigma)
    contrast = np.exp(contrast/np.log(2))
    hue = np.random.normal(scale=hue_sigma)
    return translation, scale, rotation, gamma, contrast, hue

def _resize_augment(args):
    img, rescale_size, bbox = args
    augment_params = sample_img_augment_params(
        translation_sigma=2.00, scale_sigma=0.01, rotation_sigma=0.01,
        gamma_sigma=0.05, contrast_sigma=0.05, hue_sigma=0.01
    )
    img = img_augment(img, *augment_params, border_mode='nearest')
    img = _resize((img, rescale_size, bbox))
    return img

def img_transform(imgs, to_bc01=True):
    imgs = imgs.astype(np.float)
    imgs /= 127.5
    imgs -= 1.0
    if to_bc01:
        imgs = np.transpose(imgs, (0, 3, 1, 2))
    return imgs

saved_train_imgs = 'celeba_dataset_train.npy'
saved_test_imgs = 'celeba_dataset_test.npy'

def saved_data_exists():
    return os.path.isfile(saved_train_imgs) and os.path.isfile(saved_test_imgs)


#@mem.cache
def celeba_imgs(img_size=64, bbox=(40, 218-30, 15, 178-15), img_idxs=None,
                n_augment=0):
    if bbox[1] - bbox[0] != bbox[3] - bbox[2]:
        raise ValueError('Image is not square')
    dataset = CelebA()
    if img_idxs is None:
        img_idxs = list(range(dataset.n_imgs)) #202599
    if n_augment == 0:
        preprocess_fun = _resize
        n_imgs = len(img_idxs)
    else:
        preprocess_fun = _resize_augment
        n_imgs = n_augment

    def img_iter():
        for i in range(n_imgs):
            yield dataset.img(img_idxs[i % len(img_idxs)])

    with joblib.Parallel(n_jobs=-2) as parallel:
        imgs = parallel(joblib.delayed(preprocess_fun)
                        ((img, img_size, bbox)) for img in img_iter())
    imgs = np.array(imgs)
    return imgs

def get_train_data(train_idxs, img_size=64, n_augment=0):
    x_train = celeba_imgs(img_size, img_idxs=train_idxs, n_augment=n_augment)
    #x_train = np.transpose(x_train, (0, 3, 1, 2))  #Does transposing have an advantage?

    return x_train

def get_test_data(test_idxs, img_size):
    x_test = celeba_imgs(img_size, img_idxs=test_idxs)
    x_test = img_transform(x_test, to_bc01=True)

    return x_test

def feeds(img_size=64, n_augment=int(6e5), split='val'):
    if saved_data_exists():
        print('Found saved train data(', saved_train_imgs, ') and saved test data(', saved_test_imgs, ')! Loading...')
        x_train = np.load(saved_train_imgs)
        x_test = np.load(saved_test_imgs)
    else:
        dataset = CelebA()
        print('Saved data not found! Processing and loading data from ', dataset.img_dir)
        if split == 'val':
            train_idxs = dataset.train_idxs
            test_idxs = dataset.val_idxs
        elif split == 'test':
            train_idxs = np.hstack((dataset.train_idxs, dataset.val_idxs))
            test_idxs = dataset.test_idxs

        x_train = get_train_data(train_idxs)
        x_test = get_test_data(test_idxs)

        np.save(saved_train_imgs, x_train)
        np.save(saved_test_imgs, x_test)

    return x_train, x_test

def get_train_data_idxs():
    dataset = CelebA()
    return dataset.train_idxs

def get_test_data_idxs():
    dataset = CelebA()
    return dataset.test_idxs

def get_train_batch(idxs):
    x_train = get_train_data(idxs)
    return x_train


def celeb_loader(batch_size, img_size=64):
    dataset = CelebA()
    list_of_idxs = list(dataset.train_idxs)

    while True:
        random.shuffle(list_of_idxs)
        idx_list_for_popping = list_of_idxs[:]

        while len(idx_list_for_popping) != 0:
            img_stack = np.zeros((batch_size, img_size, img_size, 3), dtype=np.float32)
            for i in range(batch_size):
                img_stack[i, :, :, :] = get_train_data([idx_list_for_popping.pop()])/255
            yield img_stack, None


if __name__ == "__main__":
    celeba_generator = celeba_loader(3)
    imgs, dummy = next(celeba_generator)
    for celeba_img in imgs:
        print("celeba_img.shape = ", celeba_img.shape)
        plot.imshow(celeba_img)
        plot.show()


