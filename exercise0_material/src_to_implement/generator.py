import os.path
import json
import random
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        #TODO: implement constructor
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.file_path = "./data"+file_path
        self.label_path = "./data"+label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        self.epoch_counter = 0
        self.batch_cursor = 0

        # load the names of image files
        with open(self.label_path) as f:
            self.class_labels = json.load(f)
        self.image_lists = os.listdir(self.file_path)
        self.image_num = len(self.class_labels)
        if self.shuffle:
            random.shuffle(self.image_lists)

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        if self.batch_cursor * self.batch_size >= self.image_num:
            self.epoch_counter += 1
            self.batch_cursor = 0
            if self.shuffle:
                random.shuffle(self.image_lists)

        image = []
        labels = []
        for i in range(self.batch_cursor * self.batch_size, self.batch_cursor * self.batch_size + self.batch_size):
            if i >= self.image_num:
                break
            image.append(self.augment(transform.resize(np.load(f"{self.file_path}/{self.image_lists[i]}"),self.image_size)))
            labels.append(int(self.image_lists[i].split(".")[0]))
        # in this case, create one more additional batch and fill them with files from the beginning
        if self.batch_cursor*self.batch_size + self.batch_size > self.image_num:
            for i in range(0,self.batch_cursor*self.batch_size + self.batch_size - self.image_num):
                image.append(self.augment(transform.resize(np.load(f"{self.file_path}/{self.image_lists[i]}"), self.image_size)))
                labels.append(int(self.image_lists[i].split(".")[0]))

        self.batch_cursor += 1

        #TODO:
        return np.copy(image), np.copy(labels)


    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        if self.mirroring:
            img = np.flip(img, random.choice([0, 1]))
            img = np.flip(img, random.choice([0, 1]))
        if self.rotation:
            img = np.rot90(img, np.random.randint(4))
        return img


    def current_epoch(self):
        # return the current epoch number
        return self.epoch_counter

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict[x]
    
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        images, labels = self.next()
        fig = plt.figure(figsize = (self.image_size[0], self.image_size[0]))
        for i in range(self.batch_size):
            fig.add_subplot(int(np.ceil(self.batch_size/3)), 3, i+1)
            plt.axis('off')
            plt.title(self.class_name(self.class_labels[str(labels[i])]))
            plt.imshow(images[i])
        plt.show()

