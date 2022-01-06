import numpy as np
import os
import random
import torch


def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)


def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    import imageio
    image = imageio.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image


class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, config={}, device = torch.device('cpu')):
        """
        Args:
            num_classes: int
                Number of classes for classification (N-way)
            
            num_samples_per_class: int
                Number of samples per class in the support set (K-shot).
                Will generate additional sample for the querry set.
                
            device: cuda.device: 
                Device to allocate tensors to.
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        

        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]
        
        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
            num_train + num_val:]
        self.device = device

    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: str
                train/val/test set to sample from
                
            batch_size: int:
                Size of batch of tasks to sample
                
        Returns:
            images: tensor
                A tensor of images of size [B, K+1, N, 784]
                where B is batch size, K is number of samples per class, 
                N is number of classes
                
            labels: tensor
                A tensor of images of size [B, K+1, N, N] 
                where B is batch size, K is number of samples per class, 
                N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        #############################
        #### YOUR CODE GOES HERE ####
        #############################

        # Repeat below for  `batch_size` number of Times #
        sampled_input = []
        sampled_labels = []

        for b in range(batch_size):
            # Sample N different character Folders
            sampled_character_classes = random.sample( folders ,  self.num_classes )
            # Sample K images from each of these folders
            all_images = []            

            for ind, class_folder in enumerate(sampled_character_classes):
                # +1 is done, because, 1 example it in the query Set
                sampled_images_per_character_path = random.sample( os.listdir(class_folder) , self.num_samples_per_class + 1 )
                sampled_images_per_character_path = [ os.path.join(class_folder, i) for i in sampled_images_per_character_path ]
                
                sampled_images_per_character = [ image_file_to_array(i, self.dim_input) for i in sampled_images_per_character_path ]
                all_images.append(sampled_images_per_character)

            # Append the inputs
            sampled_input.append( all_images )

            # Restructure the labels
            class_labels = np.eye(self.num_classes)
            class_labels = np.tile( class_labels, (1,self.num_samples_per_class + 1))
            class_labels = np.split(class_labels, self.num_samples_per_class + 1 , axis=1)
            sampled_labels.append(class_labels)

        # plt.imshow( np.reshape(sampled_input[0 , 0, 0, :], (28,28)  ) ); plt.show()
        # sampled_input_r = np.reshape(sampled_input, (2, 5, 784))    
        sampled_input = np.array(sampled_input)    
        sampled_input =  np.transpose( sampled_input, (  0,2,1,3 ) )
        sampled_labels = np.array( sampled_labels )

        # shuffle the set inefficiently !! 
        for batch_ind in range(batch_size):
            shuffled_index = np.random.permutation(self.num_classes)
            # index the final kth set of classes & shuffle them
            train_query_input = sampled_input[ batch_ind , -1, :, : ]
            sampled_input[batch_ind, -1, :, : ] = train_query_input[ shuffled_index  ]
            # repeat for labels
            train_query_label = sampled_labels[ batch_ind , -1, :, : ]
            sampled_labels[batch_ind, -1, :, : ] = train_query_label[ shuffled_index ]

        return torch.tensor(sampled_input, dtype=torch.float), torch.tensor(sampled_labels, dtype=torch.float)