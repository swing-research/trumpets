import tensorflow as tf
import numpy as np
import cv2
import argparse
from sklearn.utils import shuffle

        
def Dataset_preprocessing(dataset = 'MNIST', image_type = True):
    
    if dataset == 'mnist':
        
        nch = 1
        r = 32
        (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
        
    elif dataset == 'fmnist':
    
        (train_images, _), (test_images, _) = tf.keras.datasets.fashion_mnist.load_data()
        r = 32
        nch = 1

    elif dataset == 'cifar10':
        (train_images, _), (test_images, _) = tf.keras.datasets.cifar10.load_data()
        r = 32
        nch = 3

    elif dataset == 'celeba':
        celeba = np.load('/raid/konik/data/celeba_64_100k.npy')
        celeba = shuffle(celeba)
        train_images, test_images = np.split(celeba, [80000], axis=0)
        print(type(train_images[0,0,0,0]))
        nch = 3
        r = 64
        
    elif dataset == 'imagenet':
        imagenet = np.load('/raid/Amir/Projects/datasets/Tiny_imagenet.npy')
        imagenet = shuffle(imagenet)
        train_images, test_images = np.split(imagenet, [80000], axis=0)
        nch = 3
        r = 64
        
    elif dataset == 'rheo':
        rheo = np.load('/raid/Amir/Projects/datasets/rheology.npy')
        rheo = shuffle(rheo)
        train_images, test_images = np.split(rheo, [1500], axis=0)
        nch = 3
        r = 64
        
        
    elif dataset == 'chest':
        chest = np.load('/raid/Amir/Projects/datasets/X_ray_dataset_128.npy')[:100000,:,:,0:1]
        chest = shuffle(chest)
        print(np.shape(chest))
        train_images, test_images = np.split(chest, [80000], axis=0)
        # print(type(train_images[0,0,0,0]))
        nch = 1
        r = 128
    
    
    elif dataset == 'church':
        church = np.load('/raid/Amir/Projects/datasets/church_outdoor_train_lmdb_color_64.npy')[:100000,:,:,:]
        church = shuffle(church)
        print(np.shape(church))
        train_images, test_images = np.split(church, [80000], axis=0)
        # print(type(train_images[0,0,0,0]))
        nch = 3
        r = 64
        
        

    training_images = np.zeros((np.shape(train_images)[0], r, r, 1))
    testing_images = np.zeros((np.shape(test_images)[0], r, r, 1))

    if train_images.shape[1] != r:

        for i in range(np.shape(train_images)[0]):
            if nch == 1:
                training_images[i,:,:,0] = cv2.resize(train_images[i] , (r,r))
            else:
                training_images[i] = cv2.resize(train_images[i] , (r,r))

        for i in range(np.shape(test_images)[0]):
            if nch == 1:
                testing_images[i,:,:,0] = cv2.resize(test_images[i] , (r,r))
            else:
                testing_images[i] = cv2.resize(test_images[i] , (r,r))

    else:
        training_images = train_images
        testing_images = test_images

    # Normalize the images to [-1, 1]
    training_images = training_images.astype('float32')
    training_images /= (training_images.max()/2)
    training_images = training_images - 1.0

    testing_images = testing_images.astype('float32')
    testing_images /= (testing_images.max()/2)
    testing_images = testing_images - 1.0

    if not image_type:
        training_images = training_images.reshape(-1, r**2)
        testing_images = testing_images.reshape(-1, r**2)

    return training_images , testing_images
      
        
  

    
def flags():

    parser = argparse.ArgumentParser()
     
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=400,
        help='number of epochs to train for')
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='batch_size')

   
    parser.add_argument(
        '--dataset', 
        type=str,
        default='mnist',
        help='which dataset to work with')
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='learning rate')
    
    
    parser.add_argument(
        '--ml_threshold', 
        type=int,
        default= 2,
        help='when should ml training begin')


    
    parser.add_argument(
        '--model_depth',
        type=int,
        default= 6,
        help='revnet depth of model')
    
    parser.add_argument(
        '--latent_depth',
        type=int,
        default= 3,
        help='revnet depth of latent model')
    
    
    parser.add_argument(
        '--learntop',
        type=int,
        default=1,
        help='Trainable top')
    
    parser.add_argument(
        '--gpu_num',
        type=int,
        default=0,
        help='GPU number')

    parser.add_argument(
        '--remove_all',
        type= int,
        default= 0,
        help='Remove the previous experiment')
    
    parser.add_argument(
        '--desc',
        type=str,
        default='Default',
        help='add a small descriptor to folder name')

    parser.add_argument('--train', 
        default=False, action='store_true')
    parser.add_argument('--notrain', 
        dest='train', action='store_false')

    parser.add_argument('--inv', 
        default=False, action='store_true')
    parser.add_argument('--noinv', 
        dest='inv', action='store_false')

    parser.add_argument('--posterior', 
        default=False, action='store_true')
    parser.add_argument('--noposterior', 
        dest='posterior', action='store_false')

    parser.add_argument('--calc_logdet', 
        default=False, action='store_true')
    parser.add_argument('--nocalc_logdet', 
        dest='calc_logdet', action='store_false')

    parser.add_argument('--inv_prob', 
        default='denoising', type=str, help='choose from denoising (default) | sr | randmask | randgauss')


    parser.add_argument(
        '--snr',
        type=float,
        default=50,
        help='measurement SNR (dB)')
    
    parser.add_argument(
        '--inv_conv_activation',
        type=str,
        default= 'linear',
        help='activation of invertible 1x1 conv layer')
    
    parser.add_argument(
        '--T',
        type=float,
        default= 1,
        help='sampling tempreture')
    
    
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed
