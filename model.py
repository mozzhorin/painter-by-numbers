import os
import argparse
import numpy as np
import urllib
import zipfile
import time

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.lib.io import file_io
from tensorflow.python.client import device_lib

import sys

def copy2bucket(file, bucket):
    with file_io.FileIO(file, mode='rb') as i_f:
        with file_io.FileIO(os.path.join(bucket, file), mode='wb+') as o_f:
            o_f.write(i_f.read())
            print('Copied', file, 'to', bucket)
            

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--numclasses',
                      type=int,
                      default=10,
                      help='Number of classes in the dataset.') 
    parser.add_argument('--dataurl',
                      type=str,
                      default='https://www.dropbox.com/s/hgyjjefanrzqrny/top10_x448.zip?raw=1',
                      help='URL to download the dataset.')      
    parser.add_argument('--exportdir',
                      type=str,
                      default='',
                      help='GCS path to export model')    
    parser.add_argument('--epochs',
                      type=int,
                      default=10,
                      help='The number of training epochs to perform.')
    parser.add_argument('--batchsize',
                      type=int,
                      default=32,
                      help='The batch size during training')    
    parser.add_argument('--optimizer',
                      type=str,
                      default='sgd',
                      help='Keras optimizer')
    parser.add_argument('--lr',
                      type=float,
                      default=0.01,
                      help='Learning rate for training.')
    parser.add_argument('--hflip',
                      type=int,
                      default=1,
                      help='Training data augmentation: horizontal flip, 0 or 1 ')
    parser.add_argument('--vflip',
                      type=int,
                      default=1,
                      help='Training data augmentation: vertical flip, 0 or 1 ')
    parser.add_argument('--rotation',
                      type=int,
                      default=45,
                      help='Training data augmentation: rotation range in grad ')
    parser.add_argument('--wshift',
                      type=float,
                      default=0.2,
                      help='Training data augmentation: width shift range, [0.0,1.0] ')
    parser.add_argument('--hshift',
                      type=float,
                      default=0.2,
                      help='Training data augmentation: height shift range, [0.0,1.0] ')
    parser.add_argument('--transferlearning',
                      type=int,
                      default=1,
                      help='Transfer learning: 0 - no (all layers are trained), \
                            1 - yes (only the last layer is trained).')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    
    args = parse_arguments()
    if args.transferlearning:
        weights = 'imagenet'
    else:
        weights = None
    
    # Tensorboard
    try:
        os.mkdir('/logs')
    except Exception as e: 
        print(e)

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)    
    

    # Download and unzip the dataset
    working_train_dir = "train/"
    working_test_dir = "test/"
    data_zip = 'data.zip'
    if not os.path.exists(data_zip):
        urllib.request.urlretrieve(args.dataurl, data_zip)
        print('Downloading the dataset')
    if not os.path.exists('train'):
        zip_ref = zipfile.ZipFile(data_zip, 'r')
        zip_ref.extractall()
        print('Unpacking the dataset')
        zip_ref.close()

    # Check the GPUs
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('Available devices:')
    print(device_lib.list_local_devices())
        
        
    # Create the model for transfer learning using pretrained ResNet50
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('Creating the model...')
    model = Sequential()
    
    model.add(ResNet50(    
      include_top=False,   
      weights=weights, 
      pooling='avg' 
    ))
    
    model.add(Dense(
      args.numclasses, 
      activation='softmax' 
    ))

    # Do not train the first layer in transfer learning
    if args.transferlearning:
        model.layers[0].trainable = False

    model.compile(
      optimizer=args.optimizer, 
      loss='categorical_crossentropy', 
      metrics=['accuracy'] 
    )
    
    # Image generators
    image_size = 224

    print('>>>>>>>>>>>>>>>>>>>>>>>')    
    print('Creating the image generators...')
    data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

    data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                       horizontal_flip=bool(args.hflip),
                                       vertical_flip=bool(args.vflip),
                                       rotation_range=args.rotation,
                                       width_shift_range = args.wshift,
                                       height_shift_range = args.hshift)

    train_generator_with_aug = data_generator_with_aug.flow_from_directory(
            working_train_dir,
            target_size=(image_size, image_size),
            batch_size=args.batchsize,
            class_mode='categorical')

    validation_generator = data_generator_no_aug.flow_from_directory(
            working_test_dir,
            target_size=(image_size, image_size),
            batch_size=args.batchsize,
            class_mode='categorical')

    # Train the model
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('Starting the training...')
    history_aug = model.fit_generator(
            train_generator_with_aug,
            epochs=args.epochs,
            validation_data=validation_generator,
            shuffle=True,
            callbacks=[tensorboard])
    
    # Save the model
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('Saiving the model localy...')
    time_stamp = str(int(time.time()))
    model_name = 'top' + str(args.numclasses) + '-' + time_stamp + '.h5'
    model.save(model_name)
    

    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('Trying to copy the model to a bucket...')
    try:
        copy2bucket(model_name, args.exportdir)
    except Exception as e: 
        print(e)
    
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('Trying to copy the logs to a bucket...')
    try:
        import pickle
        history_file = 'history-top' + str(args.numclasses) + '-' + time_stamp + '.pkl'
        with open(history_file, 'wb') as f:
            pickle.dump(history_aug.history, f)
        copy2bucket(history_file, args.exportdir)
        logs = os.listdir('/logs')
        for l in logs:
            copy2bucket('/logs'+l, args.exportdir)
    except Exception as e: 
        print(e)
        
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('Training is finished.')
