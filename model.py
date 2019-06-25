import os
import argparse
import numpy as np
import urllib.request
import zipfile

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

 

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf-data-dir',
                      type=str,
                      default='',
                      help='GCS path or local path of training data.')
    parser.add_argument('--tf-num-classes',
                      type=int,
                      default=10,
                      help='Number of classes in the dataset.')
    parser.add_argument('--tf-model-dir',
                      type=str,
                      help='GCS path or local directory.')
    parser.add_argument('--tf-export-dir',
                      type=str,
                      default='export/',
                      help='GCS path or local directory to export model')
    parser.add_argument('--tf-model-type',
                      type=str,
                      default='CNN',
                      help='Tensorflow model type for training.')
    parser.add_argument('--tf-train-epochs',
                      type=int,
                      default=10,
                      help='The number of training epochs to perform.')
    parser.add_argument('--tf-batch-size',
                      type=int,
                      default=32,
                      help='The number of batch size during training')
    parser.add_argument('--tf-learning-rate',
                      type=float,
                      default=0.01,
                      help='Learning rate for training.')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    
    args = parse_arguments()
    print(args)
    
    data_dir = ''
    working_train_dir = data_dir + "train/"
    working_test_dir = data_dir + "test/"
    if (os.path.isdir(args.tf_export_dir) == False):
        os.mkdir(args.tf_export_dir)

    # Download and unzip the dataset
    data_url = 'https://www.dropbox.com/s/gwto8r7myjuh3gu/top10.zip?raw=1'
    data_zip = 'top10.zip'
    if not os.path.exists(data_zip):
        urllib.request.urlretrieve(data_url, data_zip)
        print('Downloading the dataset')
    if not os.path.exists('train'):
        zip_ref = zipfile.ZipFile(data_zip, 'r')
        zip_ref.extractall()
        print('Unpacking the dataset')
        zip_ref.close()
        
        
    # Create the model for transfer learning using pretrained ResNet50
    print('Creating the model')
    model = Sequential()
    
    model.add(ResNet50(    
      include_top=False,   
      weights='imagenet', 
      pooling='avg' 
    ))
    
    model.add(Dense(
      args.tf_num_classes, 
      activation='softmax' 
    ))

    # do not train the first layer
    model.layers[0].trainable = False

    model.compile(
      optimizer='sgd', 
      loss='categorical_crossentropy', 
      metrics=['accuracy'] 
    )
    
    # Image generators
    image_size = 224
    
    print('Creating the image generators')
    data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

    data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                       horizontal_flip=True,
                                       rotation_range=20,
                                       width_shift_range = 0.2,
                                       height_shift_range = 0.2)

    train_generator_with_aug = data_generator_with_aug.flow_from_directory(
            working_train_dir,
            target_size=(image_size, image_size),
            batch_size=args.tf_batch_size,
            class_mode='categorical')

    validation_generator = data_generator_no_aug.flow_from_directory(
            working_test_dir,
            target_size=(image_size, image_size),
            class_mode='categorical')

    # Train the model
    print('Starting the training...')
    history_aug = model.fit_generator(
            train_generator_with_aug,
            steps_per_epoch=10,
            epochs=args.tf_train_epochs,
            validation_data=validation_generator,
            validation_steps=1)
    
    # Save the model
    print('Saiving the model')
    model.save(str(args.tf_export_dir + 'top10.h5'))
    #model.save_weights(str(args.tf_export_dir + 'top10_weights.h5'))