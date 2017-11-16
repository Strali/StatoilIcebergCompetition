import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dense, Dropout, BatchNormalization, Input, Flatten, Activation
from keras.layers.merge import Concatenate, add
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from utils import generate_data, augment_data, get_callbacks

def inception_stem():
    image_input = Input( shape = (75, 75, 3), name = 'images' )
    angle_input = Input( shape = [1], name = 'angle' )
    activation_fn = 'relu'
    bn_momentum = .99

    conv1 = Conv2D( 16, kernel_size = (3, 3), strides = 2, activation = activation_fn, padding = 'valid' ) ( (BatchNormalization(momentum=bn_momentum)) (image_input) )
    conv1 = Conv2D( 16, kernel_size = (3, 3), strides = 1, activation = activation_fn, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (conv1) )
    conv1 = Conv2D( 32, kernel_size = (3, 3), strides = 1, activation = activation_fn, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (conv1) )

    maxpool1 = MaxPooling2D( (3, 3), strides = 2, padding = 'valid' ) ( conv1 )
    conv1 = Conv2D( 64, kernel_size = (3, 3), strides = 2, activation = activation_fn, padding = 'valid' ) ( (BatchNormalization(momentum=bn_momentum)) (conv1) )

    concat1 = ( Concatenate() ([maxpool1, conv1]) )

    conv21 = Conv2D( 32, kernel_size = (1, 1), strides = 1, activation = activation_fn, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (concat1) )
    conv21 = Conv2D( 64, kernel_size = (3, 3), strides = 1, activation = activation_fn, padding = 'valid' ) ( (BatchNormalization(momentum=bn_momentum)) (conv21) )

    conv22 = Conv2D( 32, kernel_size = (1, 1), strides = 1, activation = activation_fn, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (concat1) )
    conv22 = Conv2D( 32, kernel_size = (7, 1), strides = 1, activation = activation_fn, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (conv22) )
    conv22 = Conv2D( 32, kernel_size = (1, 7), strides = 1, activation = activation_fn, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (conv22) )
    conv22 = Conv2D( 64, kernel_size = (3, 3), strides = 1, activation = activation_fn, padding = 'valid' ) ( (BatchNormalization(momentum=bn_momentum)) (conv22) )

    concat2 = ( Concatenate() ([conv21, conv22]) )

    conv3 = Conv2D( 96, kernel_size = (2, 2), strides = 2, activation = activation_fn, padding = 'valid' ) ( (BatchNormalization(momentum=bn_momentum)) (concat2) )
    maxpool2 = MaxPooling2D( (2, 2), strides = 2, padding = 'valid' ) ( (BatchNormalization(momentum=bn_momentum)) (concat2) )

    concat3 = ( Concatenate() ([conv3, maxpool2]) )

    conv4 = Conv2D( 128, kernel_size = (2, 2), strides = 2, activation = activation_fn, padding = 'valid' ) ( concat3 )
    conv5 = Conv2D( 128, kernel_size = (2, 2), strides = 1, activation = activation_fn, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (conv4) )
    concat3 = ( Flatten() (conv5) )
    concat3 = ( Concatenate()( [concat3, BatchNormalization(momentum=bn_momentum)(angle_input)]) )

    dense1 = Dropout( 0.5 ) ( (BatchNormalization(momentum=bn_momentum)) (Dense(256, activation = activation_fn) (concat3)) )
    dense2 = Dropout( 0.5 ) ( (BatchNormalization(momentum=bn_momentum)) (Dense(64, activation = activation_fn) (dense1)) )
    output = Dense( 1, activation = 'sigmoid' ) ( dense2 )

    model = Model( [image_input, angle_input], output )
    #model = Model( image_input, output )
    
    opt = Adam( lr = 1e-3, beta_1 = .9, beta_2 = .999, decay = 1e-3 )

    model.compile( loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'] )

    model.summary()

    return model

TEST = False # Should test data be passed to the model?
USE_AUGMENTATION = True # Whether or not image augmentations should be made
TRAIN_PATH = '../data/train.json'
TEST_PATH = '../data/test.json'
WEIGHT_SAVE_PATH = '../model_weights.hdf5'
PREDICTION_SAVE_PATH = '../submissions/test_submission.csv'

if TEST:
    SEED = np.random.randint( 9999 )
else:
    SEED = 42 # Constant seed for comparability between runs

BATCH_SIZE = 32
EPOCHS = 100 # Increase this

train_data = pd.read_json( TRAIN_PATH )
train_data[ 'inc_angle' ] = train_data[ 'inc_angle' ].replace('na', 0)
train_data[ 'inc_angle' ] = train_data[ 'inc_angle' ].astype(float).fillna(0.0)

X = generate_data( train_data )
X_a = train_data[ 'inc_angle' ]
y = train_data[ 'is_iceberg' ]

if DO_PLOT:
    make_plots( train_data, band_samples = True, all_bands = True )

X_train, X_val, X_angle_train, X_angle_val, y_train, y_val = train_test_split( X, X_a, y, train_size = .8, random_state = SEED )
#X_train, X_val, y_train, y_val = train_test_split( X, y, train_size = .8, random_state = SEED )
callback_list = get_callbacks( WEIGHT_SAVE_PATH, 20 )

model = inception_stem()
start_time = time.time()

if USE_AUGMENTATION:
    image_augmentation = ImageDataGenerator( rotation_range = 20,
                                             horizontal_flip = True,
                                             vertical_flip = True,
                                             width_shift_range = .3,
                                             height_shift_range =.3,
                                             zoom_range = .1 )

    input_generator = augment_data( image_augmentation, X_train, X_angle_train, y_train, batch_size = BATCH_SIZE )

    model.fit_generator( input_generator, steps_per_epoch = 4096/BATCH_SIZE, epochs = EPOCHS,
                        callbacks = callback_list, verbose = 2, 
                        validation_data = augment_data(image_augmentation, X_val, X_angle_val, y_val, batch_size = BATCH_SIZE),
                        validation_steps = len(X_val)/BATCH_SIZE )

else: 
    # Just fit model to the given training data
    model.fit( X_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, verbose = 1, 
               validation_data = ([X_val, X_angle_val], y_val), callbacks = callback_list )

m, s = divmod( time.time() - start_time, 60 )
print( 'Model fitting done. Total time: {}m {}s'.format(int(m), int(s)) )

model.load_weights( WEIGHT_SAVE_PATH )
val_score = model.evaluate( X_val, y_val, verbose = 1 )
print( 'Validation score: {}'.format(round(val_score[0], 5)) )
print( 'Validation accuracy: {}%'.format(round(val_score[1]*100, 2)) )
print( '='*28, '\n' )

if TEST:
    print( 'Loading and evaluating on test data' )
    test_data = pd.read_json( TEST_PATH )

    X_test = generate_data( test_data )
    X_a_test = test_data[ 'inc_angle' ]
    test_predictions = model.predict( [X_test, X_a_test] )

    submission = pd.DataFrame()
    submission[ 'id' ] = test_data[ 'id' ]
    submission[ 'is_iceberg' ] = test_predictions.reshape( (test_predictions.shape[0]) )

    submission.to_csv( PREDICTION_SAVE_PATH, index = False )