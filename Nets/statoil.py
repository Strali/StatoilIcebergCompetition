import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout, BatchNormalization, Input, Flatten, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.merge import Concatenate, add
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from utils import generate_data, augment_data, get_callbacks

def build_model( baseline_cnn = False ):
    #Based on kernel https://www.kaggle.com/devm2024/keras-model-for-beginners-0-210-on-lb-eda-r-d
    image_input = Input( shape = (75, 75, 3), name = 'images' )
    angle_input = Input( shape = [1], name = 'angle' )
    activation = 'elu'
    bn_momentum = 0.99
    
    # Simple CNN as baseline model
    if baseline_cnn:
        model = Sequential()

        model.add( Conv2D(16, kernel_size = (3, 3), activation = 'relu', input_shape = (75, 75, 3)) )
        model.add( BatchNormalization(momentum = bn_momentum) )
        model.add( MaxPooling2D(pool_size = (3, 3), strides = (2, 2)) )
        model.add( Dropout(0.2) )

        model.add( Conv2D(32, kernel_size = (3, 3), activation = 'relu') )
        model.add( BatchNormalization(momentum = bn_momentum) )
        model.add( MaxPooling2D(pool_size = (2, 2), strides = (2, 2)) )
        model.add( Dropout(0.2) )

        model.add( Conv2D(64, kernel_size = (3, 3), activation = 'relu') )
        model.add( BatchNormalization(momentum = bn_momentum) )
        model.add( MaxPooling2D(pool_size = (2, 2), strides = (2, 2)) )
        model.add( Dropout(0.2) )

        model.add( Conv2D(128, kernel_size = (3, 3), activation = 'relu') )
        model.add( BatchNormalization(momentum = bn_momentum) )
        model.add( MaxPooling2D(pool_size = (2, 2), strides = (2, 2)) )
        model.add( Dropout(0.2) )

        model.add( Flatten() )

        model.add( Dense(256, activation = 'relu') )
        model.add( BatchNormalization(momentum = bn_momentum) )
        model.add( Dropout(0.3) )

        model.add( Dense(128, activation = 'relu') )
        model.add( BatchNormalization(momentum = bn_momentum) )
        model.add( Dropout(0.3) )

        model.add( Dense(1, activation = 'sigmoid') )

        opt = Adam( lr = 1e-3, beta_1 = .9, beta_2 = .999, decay = 1e-3 )

        model.compile( loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'] )

        model.summary()

    else:
        img_1 = Conv2D( 32, kernel_size = (3, 3), activation = activation, padding = 'same' ) ((BatchNormalization(momentum=bn_momentum) ) ( image_input) )
        img_1 = MaxPooling2D( (2,2)) (img_1 )
        img_1 = Dropout( 0.2 )( img_1 )

        img_1 = Conv2D( 64, kernel_size = (3, 3), activation = activation, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (img_1) )
        img_1 = MaxPooling2D( (2,2) ) ( img_1 )
        img_1 = Dropout( 0.2 )( img_1 )
  
         # Residual block
        img_2 = Conv2D( 128, kernel_size = (3, 3), activation = activation, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (img_1) )
        img_2 = Dropout(0.2) ( img_2 )
        img_2 = Conv2D( 64, kernel_size = (3, 3), activation = activation, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (img_2) )
        img_2 = Dropout(0.2) ( img_2 )
        
        img_res_1 = add( [img_1, img_2] )

        #Residual block
        img_3 = Conv2D( 128, kernel_size = (3, 3), activation = activation, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (img_res_1) )
        img_3 = Dropout(0.2) ( img_3 )
        img_3 = Conv2D( 64, kernel_size = (3, 3), activation = activation, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (img_3) )
        img_3 = Dropout(0.2) ( img_3 )

        img_res_2 = add( [img_res_1, img_3] )

        # Filter resudial output
        img_res_2 = Conv2D( 128, kernel_size = (2, 2), activation = activation ) ( (BatchNormalization(momentum=bn_momentum)) (img_res_2) )
        img_res_2 = MaxPooling2D( (2,2) ) ( img_res_2 )
        img_res_2 = Dropout( 0.2 )( img_res_2 )
        img_res_2 = GlobalAveragePooling2D() ( img_res_2 )
        
        cnn_out = ( Concatenate()( [img_res_2, BatchNormalization(momentum=bn_momentum)(angle_input)]) )

        dense_layer = Dropout( 0.5 ) ( BatchNormalization(momentum=bn_momentum) (PReLU() (Dense(256, activation = None) (cnn_out))) )
        dense_layer = Dropout( 0.5 ) ( BatchNormalization(momentum=bn_momentum) (PReLU() (Dense(64, activation = None) (dense_layer))) )
        output = Dense( 1, activation = 'sigmoid' ) ( dense_layer )
        
        model = Model( [image_input, angle_input], output )

        opt = Adam( lr = 1e-3, beta_1 = .9, beta_2 = .999, decay = 1e-3 )

        model.compile( loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'] )

        model.summary()

    return model

 
TEST = False # Should test data be passed to the model?
DO_PLOT = False # Exploratory data plots
USE_AUGMENTATION = False # Whether or not image augmentations should be made
TRAIN_PATH = '../data/train.json'
TEST_PATH = '../data/test.json'
WEIGHT_SAVE_PATH = '../model_weights.hdf5'


BATCH_SIZE = 32
EPOCHS = 100 # Increase this

train_data = pd.read_json( TRAIN_PATH )
train_data[ 'inc_angle' ] = train_data[ 'inc_angle' ].replace('na', 0)
train_data[ 'inc_angle' ] = train_data[ 'inc_angle' ].astype(float).fillna(0.0)

if TEST:
    SEED = np.random.randint( 9999 )
else:
    SEED = 42 # Constant seed for comparability between runs

X = generate_data( train_data )
X_a = train_data[ 'inc_angle' ]
y = train_data[ 'is_iceberg' ]

if DO_PLOT:
    make_plots( train_data, band_samples = True, all_bands = True )

X_train, X_val, X_angle_train, X_angle_val, y_train, y_val = train_test_split( X, X_a, y, train_size = .8, random_state = SEED )
callback_list = get_callbacks( WEIGHT_SAVE_PATH, 20 )

model = build_model()
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
    model.fit( [X_train, X_angle_train], y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, verbose = 1, 
            validation_data = ([X_val, X_angle_val], y_val), callbacks = callback_list )

m, s = divmod( time.time() - start_time, 60 )
print( 'Model fitting done. Total time: {}m {}s'.format(int(m), int(s)) )

model.load_weights( WEIGHT_SAVE_PATH )
val_score = model.evaluate( [X_val, X_angle_val], y_val, verbose = 1 )
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

    PREDICTION_SAVE_PATH = '../submissions/test_submission_twostage_resnet_avg_pool.csv'
    submission.to_csv( PREDICTION_SAVE_PATH, index = False )