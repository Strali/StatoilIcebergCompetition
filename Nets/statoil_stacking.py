import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout, BatchNormalization, Input, Flatten, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.merge import Concatenate, add
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from utils import generate_data, augment_data, get_callbacks

def build_model( n_resblocks = 2, activation = 'elu', l_rate = 1e-3 ):
    image_input = Input( shape = (75, 75, 3), name = 'images' )
    angle_input = Input( shape = [1], name = 'angle' )
    activation_fn = activation
    bn_momentum = 0.99
    
    img_1 = Conv2D( 32, kernel_size = (3, 3), activation = activation_fn, padding = 'same' ) ((BatchNormalization(momentum=bn_momentum) ) ( image_input) )
    img_1 = MaxPooling2D( (2,2)) (img_1 )
    img_1 = Dropout( 0.2 )( img_1 )

    img_1 = Conv2D( 64, kernel_size = (3, 3), activation = activation_fn, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (img_1) )
    img_1 = MaxPooling2D( (2,2) ) ( img_1 )
    img_1 = Dropout( 0.2 )( img_1 )

    for block in range(n_resblocks):
        # Residual block
        img_2 = Conv2D( 128, kernel_size = (3, 3), activation = activation_fn, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (img_1) )
        img_2 = Dropout(0.2) ( img_2 )
        img_2 = Conv2D( 64, kernel_size = (3, 3), activation = activation_fn, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (img_2) )
        img_2 = Dropout(0.2) ( img_2 )
        
        img_res_1 = add( [img_1, img_2] )
        img_1 = img_res_1

    # Filter resudial output
    img_res_2 = Conv2D( 128, kernel_size = (2, 2), activation = activation_fn ) ( (BatchNormalization(momentum=bn_momentum)) (img_res_1) )
    img_res_2 = MaxPooling2D( (2,2) ) ( img_res_2 )
    img_res_2 = Dropout( 0.2 )( img_res_2 )
    img_res_2 = GlobalAveragePooling2D() ( img_res_2 )
    
    cnn_out = ( Concatenate()( [img_res_2, BatchNormalization(momentum=bn_momentum)(angle_input)]) )

    dense_layer = Dropout( 0.5 ) ( BatchNormalization(momentum=bn_momentum) (PReLU() (Dense(256, activation = None) (cnn_out))) )
    dense_layer = Dropout( 0.5 ) ( BatchNormalization(momentum=bn_momentum) (PReLU() (Dense(64, activation = None) (dense_layer))) )
    output = Dense( 1, activation = 'sigmoid' ) ( dense_layer )
    
    model = Model( [image_input, angle_input], output )

    opt = Adam( lr = l_rate, beta_1 = .9, beta_2 = .999, decay = 1e-3 )

    model.compile( loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'] )

    model.summary()

    return model


def deeper_resnet():
    image_input = Input( shape = (75, 75, 3), name = 'images' )
    angle_input = Input( shape = [1], name = 'angle' )
    activation = 'elu'
    bn_momentum = 0.99
    
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
    img_3 = Conv2D( 256, kernel_size = (3, 3), activation = activation, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (img_res_1) )
    img_3 = Dropout(0.2) ( img_3 )
    img_3 = Conv2D( 128, kernel_size = (3, 3), activation = activation, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (img_3) )
    img_3 = Dropout(0.2) ( img_3 )
    img_3 = Conv2D( 64, kernel_size = (3, 3), activation = activation, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (img_3) )
    img_3 = Dropout(0.2) ( img_3 )

    img_res_2 = add( [img_res_1, img_3] )
    img_res_2 = Conv2D( 128, kernel_size = (3, 3), activation = activation ) ( (BatchNormalization(momentum=bn_momentum)) (img_res_2) )
    img_res_2 = MaxPooling2D( (2,2) ) ( img_res_2 )
    img_res_2 = Dropout( 0.2 )( img_res_2 )

    #Residual block
    img_4 = Conv2D( 256, kernel_size = (2, 2), activation = activation, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (img_res_2) )
    img_4 = Dropout(0.2) ( img_4 )
    img_4 = Conv2D( 128, kernel_size = (2, 2), activation = activation, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (img_4) )
    img_4 = Dropout(0.2) ( img_4 )
    img_4 = Conv2D( 128, kernel_size = (2, 2), activation = activation, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (img_4) )
    img_4 = Dropout(0.2) ( img_4 )

    img_res_3 = add( [img_res_2, img_4] )

    # Filter resudial output
    img_res_3 = Conv2D( 256, kernel_size = (2, 2), activation = activation ) ( (BatchNormalization(momentum=bn_momentum)) (img_res_3) )
    img_res_3 = MaxPooling2D( (2,2) ) ( img_res_3 )
    img_res_3 = Dropout( 0.2 )( img_res_3 )
    img_res_3 = Conv2D( 128, kernel_size = (2, 2), activation = activation ) ( (BatchNormalization(momentum=bn_momentum)) (img_res_3) )
    img_res_3 = MaxPooling2D( (2,2) ) ( img_res_3 )
    img_res_3 = Dropout( 0.2 )( img_res_3 )
    img_res_3 = GlobalMaxPooling2D() ( img_res_3 )
    
    cnn_out = ( Concatenate()( [img_res_3, BatchNormalization(momentum=bn_momentum)(angle_input)]) )

    dense_layer = Dropout( 0.5 ) ( BatchNormalization(momentum=bn_momentum) (PReLU() (Dense(256, activation = None) (cnn_out))) )
    dense_layer = Dropout( 0.5 ) ( BatchNormalization(momentum=bn_momentum) (PReLU() (Dense(64, activation = None) (dense_layer))) )
    output = Dense( 1, activation = 'sigmoid' ) ( dense_layer )
    
    model = Model( [image_input, angle_input], output )

    opt = Adam( lr = 1e-3, beta_1 = .9, beta_2 = .999, decay = 1e-3 )

    model.compile( loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'] )

    model.summary()

    return model

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
    
    opt = Adam( lr = 1e-3, beta_1 = .9, beta_2 = .999, decay = 1e-3 )

    model.compile( loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'] )

    model.summary()

    return model


def fit_and_predict_oof(X_train, X_angle_train, y_train, X_test, X_angle_test, base_models, stacker, n_splits = 5):
    print( 'IN PREDICT_AND_FIT_OOF' )
    X_train = np.array(X_train)
    X_angle_train = np.array(X_angle_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    X_angle_test = np.array(X_angle_test)

    folds = list(StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = SEED).split(X_train, y_train))

    S_train = np.zeros((X_train.shape[0], len(base_models)))
    S_test = np.zeros((X_test.shape[0], len(base_models)))

    image_augmentation = ImageDataGenerator( rotation_range = 20,
                                            horizontal_flip = True,
                                            vertical_flip = True,
                                            width_shift_range = .3,
                                            height_shift_range =.3,
                                            zoom_range = .1 )
    for i, clf in enumerate(base_models):
        print( '\nFitting model {}/{}'.format(i+1, int(len(base_models))) )
        S_test_i = np.zeros((X_test.shape[0], n_splits))

        for j, (train_idx, val_idx) in enumerate(folds):
            X_fold_train = X_train[train_idx]
            X_fold_angle = X_angle_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_val = X_train[val_idx]
            X_val_angle = X_angle_train[val_idx]
            y_val = y_train[val_idx]

            if (i == 0): 
                clf.fit( [X_train, X_angle_train], y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, verbose = 1, 
                           validation_data = ([X_val, X_val_angle], y_val), callbacks = callback_list )

            else:
                input_generator = augment_data( image_augmentation, X_fold_train, X_fold_angle, y_fold_train, batch_size = BATCH_SIZE )

                clf.fit_generator( input_generator, steps_per_epoch = 4096/BATCH_SIZE, epochs = EPOCHS,
                                    callbacks = callback_list, verbose = 1, 
                                    validation_data = augment_data(image_augmentation, X_val, X_val_angle, y_val, batch_size = BATCH_SIZE),
                                    validation_steps = len(X_val)/BATCH_SIZE )

            y_pred = clf.predict([X_val, X_val_angle])             

            S_train[val_idx, i] = y_pred.reshape((y_pred.shape[0]))
            S_test_i[:, j] = clf.predict([X_test, X_a_test]).reshape(X_test.shape[0])
        S_test[:, i] = S_test_i.mean(axis=1)
        print( 'Time to fit model {} to all folds: {}m {}s'.format(i+1, int(m), int(s)) )

    results = cross_val_score(stacker, S_train, y_train, cv = 5, scoring = 'neg_log_loss')
    print('Stacker CV score: %.5f' % (-results.mean()))

    print('Fitting stacker to train data and evaluating on test data')
    stacker.fit(S_train, y_train)
    res = stacker.predict_proba(S_test)[:,1]
    return res


TEST = True # Should test data be passed to the model?
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

test_data = pd.read_json( TEST_PATH )
X_test = generate_data( test_data )
X_a_test = test_data[ 'inc_angle' ]

print( 'DATA LOADED' )

callback_list = get_callbacks( WEIGHT_SAVE_PATH, 15 )

# build_model( n_resblocks = 2, activation = 'elu', l_rate = 1e-3 )
m1 = build_model()
print( 'BUILD FIRST MODEL' )
m2 = build_model()
print( 'BUILT SECOND MODEL' )
m3 = build_model(activation = 'relu')
print( 'BUILD THIRD MODEL' )
m4 = build_model(l_rate = 2e-3)
print( 'BUILT FOURTH MODEL' )
m5 = build_model(n_resblocks = 1, activation = 'relu')
print( 'BUILT FIFTH MODEL' )
m6 = build_model(n_resblocks = 1)
print( 'BUILT SIXTH MODEL' )
m7 = deeper_resnet()
print( 'BUILT ALL MODELS' )

base_models = [m1, m2, m3, m4, m5, m6, m7]
#base_models = [m1, m2]
xgb_params = {
    'objective': 'binary:logistic',
    'learning_rate': 0.01,
    'n_estimators': 490,
    'max_depth': 4,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'min_child_weight': 10
}
stacker = XGBClassifier(xgb_params)
print( 'DEFINED STACKER' )
start_time = time.time()

# fit_and_predict_oof(X_train, X_angle_train, y_train, X_test, X_angle_test, base_models, stacker, n_splits = 5)
print( 'FITTING MODELS AND MAKING OOF PREDICTIONS' )
test_predictions = fit_and_predict_oof( X, X_a, y, X_test, X_a_test, base_models, stacker, n_splits = len(base_models))

m, s = divmod( time.time() - start_time, 60 )
print( 'Model fitting done. Total time: {}m {}s'.format(int(m), int(s)) )

submission = pd.DataFrame()
submission[ 'id' ] = test_data[ 'id' ]
submission[ 'is_iceberg' ] = test_predictions.reshape( (test_predictions.shape[0]) )

PREDICTION_SAVE_PATH = '../submissions/test_submission_xgb_stacker.csv'
submission.to_csv( PREDICTION_SAVE_PATH, index = False )
print( 'Wrote test results to file %s' % str(PREDICTION_SAVE_PATH) )