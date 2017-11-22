import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def plot_band_samples( data, band = 1, title = None ):
    fig = plt.figure( 1, figsize=(15, 15) )
    for i in range(9):
        ax = fig.add_subplot( 3, 3, i + 1 )
        arr = np.reshape( np.array(data.iloc[i, band - 1]), (75, 75) )
        ax.imshow( arr, cmap='inferno' )
        fig.suptitle( title )

    plt.show()

def plot_all_bands( data, title = None ):
    fig = plt.figure( 1, figsize = (15, 15) )
    count = 1
    for i in range(3):
        for j in range(3):
            ax = fig.add_subplot( 3, 3, count )
            ax.imshow( data[i, :, :, j], cmap = 'inferno' )
            count += 1
            if i == 0:
                if j == 0:
                    ax.set_title( 'Band 1' , fontsize = 12)
                elif j == 1:
                    ax.set_title( 'Band 2', fontsize = 12 )
                elif j == 2:
                    ax.set_title( 'Average', fontsize = 12 )
    fig.suptitle( title, fontsize = 14, fontweight = 'bold' )
    plt.show()

def make_plots( data, band_samples = True, all_bands = True ):
    ships = data[ data.is_iceberg == 0 ].sample( n = 9, random_state = 42 )
    icebergs = data[ data.is_iceberg == 1 ].sample( n = 9, random_state = 42 )

    np_ships = generate_data( ships )
    np_icebergs = generate_data( icebergs )

    if band_samples:
        plot_band_samples( ships, band = 2, title = 'Ship image samples' )
        plot_band_samples( icebergs, band = 2, title = 'Iceberg image samples' )

    if all_bands:
        plot_all_bands( np_ships, 'Image bands for ships' )
        plot_all_bands( np_icebergs, 'Image bands for icebergs' )


def get_callbacks( weight_save_path, log_path = './logs', no_improv_epochs = 10, min_delta = 1e-4 ):
    es = EarlyStopping( 'val_loss', patience = no_improv_epochs, mode = 'min', min_delta = min_delta )
    ms = ModelCheckpoint( weight_save_path, 'val_loss', save_best_only = True ) 
    #ts = TensorBoard( log_dir = './logs', batch_size = 32 )
    rl = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.4, verbose = 1,
                           patience = int(no_improv_epochs/2), min_lr = 5e-6)

    return [ es, ms, rl ]#, ts ]

def generate_data( data ):
    X_band_1=np.array( [np.array(band).astype(np.float32).reshape(75, 75) 
                        for band in data['band_1']] )
    X_band_2=np.array( [np.array(band).astype(np.float32).reshape(75, 75) 
                        for band in data['band_2']] )
    X = np.concatenate( [X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis], \
                        ((X_band_1 + X_band_2)/2)[:, :, :, np.newaxis]], axis=-1 )
    return X

def augment_data( generator, X1, X2, y, batch_size = 32 ):
    generator_seed = np.random.randint( 9999 )
    gen_X1 = generator.flow( X1, y, batch_size = batch_size, seed = generator_seed )
    gen_X2 = generator.flow( X1, X2, batch_size = batch_size, seed = generator_seed )

    while True:
        X1i = gen_X1.next()
        X2i = gen_X2.next()

        yield [ X1i[0], X2i[1] ], X1i[1]