import pandas as pd
import numpy as np
import os, json
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, BatchNormalization, Flatten, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.random import set_seed

from araus_utils import make_augmented_soundscapes, make_logmel_spectrograms

class ARAUS_Sequence_from_audio(Sequence):
    '''
    Generates sequence of batches from a given dataframe of
    responses, as well as additional data on soundscapes and
    maskers. Data is generated from raw audio, which is very
    slow.

    ======
    Inputs
    ======
    responses : pandas DataFrame
        As per csv documentation. Sequence will be run across
        rows of responses (which means that rows of interest
        should be extracted BEFORE calling the constructor).
    soundscapes : pandas DataFrame
        As per csv documentation.
    maskers : pandas DataFrame
        As per csv documentation.
    batch_size : int
        The number of samples for each batch
    shuffle : bool
        If True, will shuffle order of samples in each batch
        for every epoch. If False, order of samples in each
        batch is the same for all epochs.
    seed_val : int
        The seed to set for shuffling the order of samples
        (if shuffle is True).
    verbose : bool
        If True, prints status messages
    make_augmented_soundscapes_kwargs : dict
        Keyword arguments to pass to make_augmented_soundscapes
        when __getitem__ is called.
    make_logmel_spectrograms_kwargs : dict
        Keyword arguments to pass to make_logmel_spectrograms
        when __getitem__ is called.
        
    =======
    Example
    =======
    import pandas as pd
    import os
    
    metadata_dir = os.path.join('..','data') # Folder containing all the metadata files; edit as necessary for your system.
    maskers = pd.read_csv(os.path.join(f'{metadata_dir}','maskers.csv'))
    soundscapes = pd.read_csv(os.path.join(f'{metadata_dir}','soundscapes.csv'))
    responses = pd.read_csv(os.path.join(f'{metadata_dir}','responses.csv'), dtype = {'participant':str})
    
    # TEST SET
    df_test = responses[(responses['fold_r'] == 0) & (responses['participant'] == '10001')].set_index(['soundscape','masker','smr'])
    test_labels = responses[responses['fold_r'] == 0].groupby(['soundscape','masker','smr']).mean() # Average labels across all test set participants because they were presented with identical stimuli
    attributes = ['pleasant', 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying', 'monotonous']
    df_test[attributes] = test_labels[attributes]
    df_test.reset_index(inplace=True)

    # CROSS-VALIDATION SET
    df_xval = [responses[responses['fold_r'] == i] for i in range(1,6)]

    # COMBINED CROSS-VALIDATION & TEST SET
    df_folds = [df_test]
    df_folds.extend(df_xval)
    
    for idx, df_fold in enumerate(df_folds):
        print(f'Making features for fold #{idx}/{len(df_folds)-1}...')
        out_fpath_features = os.path.join('..','features',f'fold_{idx}_features.npy')
        out_fpath_labels   = os.path.join('..','features',f'fold_{idx}_labels.npy')

        if not os.path.exists(out_fpath_features):
            _ = make_features(df_fold, soundscapes, maskers, out_fpath = out_fpath_features)
        else:
            print(f'{out_fpath_features} already exists, skipping generation of features for fold #{idx}...')

        if not os.path.exists(out_fpath_labels):
            _ = make_labels(df_fold, out_fpath = out_fpath_labels)
        else:
            print(f'{out_fpath_features} already exists, skipping generation of labels for fold #{idx}...')
    '''
    def __init__(self, responses, soundscapes, maskers,
                       batch_size = 32,
                       shuffle = True,
                       seed_val = 2021,
                       verbose = True,
                       make_augmented_soundscapes_kwargs = {'mode': 'return',
                                                            'verbose': 0},
                       make_logmel_spectrograms_kwargs = {'n_fft': 4096,
                                                          'hop_length': 2048,
                                                          'n_mels': 64,
                                                          'verbose': 0}):
        self.responses = responses
        self.soundscapes = soundscapes
        self.maskers = maskers
        self.n_samples = len(responses)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed_val = seed_val
        np.random.seed(self.seed_val) # Seed random state based on seed_val
        self.verbose = verbose
        self.order = np.random.permutation(self.n_samples) if self.shuffle else np.arange(self.n_samples)
        self.make_augmented_soundscapes_kwargs = make_augmented_soundscapes_kwargs
        self.make_logmel_spectrograms_kwargs = make_logmel_spectrograms_kwargs
    
    def __getitem__(self, idx):
        '''
        Returns the idx-th batch of samples.
        '''
        # GET DATAFRAME ROWS CORRESPONDING TO CURRENT BATCH OF SAMPLES
        batch_idxs = self.order[idx*self.batch_size : min((idx+1)*self.batch_size, self.n_samples)]
        batch_responses = self.responses.iloc[batch_idxs,:]
        
        # GET AUGMENTED SOUNDSCAPES AS NUMPY ARRAY IN TIME DOMAIN
        n_failures, augmented_soundscapes = make_augmented_soundscapes(batch_responses, self.soundscapes, self.maskers, **self.make_augmented_soundscapes_kwargs)
        
        if self.verbose and n_failures > 0: print(f'Warning: Failed to generate {n_failures} augmented soundscapes for batch #{idx+1}')
        
        # CONVERT AUGMENTED SOUNDSCAPES TO SPECTROGRAMS
        augmented_spectrograms = []
        for augmented_soundscape in augmented_soundscapes:
            augmented_spectrogram = make_logmel_spectrograms(augmented_soundscape.T, **self.make_logmel_spectrograms_kwargs)
            augmented_spectrograms.append(np.expand_dims(augmented_spectrogram,0))
        augmented_spectrograms = np.concatenate(augmented_spectrograms)
        
        # GET GROUND-TRUTH LABELS (ISO PLEASANTNESS) FOR SPECTROGRAMS
        attributes = ['pleasant', 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying', 'monotonous'] # Define attributes to extract from dataframes
        ISOPl_weights = [1,0,-np.sqrt(2)/2,np.sqrt(2)/2, 0, np.sqrt(2)/2,-1,-np.sqrt(2)/2] # Define weights for each attribute in attributes in computation of ISO Pleasantness
        ISOPls = ((batch_responses[attributes] * ISOPl_weights).sum(axis=1)/(4+np.sqrt(32))).values

        return augmented_spectrograms, ISOPls # Returns a batch of (inputs, labels) = (augmented_spectrograms, ISO_Pls)
        
    def __len__(self):
        '''
        Returns number of batches in the Sequence (i.e. number
        of batches per epoch)
        '''
        return self.n_samples // self.batch_size + (self.n_samples % self.batch_size > 0)
    
    def on_epoch_end(self):
        '''
        Method called at the end of every epoch.
        Shuffles samples for next batch if self.shuffle is True.
        '''
        if self.shuffle:
            self.order = np.random.permutation(self.n_samples)
            
#==============================================================#===============================================================#

class ARAUS_Sequence_from_npy(Sequence):
    '''
    Generates sequence of batches from precomputed spectrogram
    arrays.
    
    ======
    Inputs
    ======
    responses : pandas DataFrame
        As per csv documentation. Sequence will be run across
        rows of responses (which means that rows of interest
        should be extracted BEFORE calling the constructor).
    npy_dir : str
        The directory where the npy files are stored (or that
        will be written to if the precompute method is called).
    batch_size : int
        The number of samples for each batch
    shuffle : bool
        If True, will shuffle order of samples in each batch
        for every epoch. If False, order of samples in each
        batch is the same for all epochs.
    seed_val : int
        The seed to set for shuffling the order of samples
        (if shuffle is True).
    verbose : bool
        If 0, prints nothing. If 1, prints status messages.
        If 2, additionally prints warning messages.
    '''
    def __init__(self, responses,
                       npy_dir = os.path.join('..','features'),
                       batch_size = 32,
                       shuffle = True,
                       seed_val = 2021,
                       verbose = 1):
        
        self.responses = responses
        self.n_samples = len(responses)
        self.npy_dir = npy_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed_val = seed_val
        np.random.seed(self.seed_val) # Seed random state based on seed_val
        self.verbose = verbose
        self.order = np.random.permutation(self.n_samples) if self.shuffle else np.arange(self.n_samples)
    
    def __getitem__(self, idx):
        '''
        Returns the idx-th batch of samples.
        '''
        # GET DATAFRAME ROWS CORRESPONDING TO CURRENT BATCH OF SAMPLES
        batch_idxs = self.order[idx*self.batch_size : min((idx+1)*self.batch_size, self.n_samples)]
        batch_responses = self.responses.iloc[batch_idxs,:]
        
        # LOAD CURRENT BATCH OF FEATURES
        augmented_spectrograms = []
        for idx, (_, row) in enumerate(batch_responses.iterrows()): # Dataframe's index may be out of order so we don't use it (and assign it to _ instead).
            # GET NECESSARY DATA FROM CURRENT ROW
            participant_id = row['participant']
            fold = row['fold_r']
            soundscape_fname = row['soundscape']
            masker_fname = row['masker']
            smr = row['smr']
            stimulus_index = row['stimulus_index']

            # READ FROM EXISTING NPY FILE
            in_fname = f'fold_{fold}_participant_{participant_id:>05}_stimulus_{stimulus_index:02d}.npy'
            in_fpath = os.path.join(self.npy_dir, in_fname)
            
            augmented_spectrograms.append(np.load(in_fpath))
        augmented_spectrograms = np.array(augmented_spectrograms)
        
        # GET GROUND-TRUTH LABELS (ISO PLEASANTNESS) FOR SPECTROGRAMS
        attributes = ['pleasant', 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying', 'monotonous'] # Define attributes to extract from dataframes
        ISOPl_weights = [1,0,-np.sqrt(2)/2,np.sqrt(2)/2, 0, np.sqrt(2)/2,-1,-np.sqrt(2)/2] # Define weights for each attribute in attributes in computation of ISO Pleasantness
        ISOPls = ((batch_responses[attributes] * ISOPl_weights).sum(axis=1)/(4+np.sqrt(32))).values

        return augmented_spectrograms, ISOPls # Returns a batch of (inputs, labels) = (augmented_spectrograms, ISO_Pls)
        
    def __len__(self):
        '''
        Returns number of batches in the Sequence (i.e. number
        of batches per epoch).
        '''
        return self.n_samples // self.batch_size + (self.n_samples % self.batch_size > 0)
    
    def precompute(self, soundscapes, maskers,
                   overwrite = False,
                   make_augmented_soundscapes_kwargs = {'mode': 'return',
                                                        'verbose': 0},
                   make_logmel_spectrograms_kwargs = {'n_fft': 4096,
                                                      'hop_length': 2048,
                                                      'n_mels': 64,
                                                      'verbose': 0}):
        '''
        Given dataframes representing some responses (self),
        soundscapes, and maskers, make features of all augmented
        soundscapes for which responses were obtained (as
        time-frequency arrays/.npy files). As many sets of features
        as there are rows in the self.responses dataframe will be
        generated.

        ======
        Inputs
        ======
        soundscapes : pandas DataFrame
            Contains at least the following columns:
                - 'soundscape' : str
                - 'gain_s' : float
                - 'insitu_leq' : float
            All entries in the column 'soundscape' should be unique.
        maskers : pandas DataFrame
            Contains at least the following columns:
                - 'masker' : str
                - 'gain_##dB' : float
                - 'leq_at_gain_##dB' : float
            All entries in the column 'masker' should be unique.
            ## must include all integers in the interval [46, 83].
        overwrite : bool
            Whether to overwrite existing files with identical file
            names.
        make_augmented_soundscapes_kwargs : dict
            Keyword arguments to pass to make_augmented_soundscapes.
        make_logmel_spectrograms_kwargs : dict
            Keyword arguments to pass to make_logmel_spectrograms.

        =======
        Returns
        =======
        None

        ============
        Dependencies
        ============
        pandas (as pd), numpy (as np), make_augmented_soundscapes
        (from araus_utils), make_logmel_spectrograms
        (from araus_utils), os
        '''
        # MAKE OUTPUT DIRECTORY IF IT DOESN'T ALREADY EXIST
        if not os.path.exists(self.npy_dir): os.makedirs(self.npy_dir)
        
        for idx, (_, row) in enumerate(self.responses.iterrows()): # Dataframe's index may be out of order so we don't use it (and assign it to _ instead).
            if self.verbose > 0: print(f'Progress: {idx+1}/{self.n_samples}.', end = '\r')
            
            # GET NECESSARY DATA FROM CURRENT ROW
            participant_id = row['participant']
            fold = row['fold_r']
            soundscape_fname = row['soundscape']
            masker_fname = row['masker']
            smr = row['smr']
            stimulus_index = row['stimulus_index']

            # CHECK IF FILE EXISTS
            out_fname = f'fold_{fold}_participant_{participant_id:>05}_stimulus_{stimulus_index:02d}.npy'
            out_fpath = os.path.join(self.npy_dir, out_fname)
            
            if os.path.exists(out_fpath) and (not overwrite):
                if self.verbose > 1: print(f'Warning: {out_fpath} already exists, skipping its generation...') 
                continue # Skip all processing steps to save time.
            
            # MAKE SPECTROGRAMS
            _, augmented_soundscape = make_augmented_soundscapes(row.to_frame().T, soundscapes, maskers,
                                                                 **make_augmented_soundscapes_kwargs)
            logmel_spectrograms = make_logmel_spectrograms(input_data = np.squeeze(augmented_soundscape).T,
                                                           **make_logmel_spectrograms_kwargs).transpose([1,0,2])
            
            # WRITE SPECTROGRAMS TO FILE
            np.save(out_fpath,logmel_spectrograms.astype(np.float32))
    
    def on_epoch_end(self):
        '''
        Method called at the end of every epoch.
        Shuffles samples for next batch if self.shuffle is True.
        '''
        if self.shuffle:
            self.order = np.random.permutation(self.n_samples)

#==============================================================#===============================================================#

def make_baseline_model():
    # Input layer
    spec_input = Input(shape = (644,64,2),name='Spec_Input')

    # CNN layer #1
    S = Conv2D(16,(7,7),padding='same',name='Conv2D_1')(spec_input)
    S = BatchNormalization(name='BN_1')(S)
    S = Activation('relu',name='Relu_1')(S)

    # CNN layer #2
    S = Conv2D(16,(7,7),padding='same',name='Conv2D_2')(S)
    S = BatchNormalization(name='BN_2')(S)
    S = Activation('relu',name='Relu_2')(S)
    S = MaxPooling2D((5,5),name='MP2D_2')(S)
    S = Dropout(0.3,name='Dropout_2')(S)

    # CNN layer #3
    S = Conv2D(32,(7,7),padding='same',name='Conv2D_3')(S)
    S = BatchNormalization(name='BN_3')(S)
    S = Activation('relu',name='Relu_3')(S)
    S = MaxPooling2D((4,100), padding='same',name='MP2D_3')(S)
    S = Dropout(0.3,name='Dropout_3')(S)

    S = Flatten(name='Flatten')(S)

    # Dense layer #1
    S = Dense(100, activation = 'relu',name='Dense')(S)
    S = Dropout(0.3,name='Dropout_Dense')(S)

    # Output layer
    output = Dense(1, activation='linear',name='ISOPl')(S) # Two regression neurons: one for pleasantness and one for eventfulness

    # Instantiate model
    baseline_model = Model(inputs=spec_input, outputs=output, name='Baseline') # About 140k params
    
    return baseline_model

#==============================================================#===============================================================#

def train_model(model,
                train_data = None,
                train_labels = None,
                validation_data = None,
                test_data = None,
                test_labels = None,
                seed = 1,
                learning_rate = 0.0001, # DCASE uses 0.001
                mtype = 'baseline',
                run = 1,
                val_fold = 1,
                save_best_only = False,
                save_weights_only = True,
                patience = 10,
                restore_best_weights = True,
                batch_size = None, # DCASE uses 16, don't specify for generators
                epochs = 2, # DCASE uses 100
                workers = 1,
                use_multiprocessing = False,
                verbose = True):
    '''
    model : Model instance
        A tensorflow.keras.models.Model instance or similar
        object with methods 'compile', 'fit', 'evaluate'.
    train_data : training data
        Will be passed to the argument x in model.fit.
    train_labels : training labels
        Will be passed to the argument y in model.fit.
    validation_data : validation data & labels
        Will be passed to the identically-named argument in
        model.fit
    test_data : training data
        Will be passed to the argument x in model.evaluate.
    test_labels : training labels
        Will be passed to the argument y in model.evaluate.
    seed : int
        The random seed used for training the model
    learning_rate : float
        The learning rate for the Adam optimizer used to train
        the model
    mtype : str
        Used in prefix for model file names, should refer
        somewhat to the type of model trained.
    run : int
        Used in prefix for model file names & model directory,
        should refer to the current run number.
    val_fold : int
        Used in prefix for model file names, should refer to the
        fold number used for validation.
    save_best_only : bool
        Will be passed to the identically-named argument in
        tensorflow.keras.callbacks.ModelCheckpoint
    save_weights_only : bool
        Will be passed to the identically-named argument in
        tensorflow.keras.callbacks.ModelCheckpoint
    patience : int
        Will be passed to the identically-named argument in
        tensorflow.keras.callbacks.EarlyStopping
    restore_best_weights : bool
        Will be passed to the identically-named argument in
        tensorflow.keras.callbacks.EarlyStopping
    batch_size : int
        Will be passed to the identically-named arguments in
        model.fit and model.evaluate
    epochs : int
        Will be passed to the identically-named argument in
        model.fit
    workers : int
        Will be passed to the identically-named argument in
        model.fit and model.evaluate
    use_multiprocessing : bool
        Will be passed to the identically-named argument in
        model.fit and model.evaluate
    verbose : bool
        Will be passed to the identically-named arguments in
        tensorflow.keras.callbacks.ModelCheckpoint,
        tensorflow.keras.callbacks.EarlyStopping, model.fit,
        and model.evaluate
    '''
    # Set random seed
    set_seed(seed)

    # Compile model
    model.compile(loss = 'mean_squared_error',
                  optimizer = Adam(learning_rate = learning_rate))

    # Make folders to store model checkpoints
    model_dir = os.path.join('..','models',f'models_{run:02d}')
    prefix = f'model_{mtype}_run_{run:02d}_fold_{val_fold}_'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Prepare checkpoint callback
    model_fpath = model_dir + os.sep + prefix + 'epoch-{epoch:03d}-loss-{val_loss:.4f}.h5'
    checkpoint = ModelCheckpoint(model_fpath,
                                 monitor = 'val_loss',
                                 verbose = verbose,
                                 save_best_only = save_best_only,
                                 save_weights_only = save_weights_only)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   verbose=verbose,
                                   restore_best_weights=restore_best_weights)
    callbacks_list = [checkpoint, early_stopping]

    # Fit model
    model_hist = model.fit(x = train_data,
                           y = train_labels,
                           batch_size = batch_size,
                           epochs = epochs,
                           initial_epoch = 0, # The label of the first epoch upon calling this function is one greater than this.
                                              # Call a number k > 0 to label the epochs starting from k+1 (useful for resuming training).
                           verbose = verbose,
                           validation_data = validation_data,
                           callbacks = callbacks_list,
                           shuffle = True,
                           workers = workers,
                           use_multiprocessing = use_multiprocessing)
    
    with open(os.path.join(model_dir,f'model_{mtype}_run_{run:02d}_fold_{val_fold}_train_val_history.json'), 'w') as fp:
        json.dump(model_hist.history, fp)
    
    loss_dict = model.evaluate(x = test_data,
                               y = test_labels,
                               batch_size = batch_size,
                               verbose = verbose,
                               workers = workers,
                               use_multiprocessing = use_multiprocessing,
                               return_dict = True)
    
    with open(os.path.join(model_dir,f'model_{mtype}_run_{run:02d}_fold_{val_fold}_test_history.json'), 'w') as fp:
        json.dump(loss_dict, fp)
    
    return model_hist.history, loss_dict

