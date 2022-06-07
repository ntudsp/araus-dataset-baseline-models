import sklearn, os, wget, hashlib, librosa

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf

from zipfile import ZipFile

def assign_clusters(D,W,m=5):
    '''
    Assigns points to clusters given a distance matrix D and
    cluster centres W.
    
    ======
    Inputs
    ======
    D : np.ndarray of shape (n_clusters, n_observations)
        A distance matrix where the (i,j)-th element is the
        distance between the i-th cluster centre and the j-th
        observation.
    W : np.ndarray of shape (n_clusters, n_features)
        A matrix of cluster centre coordinates matrix
    m : int
        The number of samples to assign to each cluster
        
    =======
    Returns
    =======
    cluster_indices : np.ndarray of shape (n_observations,)
        The cluster indices for each observation. In other
        words, cluster_indices[i] is the cluster index of the
        i-th observation.
    
    ============
    Dependencies
    ============
    numpy (as np), count (from araus_utils)
    '''
    D = D.copy() # D will be altered later, so we prevent permanent changes to the array passed to this function by creating a copy first.
    cluster_indices = -np.ones(D.shape[1],dtype = np.int32)
    for cidx, coordinates in enumerate(W): # Rows of (n_features,) in coordinates, cluster index in cidx.
        # CHOOSE THE m CLOSEST POINTS TO coordinates
        I = np.argsort(D[cidx])[:m] # shape = (m,) of indices to extract for cluster cidx.

        # ASSIGN THEIR INDICES AS cidx
        cluster_indices[I] = cidx

        # SET DISTANCES TO INFINITY TO PREVENT THEM FROM BEING CHOSEN AGAIN
        D[:,I] = np.Inf

    assert np.all(count(cluster_indices, D.shape[0]) == m) and np.all(cluster_indices >= 0) # For debugging.
    
    return cluster_indices

#==============================================================#===============================================================#

def assign_folds(C,m=5,seed_val=2021):
    '''
    Assign samples into folds randomly given the clusters in C
    
    fold_indices[i] is the fold that the i-th sample belongs to.
    '''
    fold_indices = np.empty(len(C),dtype=np.int32)
    I = np.argsort(C)

    np.random.seed(seed_val)
    R = list(np.random.permutation(m))
    for idx in I:
        if len(R) == 0:
            R = list(np.random.permutation(m))
        fold_indices[idx] = R.pop()+1
        
    return fold_indices

#==============================================================#===============================================================#

def autolabel(containers, axs, i = 0, width_scales = 2.0, height_scales = 3.0, max_m = np.inf, max_n = 4, f_height = lambda height: height):
    """
    Attach a text label near each bar displaying its height (or
    any function of it)
    
    ======
    Inputs
    ======
    containers : list of plt.BarContainer
        Each BarContainer object should in turn contain
        rectangle objects (<BarContainer object of n artists>)
    axs : plt.axes._subplots.AxesSubplot
          Axes object returned by plt.plot() or plt.subplots
          (fig, axs = plt.subplots(...))
    i : int
        The subplot number to do the autolabelling on (if using
        subplots)
    width_scales : float, 1-D np.array, or 2-D np.array
        If float, will apply scale to all bars in width
        If 1-D np.array, will apply corresponding scale to each
        row of bars
        If 2-D np.array, will apply corresponding scale to each
        bar (row-column pair).
    height_scales : float, 1-D np.array, or 2-D np.array
        Same as width_scales but for height
    max_m : float
        Maximum number of rows of bars to add text on.
    max_n : float
        Maximum number of columns of bars to add text on.
    f_height : callable (function)
        Any function taking in the height and returning a
        numerical value to place as each bar's text.
    
    =======
    Returns
    =======
    None
    
    ============
    Dependencies
    ============
    numpy (as np), matplotlib.pyplot (as plt)
    """
    M = len(containers)
    N = len(containers[0])
    
    if type(width_scales) == float:
        width_scales = np.array([[width_scales]*N]*M) # Make the array M-by-N of all that element
    elif len(np.array(width_scales).shape) == 1:
        width_scales = np.pad(np.array(width_scales), pad_width = ( 0, max(0,M-len(width_scales)) ), mode = 'edge')
        width_scales = np.repeat(np.expand_dims(np.array(width_scales),axis=-1),N,axis=-1) # Assume that the scales are same for same row.
    else:
        width_scales = np.array(width_scales)
        
    if type(height_scales) == float:
        height_scales = np.array([[height_scales]*N]*M)
    elif len(np.array(height_scales).shape) == 1:
        height_scales = np.pad(np.array(height_scales), pad_width = ( 0, max(0,M-len(height_scales)) ), mode = 'edge')
        height_scales = np.repeat(np.expand_dims(np.array(height_scales),axis=-1),N,axis=-1)
    else:
        height_scales = np.array(height_scales)
    
    for m, container in enumerate(containers):
        if m > max_m:
            continue
            
        for n, rect in enumerate(container):
            if n > max_n:
                continue
                
            x, y = rect.get_xy() # bottom left corner coordinates
            height = rect.get_height()
            height_scale = height_scales[m,n]
            width = rect.get_width()
            width_scale = width_scales[m,n]
            
            if type(axs) == np.ndarray: axs = axs[i]
                
            axs.text(x + width/width_scale, y + height/height_scale,
                '%.0f%%' % f_height(height),
                ha='center', va='bottom', fontsize=14)

#==============================================================#===============================================================#

def count(X,n):
    '''
    This function counts the number of occurrences of each
    integer in an array of non-negative integers X, assuming
    that only integers in [0,n) potentially (but not
    necessarily) appear in X.
    
    ======
    Inputs
    ======
    X : np.ndarray of shape (k,)
        A k-element numpy array containing only non-negative
        integers from 0 (inclusive) to n (exclusive).
    
    n : int
        The maximum integer that potentially (but not
        necessarily) appears in X.
    
    =======
    Returns
    =======
    counts : np.ndarray of shape (n,)
        A numpy array containing the counts of each integer
        in X. In other words, count[i] is the number of
        elements in X equal to i.
    
    ============
    Dependencies
    ============
    numpy (as np)
    
    =======
    Example
    =======
    >>> count(np.array([0,1,1,3,3,3]),n=5)
    array([1, 2, 0, 3, 0])
    '''
    counts = np.zeros(n).astype(np.int32)
    for i in X:
        counts[i] += 1
    return counts

#==============================================================#===============================================================#

def disp_axes_details(ax):
    '''
    Prints axes locations (i.e. (x,y) coordinates) given a set
    of plot axes.
    
    ======
    Inputs
    ======
    ax : Axes object
        Axes object returned by matplotlib plot functions,
        e.g. <class 'matplotlib.axes._subplots.AxesSubplot'>
        
    =======
    Returns
    =======
    None
        
    ============
    Dependencies
    ============
    matplotlib.pyplot (as plt)
    '''
    print('Axes details in format n_rows_of_bars, n_cols_of_bars | curr_row_of_bar, curr_col_of_bar, x_coordinate, y_coordinate:')
    m = len(ax.containers)
    for i in range(m):
        n = len(ax.containers[i])
        for j in range(n):
            print(f'{m:3d}, {n:3d} | {i}, {j}, {ax.containers[i][j].get_x():6.2f}, {ax.containers[i][j].get_y():6.2f}')

#==============================================================#===============================================================#

def download_file(url, out, max_tries = 10, checksum = None, overwrite = 1, verbose = True):
    '''
    Downloads a file using wget.download with some basic
    validation.
    
    ======
    Inputs
    ======
    url : str
        The url to download the file from (as passed to
        wget.download())
    out : str
        The filepath the save the downloaded file to (as
        passed to wget.download())
    max_tries : int
        The maximum number of tries to attempt a download if
        any error(s) occur during download.
    checksum : None or str
        If None, peforms no checksum verification of the file
        at out.
        If str, verifies if this matches the 64-byte BLAKE2
        hash of the file at out (as returned by
        hashlib.blake2b(open(out,'rb').read()).hexdigest()).
    overwrite : int in [0, 1, 2]
        If 0, does not overwrite existing file at out
        regardless of its checksum.
        If 1, overwrites existing file at out iff its checksum
        verification fails.
        If 2, overwrites existing file at out regardless of
        its checksum.
    verbose : bool
        If True, prints status and error messages. If False,
        prints nothing.
    
    =======
    Returns
    =======
    n_tries : int
        Number of unsuccessful tries before sucessful
        download (and checksum verification).
        
    ============
    Dependencies
    ============
    os, wget, hashlib
    '''
    n_tries = 0
    while n_tries < max_tries:
        # Attempt to download file
        try:
            # CONVERT FILEPATH ACCORDING TO OS
            out = os.path.relpath(out)
            
            # MAKE OUTPUT DIRECTORY IF IT DOESN'T ALREADY EXIST
            out_dir = os.path.dirname(out)
            if (not os.path.exists(out_dir)) and len(out_dir) > 0:
                os.makedirs(out_dir)

            # CHECK IF FILE EXISTS AND HANDLE ACCORDINGLY
            if os.path.exists(out):
                if overwrite == 0:
                    if verbose: print(f'{out} already exists. Skipping download...')
                    break
                elif overwrite == 1:
                    if checksum is None:
                        if verbose: print(f'{out} already exists. Skipping download...')
                        break
                    elif hashlib.blake2b(open(out,'rb').read()).hexdigest() == checksum:
                        if verbose: print(f'{out} already exists and matches expected checksum. Skipping download...')
                        break
                    else:
                        if verbose: print(f'{out} already exists but does not match expected checksum. Overwriting...')
                        os.remove(out)
                else: # overwrite == 2
                    os.remove(out)
            
            # DOWNLOAD FILE AT url (WITH PROGRESS BAR IF verbose)
            if verbose:
                print(f'Attempting to download from {url} to {out} (try {n_tries+1}/{max_tries})...')
                wget.download(url,out)
                print()
            else:
                wget.download(url,out,bar=None)

            # CHECK IF FILE DATA MATCHES EXPECTED CHECKSUM
            if checksum is not None:
                assert hashlib.blake2b(open(out,'rb').read()).hexdigest() == checksum
            
            # EXIT WHILE LOOP IF NO ERRORS
            break
        except AssertionError:
            n_tries += 1
            if verbose: print(f'Error: checksum of {out} does not match expected checksum.')
            continue
        except Exception as e:
            n_tries += 1
            if verbose: print(f'Error: {e}')
            continue
        
    return n_tries

#==============================================================#===============================================================#

def make_augmented_soundscapes(responses, soundscapes, maskers,
                               mode = 'file',
                               soundscape_dir = os.path.join('..','soundscapes'), # In Windows, this would be '..\\soundscapes' and in MacOSX/Linus, this would be '../soundscapes'
                               masker_dir = os.path.join('..','maskers'),
                               out_dir = os.path.join('..','soundscapes_augmented'),
                               out_format = 'wav',
                               overwrite = False,
                               stop_upon_failure = False,
                               verbose = 1):
    '''
    Given dataframes representing some responses, soundscapes,
    and maskers, make all augmented soundscapes for which
    responses were obtained (as time-domain arrays/audio
    files). As many augmented soundscapes as there are rows
    in the responses dataframe will be generated.
    
    ======
    Inputs
    ======
    responses : pandas DataFrame
        Contains at least the following columns:
            - 'participant' : str
            - 'fold_r' : int
            - 'soundscape' : str
            - 'masker' : str
            - 'smr' : int
            - 'stimulus_index' : int
        All entries in the column 'soundscape' must exist in the
        identically-named column in soundscapes.
        All entries in the column 'masker' must exist in the
        identically-named column in maskers.
        Pairs of entries in the columns ('participant',
        'stimulus_index') must be unique. 
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
    mode : str in ['file','return','both']
        The mode in which to run the function.
        'file' will write the augmented soundscapes to
        individual files (to the directory out_dir, in the 
        format out_format).
        'return' will make the function return all augmented
        soundscapes as a (len(responses),n_samples,n_channels)
        numpy array.
        'both' will write the augmented soundscapes to
        individual files jAND return all augmented soundscapes
        as a numpy array (i.e. both 'file' and 'return').
    soundscape_dir : str
        The directory where the soundscape files are stored.
    masker_dir : str
        The directory where the masker files are stored.
    out_dir : str
        The directory to write the augmented soundscapes to
        file to.
    out_format : str
        The audio file format in which to write the augmented
        soundscapes to file.
    overwrite : bool
        If True, will overwrite existing files when outputting
        augmented soundscapes with filenames matching existing
        files. If False, will not overwrite existing files.
    stop_upon_failure : bool
        If True, any error in making the augmented
        soundscapes will stop the function. If False, then
        the function will continue regardless of errors in
        making the augmented soundscapes, until attempts to
        make all augmented soundscapes have been made.
    verbose : int in [0,1,2]
        If 0, prints nothing. If 1, prints basic status
        messages. If 2, prints detailed status messages.
        
    =======
    Returns
    =======
    n_failures : int
        The number of augmented soundscapes which failed to be
        made. Ideally, this should be 0 for successful output
        of all augmented soundscapes.
        
    augmented_soundscapes : np.ndarray
        If mode is 'return' or 'both', this is a numpy array
        of shape (len(responses),n_samples,n_channels), where
        the (i,j,k)-th element is the j-th sample of the k-th
        channel of the augmented soundscape made using data
        from the i-th row of responses.
        If mode is 'file', this is an empty numpy array of
        shape (0,0,0).
    
    ============
    Dependencies
    ============
    pandas (as pd), numpy (as np), soundfile (as sf), os
    '''
    # CHECK VALID MODE
    if mode not in ['file','return','both']:
        if verbose > 0: print('Warning: Invalid argument entered for mode, defaulting to "file"...')
        mode = 'file'

    # MAKE OUTPUT DIRECTORY IF IT DOESN'T ALREADY EXIST
    if (not os.path.exists(out_dir)) and len(out_dir) > 0:
        os.makedirs(out_dir)
    
    # MAKE AUGMENTED SOUNDSCAPES
    n_failures = 0 # Will count the number of failed attempts.
    augmented_soundscapes = np.zeros((0,0,0)) # Will be used to store the augmented soundscapes.
    n_tracks = len(responses)
    if verbose > 0: print(f'Making {n_tracks} augmented soundscapes in {out_dir}...')
    for idx, (_, row) in enumerate(responses.iterrows()): # Dataframe's index may be out of order so we don't use it (and assign it to _ instead).
        if verbose > 0: print(f'Progress: {idx+1}/{n_tracks}.')
        
        try:
            # GET NECESSARY DATA FROM RESPONSES FOR CALIBRATION
            participant_id = row['participant']
            fold = row['fold_r']
            soundscape_fname = row['soundscape']
            masker_fname = row['masker']
            smr = row['smr']
            stimulus_index = row['stimulus_index']
            
            # CHECK IF FILE EXISTS
            out_fname = f'fold_{fold}_participant_{participant_id:>05}_stimulus_{stimulus_index:02d}.{out_format}'
            out_fpath = os.path.join(out_dir, out_fname)
            
            if (mode == 'file') and os.path.exists(out_fpath) and (not overwrite):
                if verbose > 0: print(f'Warning: {out_fpath} already exists, skipping its generation...') 
                continue # Skip all read and write steps to save processing time.

            # GET SOUNDSCAPE CALIBIRATION PARAMETERS
            gain_s = soundscapes[soundscapes['soundscape'] == soundscape_fname]['gain_s'].squeeze() # Gain to apply to soundscape
            leq_s = soundscapes[soundscapes['soundscape'] == soundscape_fname]['insitu_leq'].squeeze()

            # GET MASKER CALIBRATION PARAMETERS
            leq_m = leq_s - smr
            leq_m_round = np.round(leq_m,0).astype(int) # Closest integer dB value to the desired masker level that we know the correct calibration for.
            gain_c = maskers[maskers['masker'] == masker_fname][f'gain_{leq_m_round}dB'].squeeze() # This is the gain for that value,...
            leq_c = maskers[maskers['masker'] == masker_fname][f'leq_at_gain_{leq_m_round}dB'].squeeze() # and this is the Leq that was actually measured after calibration.
            gain_m = gain_c*(10**((leq_m-leq_c)/20)) # We estimate the true gain to apply to the masker within the range of 1 dB.

            # PRINT PARAMETER SUMMARY
            if verbose > 1:
                print(f'Now generating participant {participant_id}, stimulus {stimulus_index} (fold {fold}):')
                print(f'\t{soundscape_fname} (soundscape) + {masker_fname} (masker) @ SMR {smr} dB.')
                print(f'\tSoundscape Leq {leq_s:.2f} dB achieved by setting gain to {gain_s:.2e}.')
                print(f'\tMasker Leq ({leq_m:.2f} dB) achieved by setting gain to {gain_m:.2e} (interpolated from known gain {gain_c:.2e} giving Leq of {leq_c:.2f} dB).')

            # LOAD SOUNDSCAPE
            soundscape_fpath = os.path.join(soundscape_dir, soundscape_fname)
            x_s, sr_s = sf.read(soundscape_fpath)
            if not (x_s.shape == (1323000,2) and sr_s == 44100):
                if verbose > 0: print(f'Warning: Expected (1323000,2) and 44100 for soundscape shape and sampling rate but got {x_s.shape} and {sr_s}')

            # LOAD MASKER
            masker_fpath = os.path.join(masker_dir, masker_fname)
            x_m, sr_m = sf.read(masker_fpath)
            x_m = np.tile(x_m,(2,1)).T # Duplicate masker into two channels
            if not (x_m.shape == (1323000,2) and sr_s == 44100):
                if verbose > 0: print(f'Warning: Expected (1323000,2) and 44100 for masker shape (after duplication) and sampling rate but got {x_m.shape} and {sr_m}')

            # MAKE OUTPUT TRACK (= CURRENT STIMULUS)
            x = gain_s*x_s + gain_m*x_m

            # STORE SAMPLES TO OUTPUT ARRAY IF DESIRED
            if mode in ['return','both']:
                if augmented_soundscapes.shape == (0,0,0): # Means this is the first instance where generation of x was successful (possibly idx == 0 but not necessarily).
                    output_shape = (n_tracks, x.shape[0], x.shape[1])
                    if verbose > 0: print(f'Preallocating a {output_shape} array for return...')
                    augmented_soundscapes = np.zeros(output_shape) # Then we preallocate the output array (without assuming the shape of x beforehand)
                augmented_soundscapes[idx,:,:] = x
                    
            # OUTPUT TRACK TO FILE IF DESIRED
            if mode in ['file','both']:
                if os.path.exists(out_fpath):
                    if overwrite:
                        if verbose > 0: print(f'Warning: {out_fpath} already exists, overwriting it...')
                    else:
                        if verbose > 0: print(f'Warning: {out_fpath} already exists, not overwriting it...')
                        continue # Skip the writing
                if verbose > 1: print(f'\tOutputting stimulus to {out_fpath} @ {min(sr_m,sr_s)} Hz...')
                sf.write(out_fpath,x,min(sr_m,sr_s))
        except Exception as e:
            if stop_upon_failure:
                if verbose > 0: print(f'Error: Failed to make augmented soundscape #{idx+1}/{n_tracks}. Reason for failure: {e}.')
                raise
            else:
                if verbose > 0: print(f'Warning: Failed to make augmented soundscape #{idx+1}/{n_tracks}. Reason for failure: {e}. Moving on...')
                n_failures += 1
                continue
                
    return n_failures, augmented_soundscapes

#==============================================================#===============================================================#

def make_features(responses, soundscapes, maskers,
                  out_fpath = os.path.join('..','features','features.npy'),
                  make_augmented_soundscapes_kwargs = {'mode': 'return',
                                                       'verbose': 0},
                  make_logmel_spectrograms_kwargs = {'n_fft': 4096,
                                                     'hop_length': 2048,
                                                     'n_mels': 64,
                                                     'verbose': 0},
                  verbose = True):
    '''
    Given dataframes representing some responses, soundscapes,
    and maskers, make features of all augmented soundscapes
    for which responses were obtained (as time-frequency
    arrays/.npy files). As many sets of features as there
    are rows in the responses dataframe will be generated.
    
    ======
    Inputs
    ======
    responses : pandas DataFrame
        Contains at least the following columns:
            - 'participant' : str
            - 'fold_r' : int
            - 'soundscape' : str
            - 'masker' : str
            - 'smr' : int
            - 'stimulus_index' : int
        All entries in the column 'soundscape' must exist in the
        identically-named column in soundscapes.
        All entries in the column 'masker' must exist in the
        identically-named column in maskers.
        Pairs of entries in the columns ('participant',
        'stimulus_index') msut be unique. 
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
    out_path : str
        The path to output the features (as an .npy file) to.
    make_augmented_soundscapes_kwargs : dict
        Keyword arguments to pass to make_augmented_soundscapes.
    make_logmel_spectrograms_kwargs : dict
        Keyword arguments to pass to make_logmel_spectrograms.
    verbose : bool
        If True, prints status messages. If False, prints
        nothing.
        
    =======
    Returns
    =======
    features : np.ndarray
        An array of shape (len(responses), t, n_mels, c),
        containing the features (as log-mel spectrograms)
        generated for each augmented soundscape.
    
    ============
    Dependencies
    ============
    pandas (as pd), numpy (as np), make_augmented_soundscapes
    (from araus_utils), make_logmel_spectrograms
    (from araus_utils), os
    '''
    # MAKE OUTPUT DIRECTORY IF IT DOESN'T ALREADY EXIST
    out_dir, out_fname = os.path.split(out_fpath)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    # MAKE FEATURES (LOGMEL SPECTROGRAMS)
    features = []
    for idx in range(len(responses)): # Dataframe's index may be out of order so we don't use it (and assign it to _ instead).
        if verbose: print(f'Making feature #{idx+1}/{len(responses)}', end = '\r')
        _, augmented_soundscape = make_augmented_soundscapes(responses.iloc[[idx],:], soundscapes, maskers,
                                                             **make_augmented_soundscapes_kwargs)
        logmel_spectrograms = make_logmel_spectrograms(input_data = np.squeeze(augmented_soundscape).T,
                                                       **make_logmel_spectrograms_kwargs).transpose([1,0,2])
        features.append(np.expand_dims(logmel_spectrograms,axis=0))
    features = np.concatenate(features)

    # SAVE FEATURES
    if verbose: print(f'Writing features to {out_fpath}...')
    np.save(out_fpath, features.astype(np.float32), allow_pickle = True)
    
    return features

#==============================================================#===============================================================#

def make_labels(responses,
                out_fpath = os.path.join('..','features','labels.npy'),
                verbose = True):
    '''
    Given dataframes representing some responses, make labels
    for all augmented soundscapes for which responses were
    obtained. As many sets of labels as there are rows in the
    responses dataframe will be generated.
    
    ======
    Inputs
    ======
    responses : pandas DataFrame
        Contains at least the following columns:
            - 'pleasant' : float
            - 'eventful' : float
            - 'chaotic' : float
            - 'vibrant' : float
            - 'uneventful' : float
            - 'calm' : float
            - 'annoying' : float
            - 'monotonous' : float
    out_path : str
        The path to output the features (as an .npy file) to.
    verbose : bool
        If True, prints status messages. If False, prints
        nothing.
        
    =======
    Returns
    =======
    labels : np.ndarray
        An array of shape (len(responses),), containing the
        labels (i.e. ISO Pleasantness values) for each
        augmented soundscape.
    
    ============
    Dependencies
    ============
    pandas (as pd), numpy (as np), make_augmented_soundscapes
    (from araus_utils), make_logmel_spectrograms
    (from araus_utils), os
    '''
    # MAKE OUTPUT DIRECTORY IF IT DOESN'T ALREADY EXIST
    out_dir, out_fname = os.path.split(out_fpath)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    # MAKE LABELS (ISO PLEASANTNESS VALUES)
    attributes = ['pleasant', 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying', 'monotonous'] # Define attributes to extract from dataframes
    ISOPl_weights = [1,0,-np.sqrt(2)/2,np.sqrt(2)/2, 0, np.sqrt(2)/2,-1,-np.sqrt(2)/2] # Define weights for each attribute in attributes in computation of ISO Pleasantness
    labels = ((responses[attributes] * ISOPl_weights).sum(axis=1)/(4+np.sqrt(32))).values
    
    # SAVE LABELS
    if verbose: print(f'Writing labels to {out_fpath}...')
    np.save(out_fpath, labels.astype(np.float32), allow_pickle = True)
    
    return labels

#==============================================================#===============================================================#

def make_logmel_spectrograms(input_data = np.sin(np.linspace(0,440,44100)*np.pi), sr = None, n_fft = 1024, hop_length = 512, center = False,
                             n_mels = 64, fmin = 20, fmax = 20000, ref = 1.0, dtype = np.float32, verbose = True,
                             **kwargs):
    '''
    Makes log-mel spectrograms of time-series data using librosa,
    given the data as an array or a filepath to the data.
    
    ======
    Inputs
    ======
    input_data : str or np.ndarray
        If a string, it is the filepath of the input data (that
        will be read by sf.read).
        If it is an np.ndarray, it should be either a (n,)- or
        (c,n)-shaped array if it is single-channel or multi-
        channel respectively, where c is the number of channels in
        the signal and n is the number of samples in each channel.
        It will represent the signal in floating point numbers
        between -1 and 1. This function will convert a (n,)-
        shaped array to a (1,n)-shaped array while running.
        Default is a one-second 440Hz sine tone sampled at 44100Hz.
    sr : int
        The sampling rate of the signal specified by input_data. If        
        it is not specified, then the native sampling rate of
        the file in input_data will be used (if input_data is a
        string) or a default sampling rate of 44100Hz will be used
        (if input_data is an np.ndarray).
    n_fft : int
        The number of samples in each time window for the
        calculation of the STFT of the input signal using
        librosa.core.stft.
    hop_length : int
        The number of overlapping samples in each time window for
        the calculation of the STFT of the input signal using
        librosa.core.stft.
    center : bool
        If True, centers the window at the current time index of
        the signal before performing FFT on it. If False, does not
        center the window, i.e. the current time index IS the 
        first index of the window. This is as per the parameter in
        librosa.core.stft.
    n_mels : int
        The number of mel bands used for calculation of the log-
        frequency power spectrom by librosa.feature.melspectrogram
    fmin : float
        The minimum frequency of the mel-filter bank used to
        calculate the log-frequency power spectrum in
        librosa.feature.melspectrogram.
    fmax : float
        The maximum frequency of the mel-filter bank used to
        calculate the log-frequency power spectrum in
        librosa.feature.melspectrogram.
    ref : float or callable
        The reference value or a function which returns the
        reference value for which dB computation is done in
        librosa.power_to_db.
    dtype : type
        The datatype to typecast the log-mel spectorgram values
        to.
    verbose : bool
        If True, prints status and warning messages. If False,
        prints nothing.
    **kwargs : dict
        Additional keyword arguments to pass to librosa.core.stft.

    =======
    Returns
    =======
    logmel_spectrograms : np.ndarray
        An (n_mels, t, c)-shaped array containing the log-mel
        spectrograms of each channel, where n_mels is the number
        of mel bands specified in the input argument to this
        function, t is the number of time bins in the STFT, and
        c is the number of channels in input_data. Note that t
        depends on the input argument center as follows:
        If center is True, then t = int((n+n_fft//2*2-n_fft)/hop_length)+1.
        If center is False, then t = int((n-n_fft)/hop_length)+1.
        
    ============
    Dependencies
    ============
    librosa, numpy (as np), soundfile
    
    =======
    Example
    =======
    >>> s = make_logmel_spectrograms()
    >>> s.T
    array([[[-12.6684618 ,  19.41913414,  29.54381371, ..., -50.45608521,
             -50.45608521, -50.45608521],
            [-12.48330879,  19.42097473,  29.54356003, ..., -50.45608521,
             -50.45608521, -50.45608521],
            [-12.27776623,  19.42311096,  29.54326248, ..., -50.45608521,
             -50.45608521, -50.45608521],
            ...,
            [-12.74295616,  19.41841507,  29.54391479, ..., -50.45608521,
             -50.45608521, -50.45608521],
            [-12.66043186,  19.41921234,  29.54380417, ..., -50.45608521,
             -50.45608521, -50.45608521],
            [-12.47159004,  19.42109299,  29.54354286, ..., -50.45608521,
             -50.45608521, -50.45608521]]])

    '''
    # GET DATA IN THE FORMAT WE WANT
    if type(input_data) == str: # If the input data entered is a string,...
        input_data, native_sr = sf.read(input_data) # ...then we read the filename specified in the string.
        input_data = input_data.T # Transpose the input data to fit the (c,n)-shape desired.
        sr = native_sr if sr == None else sr
    else: # else we assume it is an array-like object, e.g. np.ndarray
        sr = 44100 if sr == None else sr

    if len(input_data.shape) == 1: # At this point, input_data should be either a (n,)- or (c,n)-shaped array. If it's a (n,)-shaped array,...
        input_data = np.expand_dims(input_data,0) # ...then convert it into a (1,n)-shaped array.
        
    # PREALLOCATE ARRAY FOR LOG-MEL SPECTROGRAM
    c = input_data.shape[0] # Number of channels.
    n = input_data.shape[1] # Number of samples.
    t = int((n+n_fft//2*2-n_fft)/hop_length)+1 if center else int((n-n_fft)/hop_length)+1 # Number of STFT time bins.
    logmel_spectrograms = np.zeros((n_mels, t, c))
    
    if c > n and verbose:
        print('Warning: The input data appears to have more channels than samples. Perhaps you intended to input its transpose?')
    
    # CALCULATE LOG-MEL SPECTROGRAMS BY CHANNEL
    for i in range(c):
        # Firstly, calculate the short-time Fourier transform (STFT) of the signal with librosa.core.stft.
        # We typecast input_data[i] as a Fortran-contiguous array because librosa.core.stft does vectorised operations on it,
        # and numpy array slices are typically not Fortran-contiguous.
        input_stft = librosa.core.stft(y = np.asfortranarray(input_data[i]), n_fft = n_fft, hop_length = hop_length, center = center, **kwargs)

        # Then, calculate the mel magnitude spectrogram of the STFT'd signal with librosa.feature.melspectrogram.
        mel_spectrogram = librosa.feature.melspectrogram(S = np.abs(input_stft)**2, sr = sr, n_mels = n_mels, hop_length = hop_length, n_fft = n_fft, fmin = fmin, fmax = fmax)

        # Convert the power spectrogram into into units of decibels with librosa.power_to_db.
        logmel_spectrogram = librosa.power_to_db(mel_spectrogram, ref = ref)

        # Typecast all elements to desired dtype to ensure all output data types match.
        logmel_spectrogram = logmel_spectrogram.astype(dtype)

        # Assign current channel's log-mel spectrogram to the output array logmel_spectrograms
        logmel_spectrograms[:,:,i] = logmel_spectrogram

    return logmel_spectrograms
    
#==============================================================#===============================================================#

def plot_categorical_participant_info(participants,
                                      cat_keys = ['language_a','language_b','language_c','gender','ethnic','occupation','education_a','education_b','dwelling','citizen','residence_length'],
                                      ordinate_labels = ['Multilingual?','English native?','English best?','Gender','Ethnic group','Occupation','Highest education','Current education','Dwelling type','Citizen?','Residence length'],
                                      legend_labels = [{0:'No',1:'Yes'},
                                                       {-1:'N.A.',0:'No',1:'Yes'},
                                                       {-1:'N.A.',0:'No',1:'Yes'},
                                                       {0:'Male',1:'Female'},
                                                       {0:'Others',1:'Chinese',2:'Malay',3:'Indian'},
                                                       {0:'Others',1:'Student',2:'Employed',3:'Retired',4:'Unemp.'},
                                                       {0:'Others',3:'Sec',4:'ITE',5:'High Sch.',6:'Poly.',7:'Bach. Deg.',8:'Mast. Deg.',9:'PhD'},
                                                       {-1:'N.A.',0:'Others',6:'Poly',7:'Bach. Deg.',8:'Mast. Deg.',9:'PhD'},
                                                       {0:'Others',1:'Pub. Apt.',2:'Dorm',3:'Priv. Pty.',4:'Priv. Apt.'},
                                                       {0:'No',1:'Yes'},
                                                       {0:'< 10 yrs',1:'≥ 10 yrs'}],
                                      height_scaless = [{0:10.0,1:2.2},
                                                        {-1:10.0,0:2.5,1:2.2},
                                                        {-1:10.0,0:2.2,1:3.},
                                                        {0:2.5,1:2.4},
                                                        {0:[np.inf]*7,1:[2.2]*7,2:[-1.5,-5,-6,-2,-4,np.inf,np.inf],3:[5,5,7,np.inf,np.inf,np.inf,np.inf]},
                                                        {0:np.inf,1:2.2,2:3.0},
                                                        {0:5,5:2.5,6:3,7:3},
                                                        {-1:[3]*7,0:[10,4.5,4.5,np.inf,4.5,np.inf,np.inf],7:[2.5]*7},
                                                        {1:[2.4]*7,2:[3.5]*7,3:[-0.5,-5,np.inf,np.inf,-2.5,np.inf,np.inf],4:[3.0]*7},
                                                        {0:3.0,1:2.2},
                                                        {0:3.0,1:2.2}],
                                      bar_orders = [[1,0],
                                                    [1,0,-1],
                                                    [1,0,-1],
                                                    [1,0],
                                                    [0,1,2,3],
                                                    [0,1,2],
                                                    [0,5,6,7],
                                                    [0,-1,7],
                                                    [1,2,3,4],
                                                    [1,0],
                                                    [1,0]], 
                                      grouped_cols = [[0],
                                                      [0],
                                                      [0],
                                                      [0],
                                                      [0],
                                                      [0,3,4], 
                                                      [0,3,4,8,9],
                                                      [0,6,8,9],
                                                      [0],
                                                      [0],
                                                      [0]],
                                      figsize = (8,4),
                                      save_to_file = True,
                                      out_dir = os.path.join('..','figures'),
                                      file_format = 'pdf',
                                      verbose = False):
    '''
    Makes stacked bar plots for categorical data in a given
    dataframe participants (assumed to contain data from
    the participant information questionnaire for the dataset).
    
    ======
    Inputs
    ======
    participants : pandas DataFrame
        A dataframe containing the keys in cat_keys
    cat_keys : list of str
        The keys corresponding to the columns in participants
        that plots will be made for. One plot will be made
        for each key in this list.
    ordinate_labels : list of str
        Ordinate (y-axis) labels for the plots.
    legend_labels : list of dict
        Labels for the bars in each plot, where each dictionary
        has keys corresponding to possible categories and 
        values corresponding to the names of those categories
        for the plot legend.
    height_scaless : list of dict
        Scale factors to control text label heights for bars
        in each plot, where each dictionary has keys
        corresponding to possible categories and values
        corresponding to the scale factor for the bar for
        that category
    bar_orders : list of lists
        Orders of stacked bars (from bottom to top). Category
        labels can be omitted to not plot bars for those
        labels.
    grouped_cols : list of lists
        Category labels to be considered as "Others" in the
        plots.
    figsize : tuple of float
        The size of the plots when displayed (or outputted to
        file).
    save_to_file : bool
        If True, saves the plots to files. If False, only
        calls plt.show() without saving plots to files.
    out_dir : str
        The directory to output the plots to (if save_to_file
        is True).
    file_format : str
        The file format that the plots are saved as (if
        save_to_file is True).
    verbose : bool
        If True, prints raw data and axes coordinates (for
        manual adjustment). If False, prints no additional
        text beyond the plots.
        
    =======
    Returns
    =======
    None
    
    ============
    Dependencies
    ============
    matplotlib.pyplot (as plt), pandas (as pd), numpy (as np), autolabel (from araus_utils), autolabel (from araus_utils), os
    '''
    for cat_key, ordinate_label, legend_label, height_scales, bar_order, grouped_col in zip(cat_keys, ordinate_labels, legend_labels, height_scaless, bar_orders, grouped_cols):
        # PROCESS DATA TO PLOT
        summary_df = participants[participants['fold_p'] != 0].value_counts(['fold_p',cat_key]).reset_index().pivot_table(index='fold_p',columns=cat_key)
        add_df = pd.DataFrame({'fold_p': [6,7],cat_key:[0,0],0:[0,0]}).pivot_table(index='fold_p',columns=cat_key) # Create artificial entries at x = 6 and 7 to stretch the x axis s.t. the legends can be accommodated.
        summary_df = pd.concat((summary_df,add_df)) # Add the artificial entries
        summary_df = summary_df.fillna(0) # Fill NaNs with 0s to prevent errors with sums later
        summary_df[(0,0)] = summary_df[[(0,i) for i in grouped_col]].sum(axis=1) # Summarise the columns (by summing) that are in grouped_cols
        summary_df = summary_df[[(0,i) for i in bar_order]] # Reorder columns (= bar plot order)
        if verbose:
            print(f'summary_df is as follows:')
            print(summary_df)

        # MAKE STACKED BAR PLOT
        row_labels = [key[1] for key in summary_df.keys()] # These are the numerical versions for legend_labels
        ax = summary_df.plot.bar(stacked=True,edgecolor='black',figsize=figsize)
        ax.set_ylim(0,120)
        ax.set_ylabel(f'Frequency ({ordinate_label})')
        ax.set_xlabel('Fold')
        ax.set_xticklabels(labels=[1,2,3,4,5,'',''],rotation=0)
        ax.set_xticks([0,1,2,3,4])
        ax.legend([legend_label[i] for i in bar_order], loc = 'upper right')
        ax.set_axisbelow(True) # Makes grid go behind bars
        plt.grid(visible=True)
        if verbose:
            disp_axes_details(ax)
        
        # ATTEMPT TO LABEL BARS
        try:
            autolabel(ax.containers, ax, height_scales = [height_scales[i] for i in bar_order] if type(height_scales) == dict else height_scales, f_height = lambda height: height*100/120)
        except:
            if verbose: print(f'autolabel failed to label bar percentages for {cat_key}')
            
        # SAVE TO FILE AND/OR DISPLAY
        if save_to_file:
            if (not os.path.exists(out_dir)) and len(out_dir) > 0: os.makedirs(out_dir) # Make the output directory if it doesn't already exist
            out_fname = f'demographic_{cat_key}.{file_format}'
            out_fpath = os.path.join(out_dir, out_fname)
            plt.savefig(out_fpath, bbox_inches='tight')
        plt.show()
        
#==============================================================#===============================================================#

def plot_consistency_metrics(consistency,
                             const_keys = ['first_vs_last','pleasant_vs_annoying','eventful_vs_uneventful','calm_vs_chaotic','vibrant_vs_monotonous','ISOPl_vs_pl','ISOEv_vs_ev'],
                             ordinate_labels = ['MAD bet. 1st & last stimuli',
                                                'MAD bet. "pl" (reversed) & "an"',
                                                'MAD bet. "ev" (reversed) & "ue"',
                                                'MAD bet. "ca" (reversed) & "ch"',
                                                'MAD bet. "vi" (reversed) & "mo"',
                                                'MSE bet. "ISOPl" & "pl"',
                                                'MSE bet. "ISOEv" & "ev"'],
                             figsize = (8,4),
                             save_to_file = True,
                             out_dir = os.path.join('..','figures'),
                             file_format = 'pdf'):
    '''
    Makes violin plots of consistency metrics in a given
    dataframe consistency (assumed to be derived from the
    responses in the dataset).
    
    ======
    Inputs
    ======
    consistency : pandas DataFrame
        A dataframe containing the keys in const_keys 
        (should be derived from reponses).
    const_keys : list of str
        The keys corresponding to the columns in participants
        that plots will be made for. One plot will be made
        for each key in this list.
    ordinate_labels : list of str
        Ordinate (y-axis) labels for the plots.
    figsize : tuple of float
        The size of the plots when displayed (or outputted to
        file).
    save_to_file : bool
        If True, saves the plots to files. If False, only
        calls plt.show() without saving plots to files.
    out_dir : str
        The directory to output the plots to (if save_to_file
        is True).
    file_format : str
        The file format that the plots are saved as (if
        save_to_file is True).
        
    =======
    Returns
    =======
    None
    
    ============
    Dependencies
    ============
    matplotlib.pyplot (as plt), pandas (as pd), seaborn (as sns), os
    '''
    for key, ordinate_label in zip(const_keys, ordinate_labels):
        # MAKE PLOT
        plt.figure(figsize=figsize)
        sns.violinplot(data=[consistency[consistency['fold_p'] == i][key] for i in range(1,6)],
                       saturation = 1,
                       cut = 0) # cut=0 prevents extrapolation outside observed data boundaries
        plt.xlabel('Fold')
        plt.xticks(ticks=range(0,5),labels=range(1,6))
        plt.ylabel(ordinate_label)

        plt.ylim([0,2.25])

        plt.grid(visible=True)
        
        # SAVE TO FILE AND/OR DISPLAY
        if save_to_file:
            if (not os.path.exists(out_dir)) and len(out_dir) > 0: os.makedirs(out_dir) # Make the output directory if it doesn't already exist
            out_fname = f'consistency_{key}.{file_format}'
            out_fpath = os.path.join(out_dir, out_fname)
            plt.savefig(out_fpath,bbox_inches='tight')
        plt.show()

#==============================================================#===============================================================#

def plot_continuous_participant_info(participants,
                                     cont_keys = ['age','annoyance_freq','quality','wnss','pss','who','panas_pos','panas_neg'],
                                     ordinate_labels = ['Age (years)','Annoyance frequency','Sound quality','Noise Sensitivity','Perceived Stress','Well-Being','Positive affect','Negative affect'],
                                     ymins = [0,  0,  0, 10,  0,  0, 10, 10],
                                     ymaxs = [0, 10, 10, 50, 40, 25, 50, 50],
                                     figsize = (8,4),
                                     save_to_file = True,
                                     out_dir = os.path.join('..','figures'),
                                     file_format = 'pdf'):
    '''
    Makes violin plots for numerical data in a given dataframe
    participants (assumed to contain data from the participant
    information questionnaire for the dataset).
    
    ======
    Inputs
    ======
    participants : pandas DataFrame
        A dataframe containing the keys in cont_keys
    cont_keys : list of str
        The keys corresponding to the columns in participants
        that plots will be made for. One plot will be made
        for each key in this list.
    ordinate_labels : list of str
        Ordinate (y-axis) labels for the plots.
    ymins : list of float
        Lower y-axis limits for the plots.
    ymaxs : list of float
        Upper y-axis limits for the plots.
    figsize : tuple of float
        The size of the plots when displayed (or outputted to
        file).
    save_to_file : bool
        If True, saves the plots to files. If False, only
        calls plt.show() without saving plots to files.
    out_dir : str
        The directory to output the plots to (if save_to_file
        is True).
    file_format : str
        The file format that the plots are saved as (if
        save_to_file is True).
        
    =======
    Returns
    =======
    None
    
    ============
    Dependencies
    ============
    matplotlib.pyplot (as plt), pandas (as pd), seaborn (as sns), os
    '''

    for key, ordinate_label, ymin, ymax in zip(cont_keys, ordinate_labels, ymins, ymaxs):
        # MAKE PLOT
        plt.figure(figsize = figsize)
        sns.violinplot(data = [participants[participants['fold_p'] == i][key] for i in range(1,6)],
                       saturation = 1,
                       cut = 0) # cut = 0 prevents extrapolation outside observed data boundaries
        plt.xlabel('Fold')
        plt.xticks(ticks = range(0,5), labels = range(1,6))
        plt.ylabel(ordinate_label)
        if key != 'age':
            plt.axhline(ymin, color='red', linestyle='--')
            plt.axhline(ymax, color='red', linestyle='--')
        plt.grid(visible=True)
        
        # SAVE TO FILE AND/OR DISPLAY
        if save_to_file:
            if (not os.path.exists(out_dir)) and len(out_dir) > 0: os.makedirs(out_dir) # Make the output directory if it doesn't already exist
            out_fname = f'demographic_{key}.{file_format}'
            out_fpath = os.path.join(out_dir, out_fname)
            plt.savefig(out_fpath, bbox_inches='tight')
        plt.show()

#==============================================================#===============================================================#

def plot_times_taken(times,
                     figsize = (8,4),
                     save_to_file = True,
                     out_dir = os.path.join('..','figures'),
                     file_format = 'pdf'):
    '''
    Make violin plot of mean time taken per stimulus in a given
    dataframe consistency (assumed to be derived from the
    responses in the dataset).
    
    ======
    Inputs
    ======
    times : pandas DataFrame
        A dataframe containing the key 'time_taken'. This should
        be derived from the participants and responses
        dataframes.
    figsize : tuple of float
        The size of the plot when displayed (or outputted to
        file).
    save_to_file : bool
        If True, saves the plot to file. If False, only
        calls plt.show() without saving plot to file.
    out_dir : str
        The directory to output the plot to (if save_to_file
        is True).
    file_format : str
        The file format that the plot is saved as (if
        save_to_file is True).
    
    =======
    Returns
    =======
    None
    
    ============
    Dependencies
    ============
    matplotlib.pyplot (as plt), pandas (as pd), seaborn (as sns), os
    '''    
    # MAKE PLOT
    plt.figure(figsize=figsize)
    sns.violinplot(data=[times[times['fold_p'] == i]['time_taken'] for i in range(1,6)],
                   saturation = 1,
                   cut=0) # cut=0 prevents extrapolation outside observed data boundaries
    plt.xlabel('Fold')
    plt.xticks(ticks=range(0,5),labels=range(1,6))
    plt.ylabel('Mean time per stimulus (s)')
    plt.grid(visible=True)
    
    # SAVE TO FILE AND/OR DISPLAY
    if save_to_file:
        if (not os.path.exists(out_dir)) and len(out_dir) > 0: os.makedirs(out_dir) # Make the output directory if it doesn't already exist
        out_fname = f'times_taken.{file_format}'
        out_fpath = os.path.join(out_dir, out_fname)
        plt.savefig(out_fpath,bbox_inches='tight')
    plt.show()
    
#==============================================================#===============================================================#

def SOM(data = None, n1 = 10, n2 = 10, n_SO_Iterations = 3000,
        n_Conv_Iterations = 50000, eta_SO = 1, eta_Conv = 0.1,
        seed_val = None, verbose = False):
    '''
    This function calculates the cluster prototypes for a given
    data set using a self-organising map (SOM) neural network
    (Kohonen, 1990).
    
    ======
    Inputs
    ======
    data : np.ndarray or None
        An array of shape (n_observations, n_features) 
        representing the data matrix, where each row represents
        one sample and each column is a feature of the sample.
        If None, a random array of shape (10000,2) is generated.
    n1 : int
        The number of rows in the 2D lattice of neurons in the
        SOM neural network.
    n2 : int
        The number of columns in the 2D lattice of neurons in
        the SOM neural network
    n_SO_Iterations : int
        The number of iterations in the self-organising phase.
        In the self-organising phase, the learning rate will
        decrease exponentially from eta_SO to eta_Conv, but will
        always stay above eta_Conv. Likewise, the best match
        neuron's neighbourhood will also decrease in this phase.
        It is recommended to set this to an integer in the order
        of 1000.
    n_Conv_Iterations : int
        The number of iterations in the convergence phase.
        In the convergence phase, the learning rate will be set
        to eta_Conv and weight updates will occur only for the
        best match neuron.
        It is recommended to set this to an integer at least
        500*n1*n2.
    eta_SO : float
        The initial learning rate for the self-organising phase.
        It will decrease with increasing iterations in the self-
        organising phase.
        It is recommended to set this to be greater than eta_Conv
        although the function performs no checks for this.
    eta_Conv : float
        The learning rate for the convergence phase.
        It will remain constant for the whole convergence phase.
        It is recommended to set this to be less than eta_SO
        although the function performs no checks for this.
    seed_val : int
        The value of the seed to use for np.random.seed. Set this
        to a constant for reproducible results.
    verbose : bool
        Set to True to print status messages, False to print
        nothing.
    
    =======
    Returns
    =======
    dist_mat : np.ndarray
        An array of shape (n1*n2, n_observations), where the
        (i,j)-th element being the distance between the ith
        cluster centre and the jth data point in the data
        matrix.
    
    weights : np.ndarray
        An array of shape (n1*n2, n_features) containing the
        weights of each of the n1*n2 neurons, i.e. te 
        coordinates of the n1*n2 cluster prototypes.
    
    ============
    Dependencies
    ============
    numpy (as np), sklearn
    '''
    # SET RANDOM SEED
    if seed_val is not None:
        np.random.seed(seed_val)
    
    # GENERATE RANDOM DATA IF DATA IS NOT GIVEN
    if data is None:
        data = np.random.rand(10000,2)

    # GENERATE INITIAL WEIGHT VECTORS
    n_observations, n_features = data.shape
    n_clusters = n1*n2
    weights = np.random.randn(n_clusters, n_features)

    # DEFINE INITIAL VALUES FOR TIME CONSTANTS AND NEIGHBOURHOOD FUNCTION SIZE
    sigma0 = np.sqrt((n1-1)**2 + (n2-1)**2)/2
    tau1 = n_SO_Iterations/np.log(sigma0)

    # GENERATE COORDINATES OF NEURONS IN LATENT SPACE
    coordinates = np.array([[x,y] for x in range(n2) for y in range(n1)])
    network_dist_mat = sklearn.metrics.pairwise_distances(coordinates)

    # SELF-ORGANISING PHASE
    for n in range(n_SO_Iterations):
        if verbose and (n+1)%100 == 0: print(f'Now in self-organising phase iteration #{n+1}',end='\r')

        eta = (eta_SO-eta_Conv) * np.exp(-n/n_SO_Iterations) + eta_Conv # Learning rate for current iteration
        sigma = sigma0 * np.exp(-n/tau1) # Radius of neighbourhood

        # RANDOMLY SELECT TRAINING SAMPLE
        ridx = np.random.randint(n_observations)
        rand_sample = np.expand_dims(data[ridx,:],0) # shape = (1,n_features)

        dist2rand_sample = np.sum((weights-rand_sample)**2,axis=1,keepdims=True)
        winner_index = np.argmin(dist2rand_sample)
        h = np.exp(-np.expand_dims(network_dist_mat[:,winner_index],1)/2/(sigma**2))
        weights += eta*h*(rand_sample-weights)

    if verbose: print()
        
    # CONVERGENCE PHASE
    for n in range(n_Conv_Iterations):
        if verbose and (n+1)%100 == 0: print(f'Now in convergence phase iteration #{n+1}',end='\r')
        
        # RANDOMLY SELECT TRAINING SAMPLE
        ridx = np.random.randint(n_observations)
        rand_sample = np.expand_dims(data[ridx,:],0) # shape = (n_features,)

        dist2rand_sample = np.sum((weights-rand_sample)**2,axis=1,keepdims=True)
        winner_index = np.argmin(dist2rand_sample)
        weights[winner_index,:] += eta_Conv*(np.squeeze(rand_sample) - weights[winner_index,:])
        
    # OBTAIN FINAL DISTANCE MATRIX
    dist_mat = sklearn.metrics.pairwise_distances(weights,data) # This is equivalent to pdist(X,Y) in MATLAB,
                                                                # where the (i,j)-th element of pdist(X,Y) is the distance
                                                                # between the ith point of X and the jth point of Y.

    return dist_mat, weights

#==============================================================#===============================================================#

def split_usotw_track(in_fpath = os.path.join('..','soundscapes_raw','R0001_segment_binaural.wav'),
                      out_dir = os.path.join('..','soundscapes'),
                      overwrite = False,
                      verbose = True):
    '''
    Splits 60-second tracks from the Urban Soundscapes of
    the World database (USotW) into two 30-second chunks and
    saves them to separate files according to ARAUS dataset
    format. Note that the USotW tracks have a sampling
    frequency of 48 kHz but for the ARAUS dataset, we use
    44.1 kHz for compatibility with the masker tracks.
    
    ======
    Inputs
    ======
    in_fpath : str
        The path to the 60-second USotW track. We assume
        that the file name is of the form
        R####_segment_binaural.wav, where #### is a
        (possibly 0-padded) 4-digit number.
    out_dir : str
        The directory to output the split tracks to.
        _44100_1 and _44100_2 will be appended to the file
        names of each half.
    overwrite : bool
        If True, overwrites existing files at output
        filepaths. If False, does not overwrite.
    verbose : bool
        If True, prints status and error messages. If False,
        prints nothing.
        
    =======
    Returns
    =======
    successful : bool
        If True, then processing was successful.
        If False, then one or more errors occurred during
        processing.
        
    ============
    Dependencies
    ============
    librosa, os, numpy (as np), sounfile (as sf)
    '''
    # MAKE OUTPUT DIRECTORY IF IT DOESN'T ALREADY EXIST
    if (not os.path.exists(out_dir)) and len(out_dir) > 0:
        os.makedirs(out_dir)

    # PREPARE OUTPUT FILEPATHS
    out_fname1 = f'{os.path.splitext(os.path.split(in_fpath)[-1])[0]}_44100_1.wav'
    out_fname2 = f'{os.path.splitext(os.path.split(in_fpath)[-1])[0]}_44100_2.wav'
    out_fpath1 = os.path.join(out_dir, out_fname1) # Filepath to store first half
    out_fpath2 = os.path.join(out_dir, out_fname2) # Filepath to store second half
    
    if os.path.exists(out_fpath1) and os.path.exists(out_fpath2) and (not overwrite):
        if verbose: print(f'{out_fpath1} and {out_fpath2} both already exist. Skipping processing...')
        return True
    else:
        if verbose: print(f'Now processing {in_fpath}, outputting to {out_fpath1} and {out_fpath2}...')

    # CHECK IF INPUT FILE EXISTS
    if not os.path.exists(in_fpath):
        if verbose: print(f'Error: {in_fpath} does not exist.')
        return False

    # READ FILE AND RESAMPLE TO 44.1 kHz
    x, sr = librosa.load(in_fpath, sr = 44100, mono = False) 
    if x.shape != (2, 2646000):
        if verbose: print(f'Warning: Audio file has shape {x.shape} when read but expected (2, 2646000)')

    # SPLIT TRACK INTO HALVES
    y1 = x[:,:44100*30].T # Extract first half of track
    y2 = x[:,44100*30:].T # Extract second half of track

    # WRITE HALVES
    if (not os.path.exists(out_fpath1)) or overwrite:
        sf.write(out_fpath1,y1,sr)
    if (not os.path.exists(out_fpath2)) or overwrite:
        sf.write(out_fpath2,y2,sr)

    return True

#==============================================================#===============================================================#

def unzip(zip_fpath, out_dir = '..', checksum = None, delete_zip = False, verbose = True, **kwargs):
    '''
    Unzips file with zipfile with some basic validation.
    
    ======
    Inputs
    ======
    zip_fpath : str
        The filepath of the zip file.
    out_dir : str
        The directory to extract the zip file's contents to.
    checksum : None or str
        If None, peforms no checksum verification of the file
        at zip_fpath.
        If str, verifies if this matches the 64-byte BLAKE2
        hash of the file at zip_fpath (as returned by
        hashlib.blake2b(open(zip_fpath,'rb').read()).hexdigest())
        before unzipping the file. File is not unzipped if
        checksum does not match.
    delete_zip : bool
        If True, deletes zip file at filepath after successful
        extraction of files. If False, does not delete zip
        file after extraction.
    verbose : bool
        If True, prints status and error messages. If False,
        prints nothing.
    **kwargs : dict
        Additional keyword arguments to ZipFile.extractall
    
    =======
    Returns
    =======
    successful : bool
        If True, zip file was extracted successfully.
        If False, one or more errors occurred during extraction.
        
    ============
    Dependencies
    ============
    ZipFile (from zipfile), hashlib, os
    '''
    # CHECK FILE'S EXISTENCE
    if not os.path.exists(zip_fpath):
        if verbose: print(f'Error: {zip_fpath} does not exist.')
        return False
    
    # VERIFY CHECKSUM
    if (checksum is not None) and (hashlib.blake2b(open(zip_fpath,'rb').read()).hexdigest() != checksum):
        if verbose: print(f'Error: checksum of {zip_fpath} does not match expected checksum.')
        return False
    
    # ATTEMPT TO EXTRACT CONTENTS
    if verbose: print(f'Now unzipping {zip_fpath}...')
    try:
        with ZipFile(zip_fpath) as zip_fh:
            zip_fh.extractall(path = out_dir, **kwargs)
    except Exception as e:
        if verbose: print(f'Error: {e}')
        return False
    
    # DELETE ORIGINAL ZIP FILE IF DESIRED
    if delete_zip:
        if verbose: print(f'Now deleting {zip_fpath}...')
        os.remove(zip_fpath)
        
    return True
