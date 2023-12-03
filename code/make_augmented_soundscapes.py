import argparse, os
import pandas as pd
from araus_utils import make_augmented_soundscapes

#------------------------------ Define argument parser ------------------------------#

parser = argparse.ArgumentParser(description='Makes augmented soundscapes for which responses were collected in the ARAUS dataset.')

parser.add_argument('responses_csv_fpath', metavar = 'PATH_TO_RESPONSES_CSV', nargs = '?', default = os.path.join("..", "data", "responses.csv"),
                    help=f'The path to the responses CSV file (containing at least the columns "participant", "fold_r", "soundscape", "masker", "smr", and "stimulus_index"). Default: {os.path.join("..", "data", "responses.csv")}.')
parser.add_argument('soundscapes_csv_fpath', metavar = 'PATH_TO_SOUNDSCAPES_CSV', nargs = '?', default = os.path.join("..", "data", "soundscapes.csv"),
                    help=f'The path to the soundscapes CSV file (containing at least the columns "soundscape", "gain_s", and "insitu_leq"). Default: {os.path.join("..", "data", "soundscapes.csv")}.')
parser.add_argument('maskers_csv_fpath', metavar = 'PATH_TO_MASKERS_CSV', nargs = '?', default = os.path.join("..","data","maskers.csv"),
                    help=f'The path to the maskers CSV file (containing at least the columns "masker", "gain_##dB", and "leq_at_gain_##dB", where ## are integers from 46 to 83, inclusive). Default: {os.path.join("..","data","maskers.csv")}.')
parser.add_argument('-sd','--soundscape-dir', dest = 'soundscape_dir', metavar = 'SOUNDSCAPE_DIR', nargs = '?', default = os.path.join("..", "soundscapes"),
                   help = f'The directory where the soundscape files (entries in the "soundscapes" columns) are stored. Default: {os.path.join("..", "soundscapes")}.')
parser.add_argument('-md','--masker-dir', dest = 'masker_dir', metavar = 'MASKER_DIR', nargs = '?', default = os.path.join("..", "maskers"),
                   help = f'The directory where the masker files (entries in the "maskers" columns) are stored. Default: {os.path.join("..", "maskers")}.')
parser.add_argument('-od','--out-dir', dest = 'out_dir', metavar = 'OUT_DIR', nargs = '?', default = os.path.join("..","soundscapes_augmented"),
                   help = f'The directory to output the augmented soundscapes. Default: {os.path.join("..","soundscapes_augmented")}.')
parser.add_argument('-of','--out-format', dest = 'out_format', metavar = 'OUT_FORMAT', nargs = '?', default = 'wav',
                   help = 'The audio file format to output the augmented soundscapes in. Default: wav.')
parser.add_argument('-o','--overwrite',dest='overwrite', metavar='O', type = int, nargs='?', default = 0, choices = [0,1],
                    help='If 1, will overwrite existing files when outputting augmented soundscapes with filenames matching existing files. If 0, will not overwrite existing files. Default: 0.')
parser.add_argument('-s','--stop-upon-failure',dest='stop_upon_failure',metavar='S',type=int,nargs='?', default = 0, choices = [0,1],
                   help = 'If 1, any error in making the augmented soundscapes will stop processing. If 0, then processing will continue regardless of errors in making the augmented soundscapes, until attempts to make all augmented soundscapes have been made. Default: 0.')
parser.add_argument('-v','--verbose',dest='verbose', metavar='V', type = int, nargs='?', default = 1, choices = [0,1,2],
                    help='Verbosity of output. If 0, prints nothing. If 1, prints basic status messages. If 2, prints detailed status messages. Default: 1.')
parser.add_argument('-f','--folds', dest='folds',metavar = 'F', type = int, nargs = '*', default = [0,1,2,3,4,5], choices = [-1,0,1,2,3,4,5,6,7],
                   help='Folds to output augmented soundscapes for. Enter multiple values to output multiple folds. Possible values: 0 = ARAUSv1 test set, 1/2/3/4/5 = ARAUSv1 cross-validation set folds, 6/7 = ARAUSv2 test set, -1 = attention/calibration/practice stimuli. Default: [0,1,2,3,4,5].')

args = parser.parse_args()

#------------------------------ Make augmented soundscapes ------------------------------#

folds = list(set(args.folds))
responses =  pd.read_csv(args.responses_csv_fpath)
soundscapes = pd.read_csv(args.soundscapes_csv_fpath)
maskers = pd.read_csv(args.maskers_csv_fpath)

if args.verbose > 0: print('Making augmented soundscapes for the following folds:', folds)
responses_f = responses[responses.fold_r.isin(folds)]
make_augmented_soundscapes(responses_f, soundscapes, maskers,
                           mode              = 'file'                ,
                           soundscape_dir    = args.soundscape_dir   ,
                           masker_dir        = args.masker_dir       ,
                           out_dir           = args.out_dir          ,
                           out_format        = args.out_format       ,
                           overwrite         = args.overwrite        ,
                           stop_upon_failure = args.stop_upon_failure,
                           verbose           = args.verbose          )
