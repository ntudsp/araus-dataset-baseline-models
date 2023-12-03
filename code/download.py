import argparse, os, glob, shutil
import pandas as pd
from araus_utils import download_file, unzip, split_usotw_track

#------------------------------ Define argument parser ------------------------------#

parser = argparse.ArgumentParser(description='Downloads and does initial processing of data for ARAUS dataset.')

parser.add_argument('manifest_fpath', metavar = 'PATH_TO_MANIFEST', nargs = '?', default = 'manifest.csv',
                    help='The path to the manifest CSV file (containing the columns "url","out","checksum"). The function will attempt to download all the files at "url" into the paths specified by "out" and verifies that the downloaded files match "checksum". Default: manifest.csv.')
parser.add_argument('-d','--delete-zip',dest='delete_zip', metavar='D', type = int, nargs='?', default = 0, choices = [0,1],
                    help='Whether to delete ZIP archives after successful extraction. Enter 1 to delete, or 0 to not delete. Default: 0.')
parser.add_argument('-m','--max-tries',dest='max_tries', metavar='M', type = int, nargs='?', default = 10,
                    help='The maximum number of tries the script will attempt to download a given file. Default: 10.')
parser.add_argument('-o','--overwrite',dest='overwrite', metavar='O', type = int, nargs='?', default = 1, choices = [0,1,2],
                    help='Controls strictness of overwriting. If 0, does not overwrite existing files at "out" regardless of their checksums. If 1, overwrites existing files at "out" iff their checksum verifications fail. If 2, overwrites existing files at "out" regardless of their checksums. Default: 1.')
parser.add_argument('-s','--soundscape-out-dir',dest='soundscape_out_dir',metavar='S',nargs='?',default=os.path.join("..","soundscapes"),
                    help=f'The directory to save processed (i.e. resampled and split) soundscape files to. Default: {os.path.join("..","soundscapes")}')
parser.add_argument('-v','--verbose',dest='verbose', metavar='V', type = int, nargs='?', default = 1, choices = [0,1],
                    help='Verbosity of output. Enter 1 to print status and error messages, or 0 to print nothing. Default: 1.')
parser.add_argument('-z','--zip-out-dir',dest='zip_out_dir',metavar='Z',nargs='?',default='..',
                    help='The directory to extract zip file contents in the manifest to. Default: ..')

args = parser.parse_args()

#------------------------------ Download files from manifest ------------------------------#

manifest = pd.read_csv(args.manifest_fpath)
for _, row in manifest.iterrows():
    # DOWNLOAD FILE IN MANIFEST
    download_file(row.url, row.out, checksum = row.checksum, max_tries = args.max_tries, overwrite = args.overwrite, verbose = args.verbose)
    
    # UNZIP FILE IF IT IS A ZIP FILE
    if row.out.split('.')[-1] == 'zip':
        unzip(row.out, out_dir = args.zip_out_dir, delete_zip = args.delete_zip, verbose = args.verbose)
    
    # SPLIT TRACK IF IT IS A WAV FILE
    # (We assume all wav files in the manifest are 1-min long @ 48 kHz)
    if row.out.split('.')[-1] == 'wav':
        split_usotw_track(in_fpath = row.out, out_dir = args.soundscape_out_dir, overwrite = max(0,args.overwrite-1), verbose = args.verbose)

#------------------------------ Re-organise and process files for V2 ------------------------------#

maskersv2_src = glob.glob(os.path.join('..','maskersv2','*.wav'))
dst_dir = os.path.join('..','maskers')
for src in maskersv2_src:
    _, fname = os.path.split(src)
    dst = os.path.join(dst_dir,fname)
    # print(src, dst)
    shutil.move(src,dst)
shutil.rmtree(os.path.join('..','maskersv2'))

binaural_src = glob.glob(os.path.join('..','binaural','*.wav'))
dst_dir = os.path.join('..','soundscapes_raw')
for src in binaural_src:
    _, fname = os.path.split(src)
    dst = os.path.join(dst_dir,fname)
    split_usotw_track(src)
    shutil.move(src,dst)
shutil.rmtree(os.path.join('..','binaural'))
