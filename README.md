# araus-dataset-baseline-models

This repository stores code to download the ARAUS dataset and train baseline models for the dataset. For more details on the dataset, please refer to our publication:

> Kenneth Ooi, Zhen-Ting Ong, Karn N. Watcharasupat, Bhan Lam, Jooyoung Hong, Woon-Seng Gan, ARAUS: A large-scale dataset and baseline models of affective responses to augmented urban soundscapes, _Building and Environment_, [Under review].

The ARAUS dataset makes use of urban soundscape recordings from the <a href="https://urban-soundscapes.s3.eu-central-1.wasabisys.com/soundscapes/index.html">Urban Soundscapes of the World (USotW) database</a>. If you use the ARAUS dataset or the USotW recordings in your work, please cite the following publication:

> Bert De Coensel, Kang Sun and Dick Botteldooren. Urban Soundscapes of the World: selection and reproduction of urban acoustic environments with soundscape in mind. In _Proceedings of the 46th International Congress and Exposition on Noise Control Engineering (Inter-noise)_, Hong Kong (2017).

The ARAUS dataset also makes use of masker recordings taken from <a href="https://freesound.org/">Freesound</a> or <a href="https://xeno-canto.org/">Xeno-canto</a>. If you use the ARAUS dataset or the masker recordings in your work, please ensure that you comply with the <a href="#license">license</a> (mainly Creative Commons licenses) and <a href="#citation">citation</a> terms of the individual source files. We have collated these terms as part of the ARAUS dataset for easy reference, but make no representation or guarantee as to their accuracy or currency. In the event of any discrepancy, please refer to the metadata of the original source files hosted on their corresponding databases.

# Getting started

Firstly, clone this repository by manually downloading it from https://github.com/kenowr/araus-dataset-baseline-models, or enter the following line from a terminal (you need to have <a href="https://github.com/git-guides/install-git">git</a> installed first, of course):

    git clone https://github.com/ntudsp/araus-dataset-baseline-models.git

You may then navigate to the downloaded folder with 

    cd araus-dataset-baseline-models

If you are using <a href="https://docs.conda.io/en/latest/">conda</a> as your package manager, you may enter the following line into a terminal to install the required packages into a conda environment (or you may install them manually using the requirements stated in `araus.yml`):

    conda env create -f araus.yml

Activate the conda environment by entering the following line into a terminal:

    conda activate araus

(If you are running the code on a computer with macOS installed, and the above commands fail, try `conda env create -f araus-mac.yml` and `conda activate araus-mac` instead.)

To download the files making up the dataset (which includes the raw audio of the individual maskers and soundscapes, as well as CSV files containing metadata and all data on the subjective responses in the dataset), you may then enter the following into a terminal (this will download ~3 GB of data from the Internet):

    cd ./code
    python download.py manifest.csv

If all files have downloaded successfully, your directory structure should match <a href="#dir_after_download">this</a>.

Then, to run the replication code and baseline models, as reported in our publication, you may enter the following line into a terminal (this opens a Jupyter Notebook in your default browser):

    jupyter lab --notebook-dir .. replication_code.ipynb

Alternatively, if you wish to only generate the augmented soundscapes to which the subjective responses in the ARAUS dataset were collected (e.g. for your own analysis or exploration), you may then enter the following line into a terminal (this will generate ~132 GB of data as WAV files):

    python make_augmented_soundscapes.py

The augmented soundscapes may be generated in <a href="https://xiph.org/flac/">FLAC</a> format instead (~50 GB of data):

    python make_augmented_soundscapes.py -of flac
    
# Directory structure

## This repository
    .
    ├── code                                               # Code used to process the raw data and output the results.
    │   ├── araus_tf.py
    │   ├── araus_utils.py
    │   ├── download.py   
    │   ├── make_augmented_soundscapes.py
    │   ├── manifest.csv
    │   └── replication_code.ipynb
    │
    ├── README.md                                          # This file.
    ├── araus-mac.yml                                      # The Anaconda environment containing required packages to run all the code in ./code (for macOS)
    └── araus.yml                                          # The Anaconda environment containing required packages to run all the code in ./code (for Windows and Ubuntu).

## After running `python download.py manifest.csv` <a name="dir_after_download">
    .
    ├── code                                               # Code used to process the raw data and output the results.
    │   ├── araus_tf.py
    │   ├── araus_utils.py
    │   ├── download.py   
    │   ├── make_augmented_soundscapes.py
    │   ├── manifest.csv
    │   └── replication_code.ipynb
    │
    ├── data                                               # Folder containing all CSV data in the ARAUS dataset
    │   ├── maskers.csv
    │   ├── participants.csv
    │   ├── participants_rejected.csv
    │   ├── participants_rejected_reasons.csv
    │   ├── responses.csv
    │   ├── responses_rejected.csv
    │   └── soundscapes.csv
    │
    ├── figures                                            # Folder containing all reference figures used in the Jupyter Notebook (46 files; all png)
    │   ├── BIRD_abs.png
    │   ├── ...
    │   └── WIND_pc.png
    │
    ├── maskers                                            # Folder containing all maskers used in the ARAUS dataset (293 files; all mono, 44.1kHz, 30 seconds in length).
    │   ├── bird_00001.wav
    │   ├── ...
    │   └── wind_10001.wav
    │
    ├── soundscapes                                        # Folder containing all soundscapes used in the ARAUS dataset (248 files; all binaural, 44.1 kHz, 30 seconds in length).
    │   ├── R0001_segment_binaural_44100_1.wav
    │   ├── ...
    │   └── R1008_segment_binaural_44100.wav
    │ 
    ├── soundscapes_raw                                    # Folder containing soundscapes from the Urban Soundscapes of the World database (121 files; all binaural, 48 kHz, 60 seconds in length).
    │   ├── R0001_segment_binaural.wav
    │   ├── ...
    │   └── R0133_segment_binaural.wav
    │
    ├── .gitignore
    ├── README.md                                          # This file.
    ├── araus-mac.yml                                      # The Anaconda environment containing required packages to run all the code in ./code (for MacOS)
    └── araus.yml                                          # The Anaconda environment containing required packages to run all the code in ./code.

For more details on the contents of each CSV file, please refer to the sections with their filenames as titles.

# Data files

All metadata on the soundscapes and maskers used for this dataset, as well as all subjective perceptual responses collected as part of this dataset, is organised into four CSV files:

- <a href="#maskers">`maskers.csv`</a> : Infomation about the maskers used in the dataset (consisting of 280 non-silent maskers in the five-fold cross-validation set, 7 non-silent maskers in the independent test set, and 6 silent ``maskers'' used when no masker is to be added to an urban soundscape).
- <a href="#soundscapes">`soundscapes.csv`</a> : Information about the soundscapes used in the dataset (consisting of 234 soundscapes from the <a href="https://urban-soundscapes.s3.eu-central-1.wasabisys.com/soundscapes/index.html">Urban Soundscapes of the World database</a> in the five-fold cross-validation set, 6 soundscapes recorded by us in the independent test set, and 7 soundscapes not used in either set (<a href="#fold_s">this section</a> explains why)).
- <a href="#participants">`participants.csv`</a> : Information about the participants (consisting of 600 in the five-fold cross-validation set and 5 in the independent test set) who provided their responses to the stimuli used in this dataset.
- <a href="#responses">`responses.csv`</a> : Information about the individual stimuli and responses (consisting of 25,200 responses in the five-fold cross-validation set, 240 responses in the independent test set, and 1,815 responses to auxiliary stimuli used for practice and data quality checks) used in this dataset.

Further details on the CSV files can be found below the corresponding subheaders in this section.

The CSV files can also be considered as individual database tables in an SQL database with the following fields as keys:

- `maskers.csv` : Primary key = `masker`
- `soundscapes.csv` : Primary key = `soundscape`
- `participants.csv` : Primary key = `participant`
- `responses.csv` : Primary key = (`participant`, `stimulus_index`), foreign keys = `masker`, `soundscape`, `participant`

There are three additional files containing rejected data from the dataset that we include in this repository for transparency and accountability, but that we do not recommend using:

- <a href="#participants">`participants_rejected.csv`</a> : Information about participants (consisting of 37 in the five-fold cross-validation set and 0 in the independent test set) whose responses were rejected from the dataset.
- <a href="#participants_rejected_reasons">`participants_rejected_reasons.csv`</a> : Information about reasons why participants' responses were rejected from the dataset.
- <a href="#responses">`responses_rejected.csv`</a> : Information about the responses that were rejected from this dataset

## `maskers.csv` <a name="maskers">

This CSV file contains information related to the maskers used to generate the stimuli for which the responses in <a href="#responses">`responses.csv`</a> were collected, as well as relevant psychoacoustic parameters of the maskers computed after calibration to an L<sub>A,eq</sub> value of 65 dB.

The maskers were processed from original source files that came from either <a href="https://freesound.org/">Freesound</a> or <a href="https://xeno-canto.org/">Xeno-canto</a>, and hence may differ from the original files found at their respective websites because:

1. Portions of files that were originally longer than 30 seconds were cut to create the corresponding 30-second masker in this repository.
2. Portions of files that were originally shorter than 30 seconds may have been repeated (or have had silence added) to create the corresponding 30-second masker in this repository.
3. Noise cancellation/high-pass filtering may have been performed (using <a href="https://www.audacityteam.org/audacity-2-3-2-released/">Audacity 2.3.2</a>) to the original file to reduce ambient/microphone noise present in the original file.

Hence, we recommend using the maskers downloaded using `./code/download.py` (or manually downloaded from <a href="https://doi.org/10.21979/N9/9OTEVX">https://doi.org/10.21979/N9/9OTEVX</a>) instead of the original source files at Freesound or Xeno-canto for analysis and training of models; information regarding the original source files has been provided only for the purposes of accountability and transparency.

### Fields

- `masker` : unique strings <a name="masker_field">
  - The name of the file containing the masker.
- `fold_m` : integers in {0, 1, 2, 3, 4, 5}
  - The fold index of the masker. The sets of maskers in each fold are pairwise disjoint.
  - Keys:
    - `0` : Test set.
    - `1` : Fold 1 of the 5-fold cross-validation set.
    - `2` : Fold 2 of the 5-fold cross-validation set.
    - `3` : Fold 3 of the 5-fold cross-validation set.
    - `4` : Fold 4 of the 5-fold cross-validation set.
    - `5` : Fold 5 of the 5-fold cross-validation set.
- `class` : strings in {"bird", "construction", "silence", "traffic", "water", "wind"}
  - The class that the masker belongs to.
  - There is only one masker in the "silence" class for each fold and its corresponding audio file is a sequence of all zeros.
- `site` :  strings in {"Freesound", "Xeno-canto", "NIL"}
  - The site where the original source file (that was processed to create the masker in this repository) can be found.
  - "NIL" is used for maskers in the "silence" class.
  - Please see the <a href="https://freesound.org/">Freesound</a> and <a href="https://xeno-canto.org/">Xeno-canto</a> websites for more information on their respective databases.
- `site_index` : non-negative integers
  - The index of the original source file (that was processed to create the masker in this repository) in the website `site`.
  - 0 is used for maskers in the "silence" class.
  - Files originating from Freesound can be accessed at `https://freesound.org/s/{site_index}/` and files originating from Xeno-canto can be accessed at `www.xeno-canto.org/{site_index}`.
- `citation` : strings <a name="citation">
  - The citation for the original source files, as required by the respective sites where the original source files were uploaded to.
  - "NIL" is used for maskers in the "silence" class.
  - **If you use any of these maskers for your work, please include the string in this field in your citations or acknowledgements.**
- `license` : strings <a name="license">
  - The license that the original source files (and hence the corresponding masker in this repository) is licensed under.
  - "NIL" is used for maskers in the "silence" class.
  - We license the maskers provided in this repository under identical licenses as the original source files on Freesound/Xeno-canto. Hence, the licenses are potentially different for different maskers. An exhaustive list of the possible licenses appearing in this field is as follows (click on the links for more information on the respective licenses):
    - <a href="http://creativecommons.org/publicdomain/zero/1.0/">Creative Commons 0</a>
    - <a href="http://creativecommons.org/licenses/by/3.0/">Creative Commons Attribution 3.0</a>
    - <a href="http://creativecommons.org/licenses/by-nc/3.0/">Creative Commons Attribution-NonCommercial 3.0</a>
    - <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivs 4.0</a>
    - <a href="https://creativecommons.org/licenses/by-nc-sa/3.0/">Creative Commons Attribution-NonCommercial-ShareAlike 3.0</a>
    - <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0</a>
    - <a href="https://creativecommons.org/licenses/by-sa/3.0/">Creative Commons Attribution-ShareAlike 3.0</a>
    - <a href="https://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0</a>
    - <a href="http://creativecommons.org/licenses/sampling+/1.0/">Creative Commons Sampling Plus 1.0</a>
- `comments` : strings
  - Comments that we have regarding the masker (if any).
- `gain_##dB` : floating point numbers
  - Gain to apply to achieve an L<sub>A,eq</sub> of `##` decibels when played back over a pair of Beyerdynamic Custom One Pro headphones, powered by a Creative SoundBlaster E5 soundcard (set at volume 40).
  - `##` is replaced with integers between 46 and 83, inclusive.
  - A gain of 1 is used for maskers in the "silence" class.
- `leq_at_gain_##dB` : floating point numbers
  - Actual L<sub>A,eq</sub> measured by a <a href="https://www.grasacoustics.com/products/head-torso-simulators-kemar/product/733-45bb">GRAS 45BB Head and Torso Simulator</a>, when a gain of `gain_##dB` was applied before playback over a pair of Beyerdynamic Custom One Pro headphones, powered by a Creative SoundBlaster E5 soundcard (set at volume 40).
  - `##` is replaced with integers between 46 and 83, inclusive.
  - An L<sub>A,eq</sub> of `##` is used for maskers in the "silence" class.
- `Savg_m` : floating point numbers
  - Mean sharpness (in acum) over time, computed according to DIN 45692 assuming free field conditions. The computation method is as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `Smax_m` : floating point numbers
  - Maximum sharpness (in acum) attained, computed according to DIN 45692 assuming free field conditions.
- `Sargmax_m` : floating point numbers
  - Time (in seconds) when maximum sharpness (in acum) was attained, computed according to DIN 45692 assuming free field conditions.
- `S##_m` : floating point number 
  - `##` percent exceedance level of sharpness (in acum), computed according to DIN 45692 assuming free field conditions. This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `Navg_m` : floating point numbers
  - Mean loudness (in sone) over time, computed according to ISO 532-1 assuming free field conditions. The computation method is as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `Nrmc_m` : floating point numbers
  - Root mean cubed loudness (in sone) over time, computed according to ISO 532-1 assuming free field conditions.
  - This is equal to the L<sub>3</sub> norm of loudness values over time, so e.g. the root mean cube of the values 1 and 2 is 2.08.
- `Nmax_m` : floating point numbers
  - Maximum loudness (in sone) attained, computed according to ISO 532-1 assuming free field conditions.
- `Nargmax_m` : floating point numbers
  - Time (in seconds) when maximum loudness (in sone) was attained, computed according to ISO 532-1 assuming free field conditions.
- `N##_m` : floating point number
  - `##` percent exceedance level of loudness (in sone), computed according to ISO 532-1 assuming free field conditions. This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `Favg_m` : floating point numbers
  - Mean fluctuation strength (in vacil) over time, at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
  - See Chapter 10 of <a href="https://www.semanticscholar.org/paper/Psychoacoustics%3A-Facts-and-Models-Fastl-Zwicker/b1c7886b079ffadfcafdc2b376e1e0ee86592d24">"Psychoacoustics: Facts and Models (3rd ed.)" by Zwicker and Fastl</a> for further information.
- `Fmax_m` : floating point numbers
  - Maximum fluctuation strength (in vacil) attained, at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `Fargmax_m` : floating point numbers
  - Time (in seconds) when maximum fluctuation strength (in vacil) was attained, at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `F##_m` : floating point number
  - `##` percent exceedance level of fluctuation strength (in vacil), at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies. This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `LAavg_m` : floating point numbers
  - Exponentially averaged A-weighted sound pressure level (in decibels) over time, computed with fast averaging (i.e. with time constant of 125 milliseconds). 
  - Filter is designed according to ISO1996-1. This is as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `LAmin_m` : floating point numbers
  - Minimum A-weighted sound pressure level (in decibels) attained.
- `LAargmin_m` : floating point numbers
  - Time (in seconds) when minimum A-weighted sound pressure level (in decibels) was attained.
- `LAmax_m` : floating point numbers
  - Maximum A-weighted sound pressure level (in decibels) attained.
- `LAargmax_m` : floating point numbers
  - Time (in seconds) when maximum A-weighted sound pressure level (in decibels) was attained.
- `LA##_m` : floating point number
  - `##` percent exceedance level of A-weighted sound pressure level (in decibels). This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `LCavg_m` : floating point numbers
  - Exponentially averaged C-weighted sound pressure level (in decibels) over time, computed with fast averaging (i.e. with time constant of 125 milliseconds). 
  - Filter is designed according to ISO1996-1. This is as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `LCmin_m` : floating point numbers
  - Minimum C-weighted sound pressure level (in decibels) attained.
- `LCargmin_m` : floating point numbers
  - Time (in seconds) when minimum C-weighted sound pressure level (in decibels) was attained.
- `LCmax_m` : floating point numbers
  - Maximum C-weighted sound pressure level (in decibels) attained.
- `LCargmax_m` : floating point numbers
  - Time (in seconds) when maximum C-weighted sound pressure level (in decibels) was attained.
- `LC##_m` : floating point number
  - `##` percent exceedance level of C-weighted sound pressure level (in decibels). This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `Ravg_m` : floating point numbers
  - Mean roughness (in asper) over time, at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
  - See Chapter 11 of <a href="https://www.semanticscholar.org/paper/Psychoacoustics%3A-Facts-and-Models-Fastl-Zwicker/b1c7886b079ffadfcafdc2b376e1e0ee86592d24">"Psychoacoustics: Facts and Models (3rd ed.)" by Zwicker and Fastl</a> for further information.
- `Rmax_m` : floating point numbers
  - Maximum roughness (in asper) attained, at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `Rargmax_m` : floating point numbers
  - Time (in seconds) when maximum roughness (in asper) was attained, at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `R##_m` : floating point number
  - `##` percent exceedance level of roughness (in asper), at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies. This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `Tgavg_m` : floating point numbers
  - Mean psychoacoustic tonality (in tonality units) over time of values greater than 0.02, computed with a frequency range of 20 Hz to 20 kHz according to ECMA 74. The computation method is as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `Tavg_m` : floating point numbers
  - Mean psychoacoustic tonality (in tonality units) over time, computed with a frequency range of 20 Hz to 20 kHz according to ECMA 74.
- `Tmax_m` : floating point numbers
  - Maximum psychoacoustic tonality (in tonality units) attained, computed with a frequency range of 20 Hz to 20 kHz according to ECMA 74.
- `Targmax_m` : floating point numbers
  - Time (in seconds) when maximum psychoacoustic tonality (in tonality units) was attained, computed with a frequency range of 20 Hz to 20 kHz according to ECMA 74.
- `T##_m` : floating point number
  - `##` percent exceedance level of psychoacoustic tonality (in tonality units), computed with a frequency range of 20 Hz to 20 kHz according to ECMA 74. This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `M#####_#_m` : floating point numbers
  - Power (in A-weighted decibels relative to 0.00002 Pa) at <a href="https://apmr.matelys.com/Standards/OctaveBands.html">one-third octave band</a> with centre frequency of `#####.#` Hz.
  - `#####_#` is replaced by values in {00005_0, 00006_3, 00008_0, 00010_0, 00012_5, 00016_0, 00020_0, 00025_0, 00031_5, 00040_0, 00050_0, 00063_0, 00080_0, 00100_0, 00125_0, 00160_0, 00200_0, 00250_0, 00315_0, 00400_0, 00500_0, 00630_0, 00800_0, 01000_0, 01250_0, 01600_0, 02000_0, 02500_0, 03150_0, 04000_0, 05000_0, 06300_0, 08000_0, 10000_0, 12500_0, 16000_0, 20000_0}

## `soundscapes.csv` <a name="soundscapes">

This CSV file contains information related to the urban soundscapes used to generate the stimuli for which the responses in <a href="#responses">`responses.csv`</a> were collected, as well as relevant psychoacoustic parameters of the urban soundscapes computed after calibration (i.e. after the soundscape was calibrated to its <a href="#insitu_leq">in-situ L<sub>A,eq</sub></a>).

### Fields

- `soundscape` : unique strings <a name="soundscape_field">
  - The name of the file containing the urban soundscape.
- `fold_s` : integers in {-1, 0, 1, 2, 3, 4, 5}<a name="fold_s">
  - The fold index of the urban soundscape. The sets of urban soundscapes in each fold are pairwise disjoint.
  - Keys:
    - `-1` : Not in any fold. This could be because (a) the stimulus has an in-situ L<sub>A,eq</sub> below 52 dB (to ensure that reproduction levels were significantly above the noise floor where the listening experiments were conducted), (b) the stimulus has an in-situ L<sub>A,eq</sub> above 77 dB (to ensure safe listening levels for our participants), or (c) the stimulus was used as the practice (first), attention (middle), and consistency check (last) stimulus for all participants (to prevent data leakage since it is present to all folds).
    - `0` : Test set.
    - `1` : Fold 1 of the 5-fold cross-validation set.
    - `2` : Fold 2 of the 5-fold cross-validation set.
    - `3` : Fold 3 of the 5-fold cross-validation set.
    - `4` : Fold 4 of the 5-fold cross-validation set.
    - `5` : Fold 5 of the 5-fold cross-validation set.
- `insitu_leq` : floating point numbers <a name="insitu_leq">
  - For soundscapes in fold 0, this value is the in-situ L<sub>A,eq</sub> (in decibels) of the urban soundscape, measured at the same time as the recording was made.
  - For soundscapes in folds -1, 1, 2, 3, 4, and 5, this value was obtained by first calibrating the 1-minute long binaural recordings available in the <a href="https://urban-soundscapes.s3.eu-central-1.wasabisys.com/soundscapes/index.html">Urban Soundscapes of the World database</a> to the L<sub>Aeq,1-min</sub> values provided on the database website (as the file `SotW_LAeq_binaural_average_LR.xlsx` available <a href="https://urban-soundscapes.org/recordings/">here</a>), then measuring the L<sub>Aeq,30-s</sub> of each half of the calibated file (corresponding to the file name in <a href="#soundscape_field">`soundscape`</a>).
- `gain_s` : positive integers
  - Gain to apply to achieve an L<sub>A,eq</sub> of `insitu_leq` decibels when played back over a pair of Beyerdynamic Custom One Pro headphones, powered by a Creative SoundBlaster E5 soundcard (set at volume 40).
- `Savg_s` : floating point numbers
  - Mean sharpness (in acum) over time, computed according to DIN 45692 assuming free field conditions. The computation method is as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `Smax_s` : floating point numbers
  - Maximum sharpness (in acum) attained, computed according to DIN 45692 assuming free field conditions.
- `Sargmax_s` : floating point numbers
  - Time (in seconds) when maximum sharpness (in acum) was attained, computed according to DIN 45692 assuming free field conditions.
- `S##_s` : floating point number 
  - `##` percent exceedance level of sharpness (in acum), computed according to DIN 45692 assuming free field conditions. This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `Navg_s` : floating point numbers
  - Mean loudness (in sone) over time, computed according to ISO 532-1 assuming free field conditions. The computation method is as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `Nrmc_s` : floating point numbers
  - Root mean cubed loudness (in sone) over time, computed according to ISO 532-1 assuming free field conditions.
  - This is equal to the L<sub>3</sub> norm of loudness values over time, so e.g. the root mean cube of the values 1 and 2 is 2.08.
- `Nmax_s` : floating point numbers
  - Maximum loudness (in sone) attained, computed according to ISO 532-1 assuming free field conditions.
- `Nargmax_s` : floating point numbers
  - Time (in seconds) when maximum loudness (in sone) was attained, computed according to ISO 532-1 assuming free field conditions.
- `N##_s` : floating point number
  - `##` percent exceedance level of loudness (in sone), computed according to ISO 532-1 assuming free field conditions. This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `Favg_s` : floating point numbers
  - Mean fluctuation strength (in vacil) over time, at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
  - See Chapter 10 of <a href="https://www.semanticscholar.org/paper/Psychoacoustics%3A-Facts-and-Models-Fastl-Zwicker/b1c7886b079ffadfcafdc2b376e1e0ee86592d24">"Psychoacoustics: Facts and Models (3rd ed.)" by Zwicker and Fastl</a> for further information.
- `Fmax_s` : floating point numbers
  - Maximum fluctuation strength (in vacil) attained, at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `Fargmax_s` : floating point numbers
  - Time (in seconds) when maximum fluctuation strength (in vacil) was attained, at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `F##_s` : floating point number
  - `##` percent exceedance level of fluctuation strength (in vacil), at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies. This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `LAavg_s` : floating point numbers
  - Exponentially averaged A-weighted sound pressure level (in decibels) over time, computed with fast averaging (i.e. with time constant of 125 milliseconds).  
  - Filter is designed according to ISO1996-1. This is as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `LAmin_s` : floating point numbers
  - Minimum A-weighted sound pressure level (in decibels) attained.
- `LAargmin_s` : floating point numbers
  - Time (in seconds) when minimum A-weighted sound pressure level (in decibels) was attained.
- `LAmax_s` : floating point numbers
  - Maximum A-weighted sound pressure level (in decibels) attained.
- `LAargmax_s` : floating point numbers
  - Time (in seconds) when maximum A-weighted sound pressure level (in decibels) was attained.
- `LA##_s` : floating point number
  - `##` percent exceedance level of A-weighted sound pressure level (in decibels). This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `LCavg_s` : floating point numbers
  - Exponentially averaged C-weighted sound pressure level (in decibels) over time, computed with fast averaging (i.e. with time constant of 125 milliseconds). 
  - Filter is designed according to ISO1996-1. This is as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `LCmin_s` : floating point numbers
  - Minimum C-weighted sound pressure level (in decibels) attained.
- `LCargmin_s` : floating point numbers
  - Time (in seconds) when minimum C-weighted sound pressure level (in decibels) was attained.
- `LCmax_s` : floating point numbers
  - Maximum C-weighted sound pressure level (in decibels) attained.
- `LCargmax_s` : floating point numbers
  - Time (in seconds) when maximum C-weighted sound pressure level (in decibels) was attained.
- `LC##_s` : floating point number
  - `##` percent exceedance level of C-weighted sound pressure level (in decibels). This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `Ravg_s` : floating point numbers
  - Mean roughness (in asper) over time, at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
  - See Chapter 11 of <a href="https://www.semanticscholar.org/paper/Psychoacoustics%3A-Facts-and-Models-Fastl-Zwicker/b1c7886b079ffadfcafdc2b376e1e0ee86592d24">"Psychoacoustics: Facts and Models (3rd ed.)" by Zwicker and Fastl</a> for further information.
- `Rmax_s` : floating point numbers
  - Maximum roughness (in asper) attained, at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `Rargmax_s` : floating point numbers
  - Time (in seconds) when maximum roughness (in asper) was attained, at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `R##_s` : floating point number
  - `##` percent exceedance level of roughness (in asper), at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies. This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `Tgavg_s` : floating point numbers
  - Mean psychoacoustic tonality (in tonality units) over time of values greater than 0.02, computed with a frequency range of 20 Hz to 20 kHz according to ECMA 74. The computation method is as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `Tavg_s` : floating point numbers
  - Mean psychoacoustic tonality (in tonality units) over time, computed with a frequency range of 20 Hz to 20 kHz according to ECMA 74.
- `Tmax_s` : floating point numbers
  - Maximum psychoacoustic tonality (in tonality units) attained, computed with a frequency range of 20 Hz to 20 kHz according to ECMA 74.
- `Targmax_s` : floating point numbers
  - Time (in seconds) when maximum psychoacoustic tonality (in tonality units) was attained, computed with a frequency range of 20 Hz to 20 kHz according to ECMA 74.
- `T##_s` : floating point number
  - `##` percent exceedance level of psychoacoustic tonality (in tonality units), computed with a frequency range of 20 Hz to 20 kHz according to ECMA 74. This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `M#####_#_s` : floating point numbers
  - Power (in A-weighted decibels relative to 0.00002 Pa) at <a href="https://apmr.matelys.com/Standards/OctaveBands.html">one-third octave band</a> with centre frequency of `#####.#` Hz.
  - `#####_#` is replaced by values in {00005_0, 00006_3, 00008_0, 00010_0, 00012_5, 00016_0, 00020_0, 00025_0, 00031_5, 00040_0, 00050_0, 00063_0, 00080_0, 00100_0, 00125_0, 00160_0, 00200_0, 00250_0, 00315_0, 00400_0, 00500_0, 00630_0, 00800_0, 01000_0, 01250_0, 01600_0, 02000_0, 02500_0, 03150_0, 04000_0, 05000_0, 06300_0, 08000_0, 10000_0, 12500_0, 16000_0, 20000_0}

## `participants.csv` and `participants_rejected.csv` <a name="participants">

These CSV files contain all the information provided by the participants of the study in the listener context questionnaire. The file `participants.csv` contains the information provided by the participants in the 5-fold cross-validation set and independent test set, whereas the file `participants_rejected.csv` contains the information provided by participants whose responses were rejected from the dataset altogether (the reasons why they were rejected can be found in <a href="#participants_rejected_reasons">`participants_rejected_reasons.csv`</a>).

When participants' responses were rejected, a different participant was assigned the same ID as the participant whose responses were rejected and presented with the same set of stimuli as the participant whose responses were rejected. Hence, we recommend using only the data in `participants.csv` for analysis and training of models; the data in `participants_rejected.csv` is provided only for the purposes of accountability and transparency.

### Fields

- `participant` : unique strings of length 5 (`participants.csv`) or unique strings (`participants_rejected.csv`) <a name="participant_field">
  - The ID of the participant who provided the current row of information about themselves. Each ID corresponds to a unique participant.
- `fold_p` : integers in {0, 1, 2, 3, 4, 5}
  - The fold index of the participant. The sets of participants in each fold are pairwise disjoint.
  - Keys:
    - `0` : Test set.
    - `1` : Fold 1 of the 5-fold cross-validation set.
    - `2` : Fold 2 of the 5-fold cross-validation set.
    - `3` : Fold 3 of the 5-fold cross-validation set.
    - `4` : Fold 4 of the 5-fold cross-validation set.
    - `5` : Fold 5 of the 5-fold cross-validation set.
- `language_a` : integers in {0, 1}
  - Response to the question "Do you speak fluently in any languages/dialects other than English?"
  - Keys:
    - `0` : No
    - `1` : Yes
- `language_b` : integers in {-1, 0, 1}
  - Response to the question "Is English your first or native language?"
  - Participants were only prompted to respond to this question if their response to `language_a` was "Yes". If the response to `language_a` was "No", then we set their response to this question to "Not applicable" by default.
  - Keys:
    - `-1` : Not applicable
    - `0` : No
    - `1` : Yes
- `language_c` : integers in {-1, 0, 1}
  - Response to the question "Among the languages/dialects you speak, would you consider yourself to be most fluent in English?"
  - Participants were only prompted to respond to this question if their response to `language_a` was "Yes". If the response to `language_a` was "No", then we set their response to this question to "Not applicable" by default.
  - Keys:
    - `-1` : Not applicable
    - `0` : No
    - `1` : Yes
- `age` : positive integers
  - Response to the question "What is your age?"
  - The responses we received ranged from 18 to 75 (inclusive).
- `gender` : integers in {0, 1}
  - Response to the question "What is your gender?"
  - Keys:
    - `0` : Male
    - `1` : Female
  - **In the questionnaire, there was a field for participants to specify genders other than "Male" and "Female". A total of 2 participants specified a gender other than "Male" or "Female". Due to the risk of identification, we have randomly replaced their responses to this question with either "Male" or "Female" (50% chance each). Each response was replaced independently.
- `ethnic` : integers in {0, 1, 2, 3}
  - Response to the question "What is your ethnic group?"
  - Keys:
    - `0` : Others
    - `1` : Chinese
    - `2` : Malay
    - `3` : Indian
- `occupation` : integers in {0, 1, 2, 3, 4}
  - Response to the question "What is your occupational status?"
  - Keys:
    - `0` : Others
    - `1` : Student
    - `2` : Employed
    - `3` : Retired
    - `4` : Unemployed
- `education_a` : integers in {0, 1, 2, ..., 8, 9}
  - Response to the question "What is the highest level of education you have _completed_?"
  - Keys:
    - `0` : Others
    - `1` : No qualification
    - `2` : Primary (PSLE), elementary school or equivalent
    - `3` : Secondary (GCE 'N' & 'O' level), middle school or equivalent
    - `4` : Institute of Technical Education or equivalent
    - `5` : Junior College ('A' level), high school or equivalent
    - `6` : Polytechnic and Arts Institution (Diploma level) or equivalent
    - `7` : University (Bachelor's Degree) or equivalent
    - `8` : University (Master's Degree) or equivalent
    - `9` : University (PhD)
- `education_b` : integers in {-1, 0, 2, 3, 4, 5, 6, 7, 8, 9}
  - Response to the question "What is the level of education you are _currently_ undergoing?"
  - Participants were only prompted to respond to this question if their response to `occupation` was "Student". If the response to `occupation` was not "Student", then we set their response to this question to "Not applicable" by default.
  - Keys:
    - `-1` : Not applicable
    - `0` : Others
    - `2` : Primary (PSLE), elementary school or equivalent
    - `3` : Secondary (GCE 'N' & 'O' level), middle school or equivalent
    - `4` : Institute of Technical Education or equivalent
    - `5` : Junior College ('A' level), high school or equivalent
    - `6` : Polytechnic and Arts Institution (Diploma level) or equivalent
    - `7` : University (Bachelor's Degree) or equivalent
    - `8` : University (Master's Degree) or equivalent
    - `9` : University (PhD)
- `dwelling` : integers in {0, 1, 2, 3, 4}
  - Response to "What dwelling type is your current _main_ residence in Singapore?"
  - Keys:
    - `0` : Others
    - `1` : Housing Development Board (HDB) flat or other public apartment
    - `2` : Hall of Residence or other student dormitory
    - `3` : Landed property
    - `4` : Condominium or other private apartment
- `citizen` : integers in {0, 1}
  - Response to "Are you a Singapore citizen?"
  - Keys:
    - `0` : No
    - `1` : Yes
- `residence_length` : integers in {0, 1}
  - Response to "Have you resided in Singapore for more than 10 years?"
  - Keys:
    - `0` : No
    - `1` : Yes
- `annoyance_freq` : integers in {0, 1, 2, ..., 9, 10}
  - Response to "How much has indoor/outdoor noise bothered, disturbed, or annoyed you over the past 12 months?" on an 11-point scale (0 = Not at all, 10 = Extremely)
- `quality` : integers in {0, 1, 2, ..., 9, 10}
  - Response to "How would you describe your satisfaction of the overall quality of the acoustic environment in Singapore?" on an 11-point scale (0 = Extremely dissatisfied, 10 = Extremely satisfied)
- `wnss` : integers in {10, 11, 12, ..., 49, 50}
  - Score on the truncated (10-item) _Weinstein Noise Sensitivity Scale_.
  - See: <a href="http://dx.doi.org/10.1037/0021-9010.63.4.458">Weinstein, N. D. (1978). Individual differences in reactions to noise: A longitudinal study in a college dormitory. _Journal of Applied Psychology_, 63, 458–466. doi:10.1037/0021‐9010.63.4.458</a>
  - Cronbach's alpha for the responses we obtained was computed to be 0.835, with the 95% confidence interval being [0.814, 0.854].
- `pss` : integers in {0, 1, 2, ..., 39, 40}
  - Score on the truncated (10-item) _Perceived Stress Scale_ developed by Cohen et al.
  - The time scale used was 1 month (so all items began with "In the last month, how often have you...")
  - For more information on the full (14-item) scale, see: <a href="https://www.cmu.edu/dietrich/psychology/stress-immunity-disease-lab/publications/scalesmeasurements/pdfs/globalmeas83.pdf">Cohen, S., Kamarck, T., and Mermelstein, R. (1983). A global measure of perceived stress. _Journal of Health and Social Behavior_, 24, 386-396.</a>
  - For more information on the truncated (10-item) scale, see: <a href="https://www.cmu.edu/dietrich/psychology/stress-immunity-disease-lab/scales/pdf/cohen,-s.--williamson,-g.-1988.pdf">Cohen, S. and Williamson, G. Perceived Stress in a Probability Sample of the United States. Spacapan, S. and Oskamp, S. (Eds.) _The Social Psychology of Health_. Newbury Park, CA: Sage, 1988.</a>
  - Cronbach's alpha for the responses was computed to be 0.875, with the 95% confidence interval being [0.860, 0.889].
- `who` : integers in {0, 1, 2, ..., 24, 25}
  - Score on the _WHO-5 Well-Being Index_.
  - The time scale used was 2 weeks (so all items began with "Over the last two weeks, ...")
  - See: <a href="https://www.psykiatri-regionh.dk/who-5/Documents/WHO-5 questionaire - English.pdf">World Health Organization, "WHO-5 Well-Being Index," 1998. [Online]. Available: `https://www.psykiatri-regionh.dk/who-5/Documents/WHO-5 questionaire - English.pdf`</a>.
  - Cronbach's alpha for the responses was computed to be 0.854, with the 95% confidence interval being [0.834, 0.871].
- `panas_pos` : integers in {10, 11, 12, ..., 49, 50}
  - Score for *Positive Affect* on the _Positive and Negative Affect Schedule_ developed by Watson et al.
  - The time scale used was 2 weeks (so all items began with "In the last two weeks, to what extent have you felt...")
  - See: <a href="https://www.semanticscholar.org/paper/Development-and-validation-of-brief-measures-of-and-Watson-Clark/f82f152244b1cb861db0f290d55302011aee28dc">Watson, D., Clark, L. A., and Tellegen, A. (1988). Development and Validation of Brief Measures of Positive and Negative Affect: The PANAS Scales. _Journal of Personality and Social Psychology_, 54(6), 1063-1070.</a>
  - Cronbach's alpha for the responses was computed to be 0.886, with the 95% confidence interval being [0.872, 0.899].
- `panas_neg` : integers in {10, 11, 12, ..., 49, 50}
  - Score for *Negative Affect* on the _Positive and Negative Affect Schedule_ developed by Watson et al.
  - The time scale used was 2 weeks (so all items began with "In the last two weeks, to what extent have you felt...")
  - See: <a href="https://www.semanticscholar.org/paper/Development-and-validation-of-brief-measures-of-and-Watson-Clark/f82f152244b1cb861db0f290d55302011aee28dc">Watson, D., Clark, L. A., and Tellegen, A. (1988). Development and Validation of Brief Measures of Positive and Negative Affect: The PANAS Scales. _Journal of Personality and Social Psychology_, 54(6), 1063-1070.</a>
  - Cronbach's alpha for the responses was computed to be 0.891, with the 95% confidence interval being [0.877, 0.903].

## `participants_rejected_reasons.csv` <a name="participants_rejected_reasons">

This CSV file contains the reasons for the rejection of responses provided by participants in <a href="#participants">`participants_rejected.csv`</a>.

### Fields
    
- `participant` : unique strings
  - The ID of the participant who was rejected. Each ID corresponds to a unique participant.
- `rejection_reason` : strings
  - The reason why this participant's responses (in <a href="#responses">`responses_rejected.csv`</a>) were rejected.

## `responses.csv` and `responses_rejected.csv` <a name="responses">

These CSV files contain the responses to the unique stimuli (= augmented soundscapes) provided by all study participants, as well as relevant psychoacoustic parameters of the stimuli computed after calibration (i.e. after:

1. the soundscape was calibrated to its <a href="#insitu_leq">in-situ L<sub>A,eq</sub></a>,
2. the masker was calibrated at the specified SMR with respect to the calibrated soundscape, and
3. the two tracks were digitally added together to make the stimulus).

Each stimulus was 30 seconds in length, and made by adding a 30-second recording of an urban soundscape (from the <a href="https://urban-soundscapes.org/">Urban Soundscapes of the World database </a> to a 30-second masker track at various soundscape-to-masker ratios (SMR).

The file `responses.csv` contains the responses provided by the participants in the 5-fold cross-validation set and independent test set, whereas the file `responses_rejected.csv` contains the responses rejected from the dataset altogether (the reasons why they were rejected can be found in <a href="#participants_rejected_reasons">`participants_rejected_reasons.csv`</a>).

When responses were rejected, a different participant was assigned the same ID as the participant whose responses were rejected and presented with the same set of stimuli as the participant whose responses were rejected. Hence, we recommend using only the data in `responses.csv` for analysis and training of models; the data in `responses_rejected.csv` is provided only for the purposes of accountability and transparency.

### Fields

- <a href="#participant_field">`participant`</a> : unique strings of length 5 (`responses.csv`) or unique strings (`responses_rejected.csv`)
  - The ID of the participant who provided the current row of responses. Each ID corresponds to a unique participant.
- `fold_r` : integers in {0, 1, 2, 3, 4, 5}
  - The fold index of the response. The sets of responses in each fold are pairwise disjoint.
  - This is identical to the fold indices associated with `soundscape` (i.e. `fold_s`). When not -1, this is also identical to the fold indices associated with `participant` and `masker` (i.e. `fold_p` and `fold_m`).
  - Keys:
    - `-1` : Present as common stimulus in test set and all folds of 5-fold cross-validation set. Used as practice (first), attention (middle), and consistency check (last) stimulus for all participants.
    - `0` : Test set.
    - `1` : Fold 1 of the 5-fold cross-validation set.
    - `2` : Fold 2 of the 5-fold cross-validation set.
    - `3` : Fold 3 of the 5-fold cross-validation set.
    - `4` : Fold 4 of the 5-fold cross-validation set.
    - `5` : Fold 5 of the 5-fold cross-validation set.
- <a href="#soundscape_field">`soundscape`</a> : unique strings
  - The name of the file containing the urban soundscape that the masker in `masker` was added to.
- <a href="#masker_field">`masker`</a> : unique strings
  - The name of the file containing the masker that was added to the urban soundscape in `soundscape`.
- `smr` : integers in {-6, -3, 0, 3, 6}
  - The soundscape-to-masker ratio that `masker` was mixed with `soundscape`.
  - E.g. if the in-situ L<sub>A,eq</sub> of `soundscape` was measured to be 65 dB, and `smr` is -3, then `masker` would be calibrated to 68 dB before being added to `soundscape` to make the stimulus.
- `stimulus_index` <a name="stimulus_index"> : integers in {1, 2, 3, ..., 50, 51}
  - The index of the stimulus for the current participant.
  - Participants were presented with stimuli in ascending order of `stimulus_index`.
  - Each participant in the 5-fold cross-validation set (`fold_r` in {1, 2, 3, 4, 5}) experienced 45 stimuli, and each participant in the test set (`fold_r` equal to 0) experienced 51 stimuli.
  - The first stimulus presented to every participant, regardless of fold, was identical (the first 30 seconds of recording R0091 from the <a href="https://urban-soundscapes.s3.eu-central-1.wasabisys.com/soundscapes/index.html">Urban Soundscapes of the World database</a>). It served as a practice stimulus for participants to familiarise themselves with the questionnaire interface.
  - The last stimulus presented to every participant, regardless of fold, was identical to the first stimulus. Responses to the first and last stimulus for a given participant may serve as a consistency check for individual participants' responses over the duration of the study.
  - For every participant, one of the stimuli with `stimulus_index` in {15, 16, 17, ..., 24, 25} was designated as an attention stimulus, which was identical to the first and last stimulus presented. HOWEVER, instructions to choose the third option for all questions were overlaid on the video for the attention stimulus. Participants were not allowed to submit their answers until they had selected the third option for all questions related to the attention stimulus.
  - We recommend removing rows corresponding to the first, last, and attention stimulus before using the data to train any model or perform any analysis, since those stimuli are identical for all participants regardless of fold.
- `time_taken` : floating point numbers greater than or equal to 30
  - The time between the initial onset of the current stimulus and the participant submitting their responses.
  - Each 30-second stimuli was continuously repeated until the participant submitted their responses for that stimulus, with 5 seconds of silence between each repetition.
  - Participants were not allowed to pause or stop the playback of the stimuli by themselves. Hence, it is guaranteed that the stimulus was played at the time intervals [0, 30], [35, 65], [70, 100], [105, 135], etc., and there was silence at the time intervals [30, 35], [65, 70], [100, 105], [135, 140] etc.
  - Participants were not allowed to submit their responses before the end of the initial playback of each 30-second stimulus. This allowed them to experience the stimuli in entirety before providing their responses.
- `is_attention` <a name="is_attention"> : integers in {0, 1}
  - Whether the present stimulus is an attention stimulus. See the details for <a href="#stimulus_index">`stimulus_index`</a> for more information on the attention stimulus.
  - If the present stimulus is an attention stimulus, then all responses for `pleasant`, `eventful`, `chaotic`, ..., `monotonous`, `appropriate` will all be 3.
  - Keys:
    - `0` : Present stimulus is not an attention stimulus.
    - `1` : Present stimulus is an attention stimulus.
- `pleasant` : integers in {1, 2, 3, 4, 5}
  - Response to the question "To what extent do you agree or disagree that the present surrounding sound environment is pleasant?" on a 5-point scale (1 = Strongly disagree, 5 = Strongly agree)
- `eventful` : integers in {1, 2, 3, 4, 5}
  - Response to the question "To what extent do you agree or disagree that the present surrounding sound environment is pleasant?" on a 5-point scale (1 = Strongly disagree, 5 = Strongly agree)
- `chaotic` : integers in {1, 2, 3, 4, 5}
  - Response to the question "To what extent do you agree or disagree that the present surrounding sound environment is pleasant?" on a 5-point scale (1 = Strongly disagree, 5 = Strongly agree)
- `vibrant` : integers in {1, 2, 3, 4, 5}
  - Response to the question "To what extent do you agree or disagree that the present surrounding sound environment is pleasant?" on a 5-point scale (1 = Strongly disagree, 5 = Strongly agree)
- `uneventful` : integers in {1, 2, 3, 4, 5}
  - Response to the question "To what extent do you agree or disagree that the present surrounding sound environment is pleasant?" on a 5-point scale (1 = Strongly disagree, 5 = Strongly agree)
- `calm` : integers in {1, 2, 3, 4, 5}
  - Response to the question "To what extent do you agree or disagree that the present surrounding sound environment is pleasant?" on a 5-point scale (1 = Strongly disagree, 5 = Strongly agree)
- `annoying` : integers in {1, 2, 3, 4, 5}
  - Response to the question "To what extent do you agree or disagree that the present surrounding sound environment is pleasant?" on a 5-point scale (1 = Strongly disagree, 5 = Strongly agree)
- `monotonous` : integers in {1, 2, 3, 4, 5}
  - Response to the question "To what extent do you agree or disagree that the present surrounding sound environment is pleasant?" on a 5-point scale (1 = Strongly disagree, 5 = Strongly agree)
- `appropriate` : integers in {1, 2, 3, 4, 5}
  - Response to the question "Overall, to what extent is the present surrounding sound environment appropriate to the present place?" on a 5-point scale (1 = Not at all, 5 = Perfectly)
- `Savg_r` : floating point numbers
  - Mean sharpness (in acum) over time, computed according to DIN 45692 assuming free field conditions. The computation method is as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `Smax_r` : floating point numbers
  - Maximum sharpness (in acum) attained, computed according to DIN 45692 assuming free field conditions.
- `Sargmax_r` : floating point numbers
  - Time (in seconds) when maximum sharpness (in acum) was attained, computed according to DIN 45692 assuming free field conditions.
- `S##_r` : floating point number 
  - `##` percent exceedance level of sharpness (in acum), computed according to DIN 45692 assuming free field conditions. This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `Navg_r` : floating point numbers
  - Mean loudness (in sone) over time, computed according to ISO 532-1 assuming free field conditions. The computation method is as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `Nrmc_r` : floating point numbers
  - Root mean cubed loudness (in sone) over time, computed according to ISO 532-1 assuming free field conditions.
  - This is equal to the L<sub>3</sub> norm of loudness values over time, so e.g. the root mean cube of the values 1 and 2 is 2.08.
- `Nmax_r` : floating point numbers
  - Maximum loudness (in sone) attained, computed according to ISO 532-1 assuming free field conditions.
- `Nargmax_r` : floating point numbers
  - Time (in seconds) when maximum loudness (in sone) was attained, computed according to ISO 532-1 assuming free field conditions.
- `N##_r` : floating point number
  - `##` percent exceedance level of loudness (in sone), computed according to ISO 532-1 assuming free field conditions. This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `Favg_r` : floating point numbers
  - Mean fluctuation strength (in vacil) over time, at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
  - See Chapter 10 of <a href="https://www.semanticscholar.org/paper/Psychoacoustics%3A-Facts-and-Models-Fastl-Zwicker/b1c7886b079ffadfcafdc2b376e1e0ee86592d24">"Psychoacoustics: Facts and Models (3rd ed.)" by Zwicker and Fastl</a> for further information.
- `Fmax_r` : floating point numbers
  - Maximum fluctuation strength (in vacil) attained, at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `Fargmax_r` : floating point numbers
  - Time (in seconds) when maximum fluctuation strength (in vacil) was attained, at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `F##_r` : floating point number
  - `##` percent exceedance level of fluctuation strength (in vacil), at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies. This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `LAavg_r` : floating point numbers
  - Exponentially averaged A-weighted sound pressure level (in decibels) over time, computed with fast averaging (i.e. with time constant of 125 milliseconds). 
  - Filter is designed according to ISO1996-1. This is as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `LAmin_r` : floating point numbers
  - Minimum A-weighted sound pressure level (in decibels) attained.
- `LAargmin_r` : floating point numbers
  - Time (in seconds) when minimum A-weighted sound pressure level (in decibels) was attained.
- `LAmax_r` : floating point numbers
  - Maximum A-weighted sound pressure level (in decibels) attained.
- `LAargmax_r` : floating point numbers
  - Time (in seconds) when maximum A-weighted sound pressure level (in decibels) was attained.
- `LA##_r` : floating point number
  - `##` percent exceedance level of A-weighted sound pressure level (in decibels). This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `LCavg_r` : floating point numbers
  - Exponentially averaged C-weighted sound pressure level (in decibels) over time, computed with fast averaging (i.e. with time constant of 125 milliseconds). 
  - Filter is designed according to ISO1996-1. This is as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `LCmin_r` : floating point numbers
  - Minimum C-weighted sound pressure level (in decibels) attained.
- `LCargmin_r` : floating point numbers
  - Time (in seconds) when minimum C-weighted sound pressure level (in decibels) was attained.
- `LCmax_r` : floating point numbers
  - Maximum C-weighted sound pressure level (in decibels) attained.
- `LCargmax_r` : floating point numbers
  - Time (in seconds) when maximum C-weighted sound pressure level (in decibels) was attained.
- `LC##_r` : floating point number
  - `##` percent exceedance level of C-weighted sound pressure level (in decibels). This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `Ravg_r` : floating point numbers
  - Mean roughness (in asper) over time, at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
  - See Chapter 11 of <a href="https://www.semanticscholar.org/paper/Psychoacoustics%3A-Facts-and-Models-Fastl-Zwicker/b1c7886b079ffadfcafdc2b376e1e0ee86592d24">"Psychoacoustics: Facts and Models (3rd ed.)" by Zwicker and Fastl</a> for further information.
- `Rmax_r` : floating point numbers
  - Maximum roughness (in asper) attained, at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `Rargmax_r` : floating point numbers
  - Time (in seconds) when maximum roughness (in asper) was attained, at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `R##_r` : floating point number
  - `##` percent exceedance level of roughness (in asper), at 1/1 Bark resolution, computed as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies. This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `Tgavg_r` : floating point numbers
  - Mean psychoacoustic tonality (in tonality units) over time of values greater than 0.02, computed with a frequency range of 20 Hz to 20 kHz according to ECMA 74. The computation method is as recommended in Table D.1 of ISO 12913-3:2019 for soundscape studies.
- `Tavg_r` : floating point numbers
  - Mean psychoacoustic tonality (in tonality units) over time, computed with a frequency range of 20 Hz to 20 kHz according to ECMA 74.
- `Tmax_r` : floating point numbers
  - Maximum psychoacoustic tonality (in tonality units) attained, computed with a frequency range of 20 Hz to 20 kHz according to ECMA 74.
- `Targmax_r` : floating point numbers
  - Time (in seconds) when maximum psychoacoustic tonality (in tonality units) was attained, computed with a frequency range of 20 Hz to 20 kHz according to ECMA 74.
- `T##_r` : floating point number
  - `##` percent exceedance level of psychoacoustic tonality (in tonality units), computed with a frequency range of 20 Hz to 20 kHz according to ECMA 74. This is the value exceeded `##` percent of the time.
  - `##` is replaced by integers in {05, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
- `M#####_#_r` : floating point numbers
  - Power (in A-weighted decibels relative to 0.00002 Pa) at <a href="https://apmr.matelys.com/Standards/OctaveBands.html">one-third octave band</a> with centre frequency of `#####.#` Hz.
  - `#####_#` is replaced by values in {00005_0, 00006_3, 00008_0, 00010_0, 00012_5, 00016_0, 00020_0, 00025_0, 00031_5, 00040_0, 00050_0, 00063_0, 00080_0, 00100_0, 00125_0, 00160_0, 00200_0, 00250_0, 00315_0, 00400_0, 00500_0, 00630_0, 00800_0, 01000_0, 01250_0, 01600_0, 02000_0, 02500_0, 03150_0, 04000_0, 05000_0, 06300_0, 08000_0, 10000_0, 12500_0, 16000_0, 20000_0}
    
# Individual conda installation commands
```
conda install seaborn
conda install -c conda-forge pingouin
conda install -c conda-forge jupyterlab
conda install pandas
conda install -c conda-forge pysoundfile
conda install -c conda-forge librosa
conda install -c conda-forge scikit-learn
conda install -c conda-forge tensorflow-gpu
conda install -c conda-forge python-wget
```

# Version history

- 0.0.0 : Initial release
