# DAT255 Course project
By fredrik Eide Fluge.

This repository represents my work on the course project for DAT255 Sprint 2024. My task was to perform deep learning on EEG recordings.
EEG recordings are measurements of the electronic signals in the brain over time. This can be used to diagnose diseases like a brain tumor, dementia and sleep disorder. Since EEGs are data recorded over time, it can be interesting to use different techniques, like still pictures of a single time frame or using recurrent networks like LSTM.
This README file should serve as a starting point to understand and use the code provided in this repository.

## Repository structure
The current project structure is as follows:
```
├─ mini_projects
│  ├─ bio_fouling
│  │  └─ <mini project about classifying bio fouling images>
│  └─ testing
│     └─ <small testing project with rust and ml>     
│
├─ imaginary-movement
│  ├─ data
│  │  ├─ files
│  │  │  └─ <edf files downloaded from >
│  │  └─ generated
│  │     └─ <generated image files>
│  │
│  ├─ data_explorer.ipynb
│  ├─ fourier_transformed.ipynb
│  ├─ side_classification.ipynb
│  ├─ task_reckognizion.ipynb
│  ├─ sql_data_manager.py
│  └─ data_merger.py
│  
└─ ADHD
   ├─ data
   │  ├─ files
   │  │  ├─ EEG
   │  │  │  └─ <matlab files downloaded from >'
   │  └─ generated
   │     └─ <generated image files> 
   │
   ├─ audio_based.ipynb
   ├─ matlab_to_images.ipynb
   ├─ multilabel_classifier.ipynb
   └─ singlelabel_classifier.ipynb
```
`mini_projects` are smaller projects done during the course, including the mini project presented in front of class. The images for this project is added to the repository.

For the course project, `ADHD` and `imaginary-movement` are the main projects in the course project. The initial project I started with was the `imaginary-movement`, but I moved over to the `ADHD` project later in an attempt to simplify my project, I'll talk more about this later.

## NOTE
for the majority of this project i tried to use LSTM cells for series training and resnet34 for timeless training, the results turned out to be quite bad. for future work I'd wish to use a different deep learning mechanism on the images as the result might have been different.

## Imaginary movement
`imaginary-movement` is the first attempt at deep learning done on EEG in this course project. This project uses the [EEG Motor Movement/Imagery Dataset from PhysioNet](https://www.physionet.org/content/eegmmidb/1.0.0/). This dataset consist of 109 participants that performed 14 recordings each of different tasks. 2 of the recordings are baseline models (lying still with open and closed eyes), and the remaining 12 consist of a combination of different tasks performed at regular intervals. This dataset used 64 channels of recording and with the large amount of data, I thought it would be a simple task to train and work on this data set. The attempt has 3 important notebooks (4 if includiing a started, but not finished notebook).
`data_explorer.ipynb` is used to preparse the data into a more manageable format. This converts the .edf files into a sqlite database, exposing the data as byte buffers, with a label buffer of equal length, This was done to prevent multiple file reads during training, and to unify the recordings to the same recording frequency(I used 160 Hz as the target frequency, as this was the denominator in the data set).

### Before running the notebooks
Download the zip file with data  
|     |     | 
| --- | --- |
| raw url to file | [https://www.physionet.org/static/published-projects/eegmmidb/eeg-motor-movementimagery-dataset-1.0.0.zip](https://www.physionet.org/static/published-projects/eegmmidb/eeg-motor-movementimagery-dataset-1.0.0.zip) |
| url to page with download | [https://www.physionet.org/content/eegmmidb/1.0.0/](https://www.physionet.org/content/eegmmidb/1.0.0/) |

And extract the data into the folder `imaginary-movement/data/files`, and move the data folders (`S001, S002, ..., *.png, *.pdf, ect..`) into the files folder. The structure should then be `imaginary-movement/data/files/S001/S001R01.edf` (for one of the data points.) Then run the `data_explorer.ipynb` notebook. This will parse and produce a database called `no_baseline_model.db`(this model skips the inital 2 records for each subject as every recording starts with a baseline). You can the run either the `task_reckognizion.ipynb`(uses all the tasks as unique labels).
For using the `side_learner.ipynb` (this is more of a failed attempt as the dataset produces is unbalanced for this training, and the baseline label (0) is favoured by the resulting model), run the `fourier_transformed.ipynb` notebook, this generates the required images in `ADHD/data/generated`.

### How it went
This dataset was my inital dataset and I put a lot of effort to preprocess the data for training, The initial edf files had the signals as continuous streams and the annotations as a combination of start time and length, this required som parsing to create a proper annotation buffer. The time values in the annotation buffer needed some treatment, as there were som rounding errors present in the buffers, this was done by iterating over the values and checking wether the specified start value corresponded with the expected value for the annotation (since we know the frequency and can determine the time of each entrty in the time series), some of the entries could not be determined, but this resulted in the loss of about 15% of the total files (1296/1526) and this seemed ok at the time.
The data is stored as is (in the shape sample_length*64), and with the inital data types (u16 for signals and u8 for labels), this produced a database of 3GiB (about the same as the inital files), this database could be accessed as a fastai dataset and was easy to plug into my notebook.

The first attempt `task_reckognizion.ipynb` uses LSTM memory cells and initally used subsequences (of one second) to train on, but was increased to the full sequences as the chunking implementation i used caused a lot of memory usage (each chunk loaded the entire signal buffer into memory, causing a lot of duplicates). After i dropped the chunking, the training time was reduced, but the result was still not the best, as I have yet to figure out how to do any form of preprocessing on the data before feeding it to the LSTM cell. This is when i started to use images to train on the data, the initial transoformation generated images of size 160*64 and packed the complex number generated by the fourier transform as the red and green channel.
[img7](./Imaginary-movement/r0o371.png)
Some of the images generated has a lot of black or repeating patterns (probably due to errors when recording), but they were not removed due to the amount of data produced (about 60000 images). The training did not yield any good results as the labels where overbalanced with the none label having over 1000 entries while left, right, and both only had about 200-300 entries each. I think the result could have been better if i'd use the entire label range rather than the generateed labels none, left, right, both. 



## ADHD
`ADHD` is the seccond attempt in this course project. This project was an new attempt at using timeless data (by transforming it to images and performing deep learning on each image). The dataset in used is collected from [A Dataset of EEG Signals from Adults with ADHD and Healthy Controls: Resting State, Cognitive function, and Sound Listening Paradigm](https://data.mendeley.com/datasets/6k4g25fhzg/1). and contains recordings of men and women with and without ADHD. this dataset was chosen because of the few labels available (man or woman, ADHD or control), and I thought it would function well for deep learning, the data in this set is stored in 4 matlab files `FADHD.mat`, `FC.mat`, `MADHD.mat` and `MC.mat`. Each file contains 11 cells containing the recordings of each subject.
```
Structure of the data:
cells[11]
  |-> subjects[X]
        |-> recordings[length, 2]
```

### Before running the notebooks
Download the dataset parts from: 
[A Dataset of EEG Signals from Adults with ADHD and Healthy Controls: Resting State, Cognitive function, and Sound Listening Paradigm](https://data.mendeley.com/datasets/6k4g25fhzg/1). There should a link to a 50.1 MB file. Unpack the zip file and move the folder `EEG/..` into `ADHD/data/files`. The resulting file tree should be `ADHD/data/files/FADHD.mat`, `ADHD/data/files/FC.mat`, `ADHD/data/files/MADHD.mat` and `ADHD/data/files/MC.mat`.

If you wish to run the notebook `multilabel_classifier.ipynb` or `singlelabel_classifier.ipynb`, run the notebook `matlab_to_images.ipynb` first, as this will generated the required time series images for the notebook. The notebook `audio_based.ipynb` is an unfinished attempt at using spectrograms and does not use the generated images (produces its own data).


### how it went
Each of the recordings contain 2 channels, F4 is common amongs all the recordings, but Cz, F3, O1 and FZ are used on different cells. Initially I generated an image with width 5 to add everything into a single image format, but with each channel placed at a specific location:
[img1](./ADHD/c0p0s1.png)
[img2](./ADHD/c9p3s64.png)
[img3](./ADHD/c10p0s3.png)
[img4](./ADHD/c6p12s77.png)


This didn't work very well and the results turned out to be preferring one side of labels, this might have been due to the fourier transform not being performed correctly(still being a times series), and messing up the results, and that 3/5 of each image is 0, (overbalanced). I tried to mitigate this by just placing lines (rather then 5) and merge the different channels together an differentiate them using the filename (like `c0pos1dCzF4.png`)
[img5](./ADHD/c0p0s9dCzF4.png)

The second attempt at ADHD split the data into 4 data sets (based on what channels are present) and were trained on either the gender label or the adhd label. This resulted int 8 models, and a common pattern amongs them were to favour male negative(I think this is caused by male and negative being the label with value 0).

I used the `resnet34` model for the ADHD, and i think i could have gotten a better result if i'd used a different model or a untrained resnet34, as the data does not actually correspond to any normal image data (no circles or lines present).

The last created file `audio_based.ipynb` is an unfisished, WIP model using spectrograms for training, the reason for this is that I finally found out what it means to make a time series timeless, and I'd like to try out this method with an applicable learner for this kind of data (probably not resnet34). I don't think I'll get it done by the due date but I'll try to complete it before the end of the course.
