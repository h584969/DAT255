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

And extract the data into the folder `imaginary-movement/data/files`, and move the data folders (`S001, S002, ..., *.png, *.pdf, ect..`) into the files folder. The structure should then be `imaginary-movement/data/files/S001/S001R01.edf` (for one of the data points.) Then run the `data_explorer.ipynb` notebook. This will parse and produce a database called `no_baseline_model.db`(this model skips the inital 2 records for each subject as every recording starts with a baseline).



## ADHD
`ADHD` is the seccond attempt in this course project. This project was an new attempt at using timeless data (by transforming it to images and performing deep learning on each image). The dataset in used is collected from [A Dataset of EEG Signals from Adults with ADHD and Healthy Controls: Resting State, Cognitive function, and Sound Listening Paradigm](https://data.mendeley.com/datasets/6k4g25fhzg/1). and contains recordings of men and women with and without ADHD. this dataset was chosen because of the few labels available (man or woman, ADHD or control), and I thought it would function well for deep learning, the data in this set is stored in 4 matlab files `FADHD.mat`, `FC.mat`, `MADHD.mat` and `MC.mat`. Each file contains 11 cells containing the recordings of each subject.
```
Structure of the data:
cells[11]
  |-> subjects[X]
        |-> recordings[length, 2]
```

``

### Before running the notebooks

Each of the recordings contain 2 channels, F4 is common amongs all the recordings, but Cz, F3, O1 and FZ are used on different cells. Initially I generated an image with width 5 to add everything into a single image format, but with each channel placed at a specific location:
[img1](./ADHD/c0p0s1.png)
[img4](./ADHD/c9p3s64.png)
[img2](./ADHD/c10p0s3.png)
[img3](./ADHD/c6p12s77.png)


This didn't work very well and the results