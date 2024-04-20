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
`mini_projects` 