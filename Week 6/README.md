# Week 6: Multimodal human analysis

## Group Name: Group 08
### Team Members:
- Cristina Aguilera (Cristina.AguileraG@autonoma.cat)
- Ainoa Contreras (Ainoa.Contreras@autonoma.cat)
- Jordi Morales (Jordi.MoralesC@autonoma.cat)
- Luis González (Luis.GonzalezGu@autonoma.cat)

## Folder structure 
The code and data is structured as follows:<br>
   ```
    |____data/
    |    |____test/
    |    |____train/
    |    |____valid/
    |    |____test_set_age_labels.csv
    |    |____train_set_age_labels.csv
    |    |____valid_set_age_labels.csv
    |____evaluation_script/
    |    |____evaluate.py/
    |    |____test_set_age_labels.csv
    |____README.md
    |____data_utils.py
    |____data_utils_g.py
    |____exploration.ipynb
    |____plot_utils.py
    |____task_b_model.py
    |____task_b_train.py
    |____task_g_model.py
    |____task_g_train.py
    |____task_h_model.py
    |____task_h_train.py
    |____train_utils.py
    |____train_utils_g.py
   ```

In this structure:
* `data/`: Data folder. It has to be downloaded from the original source's URL.
* `evaluation_script/evaluate.py`: Baseline code for evaluating predictions (Accuracy and Bias), modified to also process aggregated predictions.
* `evaluation_script/test_set_age_labels.csv`: File containing the Dataset's ground truths for AgeGroup, Gender and Ethnicity.
* `README.mb`: ReadME file.
* `data_utils.py`: Utils files.
* `data_utils_g.py`: Utils files specific for Task G.
* `plot_utils.py`: Utils file for plotting results during training.
* `train_utils.py`: Utils files containing training-related functions.
* `train_utils_g.py`: Utils files containing training-related functions, specific for task G.
* `exploration.ipynb`: Data exploration Python notebook, used in Task A.
* `task_b_model.py`: Model Class used in Task B.
* `task_g_model.py`: Model Class used in Task G.
* `task_h_model.py`: Model Class used in Task H.
* `task_b_train.py`: Training Script used in Task B.
* `task_g_train.py`: Training Script used in Task G.
* `task_h_train.py`: Training Script used in Task H.

## Instructions

`exploration.ipynb`: Since it is a Python Notebook, it is only required to have a text editor able of running this kind of file (Jupyter Notebook or VSCode).

`task_X_train.py`: Running Training script for task (B, G or H).
    ```
    
    $ python task_X_train.py  data_path mode

        Train/test model for Task X (B, G or H)

        positional arguments:
        data_path            Directory containing the Dataset data (example: ./data).
        mode                 Execution mode: 'train' or 'test', not required for task H.
       
        
`evaluation_script/evaluate.py`: Obtain evaluation metrics. The path to the predictions file has to be set from inside the code. 
    ```
    
    $ python evaluation_script/evaluate.py
    
