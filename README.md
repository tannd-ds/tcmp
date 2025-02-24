# TCMP

Official PyTorch implementation of Temporal Convolutional Motion Predictor (TCMP) in Our paper **ime-series Meets Complex Motion Modeling: Robust and Computational-effective Motion Predictor for Multi-object Tracking**

## Abstract

Multi-object tracking (MOT) is critical in numerous real-world applications, including surveillance, autonomous driving, and robotics. Accurately predicting object motion is fundamental to MOT, but current methods struggle with the complexities of real-world, non-linear motion (e.g., sudden stops, sharp turns). Traditional approaches, optimized for linear trajectories, are often fail to capture the complexities of these unpredictable movements.

To address this, we propose **T**emporal **C**onvolutional **M**otion **P**redictor (TCMP), a novel framework for MOT that leverages a modified Temporal Convolutional Network (TCN) with dilated convolutions and a regression head to predict motion effectively across arbitrary temporal context lengths.

Experimental results demonstrate that our approach achieves state-of-the-art performance, specifically improves upon the previous best method in several key metrics: HOTA (a measure of overall tracking accuracy) increases from 62.3\% to 63.4\%, IDF1 (a measure of identity preservation) rises from 63.0\% to 65.0\%, and AssA (a measure of association accuracy) improves from 47.2\% to 49.1\%, while is only 0.014 times the size (in terms of parameters) and requires only 0.05 times the computational cost (in terms of FLOPs). These findings highlight the robustness of our method to advance MOT systems by ensuring adaptability, accuracy, and efficiency in complex tracking environments.

## Performance

Tracking performance on DanceTrack dataset:

| Method	       | HOTA	 | IDF1	 | AssA	 | MOTA	 | DetA |
|---------------|-------|-------|-------|-------|------|
| FairMOT	      | 39.7	 | 40.8	 | 23.8	 | 82.2	 | 66.7 |
| MOTR	         | 54.2	 | 51.5	 | 40.2	 | 79.7	 | 73.5 |
| OC-SORT	      | 55.1	 | 54.9	 | 40.4	 | 92.2	 | 80.4 |
| MotionTrack	  | 58.2	 | 58.6	 | 41.7	 | 91.3	 | 81.4 |
| GeneralTrack	 | 59.2	 | 59.7	 | 42.8	 | 91.8	 | 82.0 |
| DiffMOT	      | 62.3	 | 63	   | 47.2	 | 92.8	 | 82.5 |
| TCMP (Ours)   | 63.4  | 65.0  | 49.1  | 92.0  | 82.0 |

Tracking performance on SportsMOT dataset 

> Note: * indicate that their detectors are trained on SportsMOT train and val sets.

| Method	       | HOTA     | IDF1  | AssA  | MOTA  | DetA  |
|---------------|----------|-------|-------|-------|-------|
| FairMOT	      | 39.3	    | 53.5	 | 34.7	 | 86.4	 | 70.2  |
| GTR           | 54.5     | 55.8  | 45.9  | 67.9  | 64.8  |
| QDTrack       | 60.4     | 62.3  | 47.2  | 90.1  | 77.5  |
| CenterTrack   | 62.7     | 60.0  | 48.0  | 90.8  | 82.1  |
| ByteTrack     | 62.8     | 69.8  | 51.2  | 94.1  | 77.1  |
| TransTrack    | 68.9     | 71.5  | 57.5  | 92.6  | 82.7  |
| MOTIP         | 71.9     | 75    | 62    | 92.9  | 83.4  |
| OC-SORT	      | 71.9	    | 72.2	 | 59.8	 | 94.5	 | 86.4  |
| DiffMOT	      | 72.1	    | 72.8	 | 60.5	 | 94.5	 | 86    |
| TCMP (Ours)   | 73.3     | 74.16 | 62.6  | 94.12 | 85.93 |
| ---           | ---      | ---   | ---   | ---   | ---   |
| ByteTrack*    | 64.1     | 71.4  | 52.3  | 95.9  | 78.5  |
| MixSort-Byte* | 65.7     | 74.1  | 54.8  | 96.2  | 78.8  |
| OC-SORT*      | 73.7     | 74.0  | 61.5  | 96.5  | 88.5  |
| MotionTrack*	 | 74	      | 74	   | 61.7	 | 96.6	 | 88.8  |
| GeneralTrack* | 	  74.1	 | 76.4	 | 61.7	 | 96.8	 | 89.0  |
| DiffMOT*	     | 76.2	    | 76.1	 | 65.1	 | 97.1	 | 89.3  |
| TCMP* (Ours)  | 76.3     | 76.5  | 65.3  | 96.8  | 89.2  |


## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/tannd-ds/tcmp.git
    cd tcmp
    ```

2. Create a virtual environment and activate it:
    ```sh
    conda create -n tcmp python=3.10
    conda activate tcmp
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Install external packages:
   ```sh
   cd external/YOLOX/
   pip install -r requirements.txt && python setup.py develop
   cd ../../external/deep-person-reid/
   pip install -r requirements.txt && python setup.py develop
   cd ../../external/fast_reid/
   pip install -r docs/requirements.txt
   ```
   
## Data Preparation

Prepare the datasets by following the directory structure below:

- DanceTrack: 
```
 dancetrack/
 |-- train/
 |   |-- dancetrack0001/
 |   |   |-- img1/
 |   |   |   |-- 00000001.jpg
 |   |   |   |-- ...
 |   |   |-- gt/
 |   |   |   |-- gt.txt            
 |   |   |-- seqinfo.ini
 |   |-- ...
 |-- val/
 |   |-- ...
 |-- test/
     |-- ...
```

- SportsMOT:
```
 sportsmot/
 |-- train/
 |   |-- v_1LwtoLPw2TU_c006/
 |   |   |-- img1/
 |   |   |   |-- 00000001.jpg
 |   |   |   |-- ...
 |   |   |-- gt/
 |   |   |   |-- gt.txt            
 |   |   |-- seqinfo.ini
 |   |-- ...
 |-- val/
 |   |-- ...
 |-- test/
     |-- ...
```

> Note: Remember to adjust the paths in the configuration file accordingly.

## Usage
### Training
To train the motion model, run:
```sh
python main.py \
       --config path/to/config.yml \
       --eval False \
       --model_dir whatever_model_name_you_want
```

Your trained model will be saved to `./experiments/{model_dir}/weights/`.

### Evaluation

#### Prepare the detection results

- You can obtain the detections by the link that we have provided (Update soon).

#### Prepare the ReID module
- Update soon.

#### Prepare the Motion model

- You can obtain the TCMP model by training the model as mentioned above. Note that when you run the training, the code will automatically save the model to `./experiments/{prefix}_{model_dir}/weights/`, where `prefix` is the timestamp of the training session. To evaluate the model, you need to specify the `model_dir` as the path to the model directory, along with the `epochs` you want to evaluate.

#### Evaluate the model

To evaluate a model, run:
```sh
python main.py \
       --config path/to/config.yml \
       --eval True \
       --model_dir an_exist_model_dir \
       --epochs 200 # of epochs you want to evaluate
```

## Acknowledgements
- [PyTorch](https://pytorch.org/)
- [YAML](https://yaml.org/)
