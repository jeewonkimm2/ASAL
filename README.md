# ASAL (Active Score for Active Learning)
- Active Learning Initialization for Supervised Visual Defect Detection
- [The 2023 Fall Conference of Korean Institute of Industrial Engineers](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11609814)
- Associated with [Industrial Artificial Intelligence Lab](https://iai.seoultech.ac.kr/index.do)

#### 📍 Project Objective
This study is a method of reflecting the distribution of actual industrial data in which normal and abnormal data are disproportionately mixed, and proposes an initialization methodology for solving the cold-start problem of active learning.

#### 📍 Project Duration
2023.01 ~ 2023.11

#### 📍 Team Member

<table style="border: 0.5px solid gray">
 <tr>
    <td align="center"><a href="https://github.com/jeewonkimm2"><img src="https://avatars.githubusercontent.com/u/108987773?v=4" width="130px;" alt=""></td>
    <td align="center" style="border-right : 0.5px solid gray"><a href="https://github.com/bae-sohee"><img src="https://avatars.githubusercontent.com/u/123538321?v=4" width="130px;" alt=""></td>

  </tr>
  <tr>
    <td align="center"><a href="https://github.com/jeewonkimm2"><b>Jeewon Kim</b></td>
    <td align="center" style="border-right : 0.5px solid gray"><a href="https://github.com/bae-sohee"><b>Sohee Bae</b></td>
  </tr>
</table>
<br/>


## 1. Overview
- Proposal of an Active Learning methodology using an Unlabeled Set reflecting the distribution of real industrial data with imbalanced normal and abnormal data.
- Utilization of Anomaly Detection's Anomaly Score to address the **"Cold-Start"** challenge in Active Learning.
- Construction of a balanced initial dataset using the Anomaly Score, allowing the model to grasp various data patterns in the early learning stages and reducing labeling costs.

  ![image](https://github.com/jeewonkimm2/ASAL/assets/108987773/47599c97-62a0-4f7c-bfa3-19937c747599)

  #### Overall Structure
  - Sampling Module for Initial Stage using Anomaly Score
  - Active Learning

 ## 2. Environment
- Python version is 3.9.
- Installing all the requirements may take some time. After installation, you can run the codes.
- Please notice that we used 'PyTorch' and device type as 'GPU'.
- We utilized 2 GPUs in our implementation. If the number of GPUs differs, please adjust the code accordingly based on the specific situation.
- [```requirements.txt```](https://github.com/jeewonkimm2/ASAL/blob/main/requirements.txt) file is required to set up the virtual environment for running the program. This file contains a list of all the libraries needed to run your program and their versions.

    #### In **Anaconda** Environment,

  ```
  $ conda create -n [your virtual environment name] python=3.9
  
  $ conda activate [your virtual environment name]
  
  $ pip install -r requirements.txt
  ```

  - Create your own virtual environment.
  - Activate your Anaconda virtual environment where you want to install the package. If your virtual environment is named 'testasal', you can type **conda activate testasal**.
  - Use the command **pip install -r requirements.txt** to install libraries.

 ## 3. Dataset
 - [The MVTec anomaly detection dataset (MVTec AD)](https://www.mvtec.com/company/research/datasets/mvtec-ad)
 - You need to create a folder, `./DATA`
 - Merging the Train and Test sets of the existing data, the combined dataset was then divided into Train and Test sets in a ratio of 0.8:0.2.

  ```bash
  ├── DATA
  │   ├── class1
  │   │    └── train
  │   │    └── test
  │   ├── class2
  │   │    └── train
  │   │    └── test
  │   │
  │   │    ...
  │   │
  │   │
  │   ├── class15
  │   │    └── train
  │   │    └── test
```

 ## 4. Prerequisites
 - For each category, the anomaly scores of the data are sorted in descending order and then divided into five batches, which are subsequently saved as text files.
 - In our experiment, we employed the [DRAEM (Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection)](https://openaccess.thecvf.com/content/ICCV2021/html/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.html) method to derive anomaly scores.
 - In `loss_batch` folder, you can check the anomaly scores we obtained.
 - `batch.txt` : The anomaly scores for the entire training dataset, sorted in descending order
 - `batchN.txt` : Nth batch among the five batches divided using batch.txt
  ```bash
  ├── loss_batch
  │   ├── anomaly_scores_draem
  │   │    └── class1
  │   │    │    └── batch.txt
  │   │    │    └── batch0.txt
  │   │    │    └── batch1.txt
  │   │    │    └── batch2.txt
  │   │    │    └── batch3.txt
  │   │    │    └── batch4.txt
  │   │    │
  │   │    │    ...
  │   │    │
  │   │    │
  │   │    └── class15
  │   │    │    └── batch.txt
  │   │    │    └── batch0.txt
  │   │    │    └── batch1.txt
  │   │    │    └── batch2.txt
  │   │    │    └── batch3.txt
  │   │    │    └── batch4.txt
```
- Each text file contains, in sequential order, the paths of the corresponding data. An excerpt from an example text file is provided below:
  ```bash
  ./DATA/cable/train/anomaly/anomaly_cable_missing_cable_011.png
  ./DATA/cable/train/anomaly/anomaly_cable_missing_cable_006.png
  ./DATA/cable/train/anomaly/anomaly_cable_missing_cable_004.png
  ./DATA/cable/train/anomaly/anomaly_cable_missing_cable_010.png
  ./DATA/cable/train/anomaly/anomaly_cable_missing_cable_009.png
  ./DATA/cable/train/anomaly/anomaly_cable_missing_cable_008.png
  ./DATA/cable/train/anomaly/anomaly_cable_missing_cable_000.png
  ...
  ```

## 5. Implementation
- You need to run `main.py`.

  ```bash
  python main.py --loss_type anomaly_scores_draem
  ```
  
- `--loss_type` : This should have the same name as the first subfolder within 'loss_batch'
- `checkpoint`, `main_best_acc`, and `main_best_auroc` folders will be created automatically during model training, and these folders can be used to monitor the performance of the model.

## 6. Result
  ![image](https://github.com/jeewonkimm2/ASAL/assets/108987773/2a6de16f-8855-4c83-b56f-4c067d6ae1b1)

  - ASAL(ours) : This is the proposed method in our study that utilizes Anomaly Score for balanced sampling in imbalanced datasets
  - Random: Random sampling of 10 samples from the entire Unlabeled Data
  - DRAEM_mixed: Anomaly detection performance when trained on all data, including a mixture of normal and abnormal data
  - DRAEM_normal: Anomaly detection performance when trained only on all normal data

## 7. Reference
- [1] Yi, J. S. K., Seo, M., Park, J., & Choi, D. G. (2022, October). Pt4al: Using self-supervised pretext tasks for active learning. In European Conference on Computer Vision (pp. 596-612). Cham: Springer Nature Switzerland.
- [2] Zavrtanik, V., Kristan, M., & Skočaj, D. (2021). Draem-a discriminatively trained reconstruction embedding for surface anomaly detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 8330-8339).
- Code Implementation is based on [1]
