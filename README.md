
# Path-Specific Counterfactual Fairness for Recommender Systems

The codes are associated with the following paper:
>**Path-Specific Counterfactual Fairness for Recommender Systems,**    
>Yaochen Zhu, Jing Ma, Liang Wu, Qi Guo, Liangjie Hong, Jundong Li,    
>SIGKDD 2023.

## 1. Environments
The codes are written in Python 3.6.5.
- numpy == 1.16.3
- pandas == 0.21.0
- tensorflow-gpu == 1.15.0
- tensorflow-probability == 0.8.0

## 2. Dataset Acquirement and Simulation
### 2.1 For the Amazon-VG and ML-1M datasets:
- **Acquire the Amazon-VG and ML-1M datasets:**    
	The original datasets can be found [[here]](https://grouplens.org/datasets/movielens/1m/) and [[here]](https://jmcauley.ucsd.edu/data/amazon/).    
	Preprocess the data with data_sim/raw/prepare_data.py.    

- **Preprocess the original dataset:**   
	cd to data_sim/raw folder, run   
```python prepare_data.py --dataset Name --simulate exposure```.  

 - **Fit the binarized rating distributions via VAEs:**   
	cd to data_sim folder, run   
```python train.py --dataset Name --simulate exposure```.   

- **Simulate the datasets:**   
```python simulate.py --dataset Name```.   

 - **The simulated datasets are in psf-vae/data folder**  
 
  ### 2.2 For the LinkedIn Datasets:   
This for now is a private dataset.  

## 3. Model Training and Evaluation

- **Split the simulated datasets into train/val/test:**
cd to psf-vae/data folder, run   
```python preprocess.py --dataset Name --split 10```.   

- **Train the PSF-VAE model:**   
```python train.py --dataset Name --split [0-9]```   

 - **Evaluate the model based on recommendation performance and fairness:**   
```python eval.py --dataset Name --split [0-9]```   

 - **Summarize the results into a table:**   
```python summarize.py --dataset Name```   

## **Reference**
	@inproceedings{Zhu2023Path,  
	  title={Path-Specific Counterfactual Fairness for Recommender Systems},      
	  author={Zhu, Yaochen and Ma, Jing and Wu, Liang and Guo, Qi and Hong, Liangjie and Li, Jundong},    
	  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’23)},    
	  year={2023},
	}