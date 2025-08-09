# Approximate Bayesian Computation for Hybrid Car-Following Models

This repository contains the implementation of stochastic car-following (CF) model calibration using **Approximate Bayesian Computation (ABC)**, as presented in:  
- *"Stochastic Calibration of Automated Vehicle Car-Following Control: An ABC Approach"*  
- *"A Generic Stochastic Hybrid Car-Following Model Based on ABC"*  

<img width="465" height="336" alt="image" src="https://github.com/user-attachments/assets/a0a0a3e2-0b62-4bca-8766-802ee0c6ebc7" />  
<img width="890" height="1166" alt="image" src="https://github.com/user-attachments/assets/9aa6c18d-5b1f-4e34-b650-83786e240e47" />  

---

## How to Run Calibration
We calibrate models using:  
- **AV data**: [Massachusetts AV Experiment](https://doi.org/10.1016/j.trb.2021.03.003)  
- **HDV data**: [NGSIM](https://www.fhwa.dot.gov/publications/research/operations/06137/)  

Noise-processed datasets are stored in `data/AV` and `data/HDV`.  
To preprocess raw data, please download the original datasets and run the provided preprocessing scripts.  

The upper and lower bounds of the uniform priors for:  
- **HDV CF models**: OVM, IDM, GFM, FVDM  
- **AV controllers**: LL, LLCS, HL, MPC  
are stored in `prior/*.csv`.  

**Execution steps:**  
1. Ensure the following files are downloaded into `/src`:  
   - `traj_data.py`  
   - `car_following_models_new.py`  
2. Run `approximate_bayesian_new.py` to perform ABC-based calibration.

---

## How to Run Analysis
Analysis scripts are located in `/Analysis`.  
Calibration results from the previous step must be saved and loaded into:  
- `hybrid_analysis_parallel-V2plot.ipynb`

---

## Read More
- AV stochastic calibration using ABC: [IEEE T-ITS (2025)](https://doi.org/10.1109/TITS.2025.3526318)  
- Hybrid model calibration: [TRC (2024)](https://doi.org/10.1016/j.trc.2024.104799)  

---

## Contact
For questions, please contact:  
- **Jiwan Jiang** – jiangjiwan2@gmail.com  
- **Soyoung Ahn** – sue.ahn@wisc.edu  


