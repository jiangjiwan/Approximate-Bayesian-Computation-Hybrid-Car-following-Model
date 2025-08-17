#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import timeit

from traj_data import *
from models import *


class ABC:
    def __init__(self, input_real_data, output_real_data, model, use_parallel):
        np.random.seed(0)
        self.max_para_dim = 10
        self.use_parallel = use_parallel
        self.real_input = input_real_data
        self.real_output = output_real_data
        self.model = model
        self.parameter_dim = model.nPara
        self.parameter_name = model.strPara
        
    def load_priori(self,parameter_samples):
        if parameter_samples.shape[1] != self.parameter_dim:
            raise ValueError('Trajectory Dimension Not Matching')
        
        self.parameter_samples = np.concatenate((parameter_samples, np.zeros([parameter_samples.shape[0],self.max_para_dim - parameter_samples.shape[1]])), axis = 1)
        self.num_samples = self.parameter_samples.shape[0]
        
    
    def run(self):
        using_parallel = self.use_parallel
        num_cores = multiprocessing.cpu_count()
        self.error = {}     
        
        if using_parallel:
            def process(n):
                sim_output = self.model.batch_simulation(self.real_input, self.real_output, self.parameter_samples[n,:])          
                return sim_output.distance(self.real_output)

            self.error = Parallel(n_jobs=num_cores)(delayed(process)(n) for n in range(self.num_samples))
        else:                                     
            for n in range(self.num_samples):
                sim_output = self.model.batch_simulation(self.real_input, self.real_output, self.parameter_samples[n,:])
                self.error[n] = sim_output.distance(self.real_output)
        
        self.parameter_samples = np.concatenate((np.zeros([self.num_samples, 5])+np.Inf, self.parameter_samples), axis = 1)
        for n in range(self.num_samples):
            self.parameter_samples[n,0] = self.error[n][0]
            self.parameter_samples[n,1] = self.error[n][1]
            self.parameter_samples[n,2] = self.error[n][2]
 
            self.parameter_samples[loc,3] = sample_id_list[loc]
            self.parameter_samples[loc,4] = self.model.model_id
    
        return self.parameter_samples
    
    def run_downsample(self, sample_id_list, batchsize):
        using_parallel = self.use_parallel
        n_batch = int(np.floor(self.num_samples/batchsize))
        if n_batch * batchsize > self.num_samples:
            raise ValueError('Sample Dimension Not Matching')
        if n_batch * batchsize < self.num_samples:
            print('%d Samples are truncated.' % (self.num_samples - n_batch * batchsize))
        
        print('%d batches with %d Samples for each batch.' % (n_batch, batchsize))
        num_cores = multiprocessing.cpu_count()
        self.error = {}     
        
        if using_parallel:
            def process(n):
                list_error = []
                for bn in range(batchsize):
                    loc = n*batchsize + bn
                    sim_output = self.model.downsample_simulation(self.real_input, self.real_output, self.parameter_samples[loc,:], sample_id_list[loc])          
                    list_error.append(sim_output.distance(self.real_output))  
                return list_error
            self.error = Parallel(n_jobs=num_cores)(delayed(process)(n) for n in range(n_batch))
        else: 
            for n in range(n_batch):
                list_error = []
                for bn in range(batchsize):
                    loc = n*batchsize + bn
                    sim_output = self.model.downsample_simulation(self.real_input, self.real_output, self.parameter_samples[loc,:], sample_id_list[loc])
                    list_error.append(sim_output.distance(self.real_output))
                self.error[n] = list_error

        
        self.parameter_samples = np.concatenate((np.zeros([self.num_samples, 5])+np.Inf, self.parameter_samples), axis = 1)
        for n in range(n_batch):
            for bn in range(batchsize):
                loc = n*batchsize + bn
                
                self.parameter_samples[loc,0] = self.error[n][bn][0]
                self.parameter_samples[loc,1] = self.error[n][bn][1]
                self.parameter_samples[loc,2] = self.error[n][bn][2]
                
                self.parameter_samples[loc,3] = sample_id_list[loc]
                self.parameter_samples[loc,4] = self.model.model_id
            
        return self.parameter_samples
    
    def save_result(self, file):
        header = ''      
        header += 'Error_p, Error_s, Error_a,'
        header += 'Traj_id, Model_id'
        for i in range(self.parameter_dim):
            header += ',' + self.parameter_name[i] 
        for i in range(self.parameter_dim, self.max_para_dim):
            header += ', _'
        
        
        self.parameter_samples = self.parameter_samples[self.parameter_samples[:,0].argsort()]
        
        np.savetxt(file, 
                   self.parameter_samples, 
                   delimiter = ",", fmt='%.4f', 
                   header = header)

def uniform_priori_gen(N, bd_file):
    bd = np.loadtxt(open(bd_file, "rb"), delimiter=",", skiprows=0)
    model_dimension = bd.shape[0]
    parameter_samples = np.zeros((N, model_dimension))
    for n in range(N):
        for m in range(model_dimension):
            parameter_samples[n,m] = np.random.uniform(bd[m,0], bd[m,1])  
    
    return parameter_samples

def shell(model_name, traj_data_path, para_priori_file, result_path, ts, num_samples):
    
    num_cores = multiprocessing.cpu_count()
    batchsize = int(np.floor(num_samples / num_cores))
    use_downsample = True
    use_parallel = True
    
    leader_vec, follower_vec = read_traj(traj_data_path, ts)
    parameter_samples = uniform_priori_gen(num_samples, para_priori_file) 
    my_model = MODEL[model_name]()
    my_ABC = ABC(leader_vec, follower_vec, my_model, use_parallel)
    my_ABC.load_priori(parameter_samples)
    
    tic = timeit.default_timer()
    if use_downsample:
        sample_id_list = np.random.randint(0, leader_vec.num_veh, num_samples)
        my_ABC.run_downsample(sample_id_list, batchsize)
        my_ABC.save_result(result_path)
  
    else:   
        my_ABC.run()
        my_ABC.save_result(result_path)
        
    toc = timeit.default_timer()
    if use_parallel:
        print('Time consumption under parallel computing %d sec ' % (toc - tic))
    else:
        print('Time consumption under serial computing %d sec' % (toc - tic))

if __name__ == '__main__':
    
    num_samples = 1000001
    ts = 0.1
    Data_set = 'PRIUS_CROSS'
    traj_data_path = '../data/%s/3_Trajectory/' % Data_set
    
    for model_name in ['IDM']:
    #for model_name in ['OVM', 'GFM', 'FVM', 'IDM', 'LL', 'LLCS', 'HL','MPC']: #, 'MPC'
    #for model_name in ['MPC']: ## ,  'MPC'   'IDM', 'FVM', 'GFM','OVM'   FVM_CS' ,'Newell','IDM_CS','IDM_CTG',FVM_SIGMOID
        para_priori_file = '../priori/%s_uniform.csv' % model_name
        result_path = '../Cross_result_%d_%s_%s_3_new.csv' % (num_samples, model_name, Data_set)

        shell(model_name, traj_data_path, para_priori_file, result_path, ts, num_samples)


# In[ ]:





# In[ ]:




