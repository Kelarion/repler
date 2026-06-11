# -*- coding: utf-8 -*-
"""
@author: valer
"""

CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/'
LOAD_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/valeria_data/' 

import os, sys, re
import pickle as pkl
from time import time
import math
sys.path.append(CODE_DIR)
sys.path.append('C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/')


from read_Munuera_files import Channel_oldSorting, getSpikeCount

from scipy.io import loadmat
import numpy as np         
        
#%% Readout all matlab files  (it takes a while.... )

### SET FIRST THESE PARAMETERS

id_current_trials = 'REWARD'  #'REWARD' or 'FAIL': whether to analyze correct or error trials
event_align = 'RealVS_ON'     # task event to align: RealVS_ON=stimulus onset
# brain_area  = 'Amyg/'         # ACC OFC AMYG
brain_area  = 'ACC/'         # ACC OFC AMYG
# datatype    ='Amygsingle.mat' #Amyg, ACC, OFC and single or multi unit
datatype    ='ACCmulti.mat' #Amyg, ACC, OFC and single or multi unit


final_sessions    = []

single_neuron_filepath = LOAD_DIR + '/Multi_and_single_cell_names/'+datatype #Amyg, ACC, OFC
single_neuron_sessions = loadmat(single_neuron_filepath, squeeze_me=True)
single_neuron_sessions = single_neuron_sessions['Cell']       
nSessions   = len(single_neuron_sessions)         
listSessions = [str(obj) for obj in single_neuron_sessions] 


## organize trials per each 8 conditions:
spike_times_aligned = []    
spike_times_aligned_allConds = []

nTotSessions = 0
trials_id =[[[]]]*nSessions
behavior_performance = []
cond_labels =[]
rt_cond = [[[]]]*nSessions
for iSession in range(nSessions): 
  
   if listSessions[iSession] == 'save':continue
      
   print('iSession ', iSession,'/',nSessions-1)
   spike_times_aligned_app = []
   
   
   path_mat_neural_file = LOAD_DIR+brain_area+single_neuron_sessions[iSession][:-4]+'.mat' 

   prova = Channel_oldSorting(path_mat_neural_file)
   
   pos_good_neurons = prova.getGoodNeurons()
   tot_neurons = len(pos_good_neurons)
   tot_trials  = prova.getTotTrials() 
   
   events_arr, events_arr_labels = prova.getBehavior(tot_trials)
   event_dict = prova.getDictEvents() 
   
   spike_times = prova.getSpikeTimesTrial(tot_neurons, pos_good_neurons, events_arr, range(tot_trials))
   
   trials_after_switch = prova.getTrialsAfterSwitch(events_arr) # it returns the trial ID after one switch
   no_empty_trials = prova.getNonEmptyTrials(pos_good_neurons) # rimuovo trials vuoti, quelli con un solo spike in tutto il trials
   
   trials_behavior = np.setdiff1d(no_empty_trials,trials_after_switch)
   behavior_app, _ = prova.get_behavior_sequence(events_arr, trials_behavior)
   behavior_performance.append(behavior_app)
   cond_labels.append(prova.get_condition_label(events_arr, trials_behavior))
           
   trials_final_all = []
   trials_id[iSession] = [[]]*8
   rt_cond[iSession] = [[]]*8
   for iCondition in range(8):  # Cximages from 0->7
         
       events_name = [id_current_trials, 'Cximages', iCondition]  #give the name of the events you want to select            
       spike_trials_good, trials_final = prova.getSpecificTrials(spike_times[0], events_arr, events_name, no_empty_trials, trials_after_switch, tot_neurons)  
       trials_final_all = np.concatenate(([trials_final_all,trials_final]))            
       trials_id[iSession][iCondition]= np.where(prova.valid_trials==1)[0][trials_final]              
       
       # align spike times to event_align
       spike_times_aligned_app.append(prova.getSpikeTimesAligned(tot_neurons, spike_times[0], events_arr, event_align, trials_final))             
          
                  
   nTotSessions+=1                
   spike_times_aligned.append(spike_times_aligned_app)    
   final_sessions.append(iSession)
   
#%% compute the spike count
## spike count is a list: spike_count[N_neurons][C_conditions][n_trialsxn_bins]
## C_conditions=8

# condition [i] of Bernardi task (stim id, context, action, reward):
# [0]  (A, 0, Release, +)
# [1]  (B, 0, Hold,    +)
# [2]  (C, 0, Release, 0)
# [3]  (D, 0, Hold,    0)
# [4]  (B, 1, Release, +)
# [5]  (C, 1, Hold,    +)
# [6]  (D, 1, Release, 0)
# [7]  (A, 1, Hold,    0)

    
binMin   = 0.6 #ms
binMax   = 1
binWidth = 0.2    
binStep  = 0.2  
time_edges  = [binMin, binMax, binWidth, binStep]   
spike_count = []    
good_neurons = []
for iSession in range(nTotSessions):
   print(iSession) 
   spike_count_app = []
   for iCondition in range(8):            
       spike_count_app.append(getSpikeCount(spike_times_aligned[iSession][iCondition][0], time_edges)) # forse troppo lento, da accelerare
   spike_count.append(spike_count_app)   
   good_neurons.append(iSession)   


