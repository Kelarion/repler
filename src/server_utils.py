CODE_DIR = '/home/kelarion/github/repler/src/'
SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/multiclassification/'

REMOTE_SYNC_SERVER = 'ma3811@motion.rcs.columbia.edu' #must have ssh keys set up
REMOTE_CODE = '/burg/home/ma3811/repler/'
REMOTE_RESULTS = '/burg/theory/users/ma3811/results/'

import socket
import os
import sys
import pickle as pkl
import subprocess

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import itertools as itt

import util
import tasks
import students as stud
import experiments as exp
import grammars as gram


def send_to_server(exp_prm, net_args, opt_args, run_remote=True):
    """ 
    if run_remote=False, will just run the first experiment locally (for debugging)

    any values which are list-type will be combined together independently, any values with are 
    tuple-type will be matched together 1-1. 

    """

    ##### Pickle experiment parameters
    ##########################################

    variable_prms = {k:v for k,v in exp_prm.items() if type(v) is list and k!='experiment'}
    matched_prms = {k:v for k,v in exp_prm.items() if type(v) is tuple and k!='experiment'}
    fixed_prms = {k:v for k,v in exp_prm.items() if type(v) not in [list, tuple] and k!='experiment'}

    ############################## handle variable arguments 
    if len(matched_prms)>0: 
        if len(np.unique([len(v) for v in matched_prms.values()]))>1:
            raise ValueError('Tuple arguments must all be the same length you dingus!')
        tup_keys = tuple(matched_prms.keys())
        tup_vals = zip(*matched_prms.values())
    else:
        tup_keys = ()
        tup_vals = [()]

    if len(variable_prms)>0:
        var_k, var_v = zip(*variable_prms.items())
    else:
        var_k = ()
        var_v = ()
    ##############################

    exp_idx = 0
    for vals in list(itt.product(*var_v, tup_vals)):
        this_dset = dict(zip(var_k+tup_keys, vals[:-1]+vals[-1]), **fixed_prms)

        dset_info = {'experiment':exp_prm['experiment'], 'exp_args':this_dset}
        pkl.dump(dset_info, open(SAVE_DIR+'server_cache/task_%d.pkl'%exp_idx,'wb'))
        exp_idx += 1
    # else:
    #     exp_idx = 1
    #     dset_info = {'experiment':exp_prm['experiment'], 'exp_args':fixed_prms}
    #     pkl.dump(dset_info, open(SAVE_DIR+'server_cache/task_0.pkl','wb'))


    ##### Pickle network parameters
    ##########################################

    variable_net_prms = {k:v for k,v in net_args.items() if type(v) is list and k!='model'}
    matched_net_prms = {k:v for k,v in net_args.items() if type(v) is tuple and k!='model'}
    fixed_net_prms = {k:v for k,v in net_args.items() if type(v) not in [list, tuple] and k!='model'}

    variable_opt_prms = {k:v for k,v in opt_args.items() if type(v) is list }
    matched_opt_prms = {k:v for k,v in opt_args.items() if type(v) is tuple }
    fixed_opt_prms = {k:v for k,v in opt_args.items() if type(v) not in [list, tuple] }

    ############################## handle variable arguments 
    if len(matched_net_prms)>0: 
        if len(np.unique([len(v) for v in matched_net_prms.values()]))>1:
            raise ValueError('Tuple arguments must all be the same length you dingus!')
        tup_net_keys = tuple(matched_net_prms.keys())
        tup_net_vals = list(zip(*matched_net_prms.values()))
    else:
        tup_net_keys = ()
        tup_net_vals = [()]

    if len(matched_opt_prms)>0: 
        if len(np.unique([len(v) for v in matched_opt_prms.values()]))>1:
            raise ValueError('Tuple arguments must all be the same length you dingus!')
        tup_opt_keys = tuple(matched_opt_prms.keys())
        tup_opt_vals = list(zip(*matched_opt_prms.values()))
    else:
        tup_opt_keys = ()
        tup_opt_vals = [()]

    if len(variable_net_prms)>0:
        var_n_k, var_n_v = list(zip(*variable_net_prms.items()))
    else:
        var_n_k = ()
        var_n_v = ()

    if len(variable_opt_prms)>0:
        var_o_k, var_o_v = list(zip(*variable_opt_prms.items()))
    else:
        var_o_k = ()
        var_o_v = ()
    ##############################

    net_idx = 0
    for net_vals in list(itt.product(*var_n_v, tup_net_vals)):
        this_net = dict(zip(var_n_k+tup_net_keys, net_vals[:-1]+net_vals[-1]), **fixed_net_prms)
        for opt_vals in list(itt.product(*var_o_v, tup_opt_vals)):
            this_opt = dict(zip(var_o_k+tup_opt_keys, opt_vals[:-1]+opt_vals[-1]), **fixed_opt_prms)

            dset_info = {'model':net_args['model'], 
                         'model_args':this_net,
                         'opt_args':this_opt}

            pkl.dump(dset_info, open(SAVE_DIR+'server_cache/network_%d.pkl'%net_idx,'wb'))
            net_idx += 1

    # else:
    #     net_idx = 1
    #     dset_info = {'model':net_args['model'], 
    #                  'model_args':fixed_net_prms,
    #                  'opt_args':fixed_opt_prms}
    #     pkl.dump(dset_info, open(SAVE_DIR+'server_cache/network_0.pkl','wb'))


    print('\nSending %d jobs to server ...\n'%(net_idx*exp_idx))

    ###### Send to pickles server
    ##########################################
    if run_remote:

        print('[{}] Giving files to {}...'.format(sys.platform, REMOTE_SYNC_SERVER))

        cmd = 'rsync {local}*.pkl {remote} -v'.format(local=SAVE_DIR+'server_cache/',
            remote=REMOTE_SYNC_SERVER+':'+REMOTE_RESULTS)
        subprocess.check_call(cmd, shell=True)

        tmplt_file = open(CODE_DIR+'/job_script_template.sh','r')
        with open(SAVE_DIR+'server_cache/job_script.sh','w') as script_file:
            sbatch_text = tmplt_file.read().format(n_tot=net_idx*exp_idx - 1, n_task=exp_idx, file_dir=REMOTE_CODE)
            script_file.write(sbatch_text)
        tmplt_file.close()

        ####### Run job array
        ###########################################

        cmd = f"ssh ma3811@ginsburg.rcs.columbia.edu 'sbatch -s' < {SAVE_DIR+'server_cache/job_script.sh'}"
        subprocess.call(cmd, shell=True)

    else:
        print('Running job...')

        cmd = f"python {CODE_DIR}/run_experiment.py 0 1"
        subprocess.call(cmd, shell=True)


# for retrieving from the server
def get_all_experiments(exp_prm, net_args, opt_args, bool_friendly=True):

    ##### Loop over experiment parameters
    ##########################################

    variable_prms = {k:v for k,v in exp_prm.items() if type(v) is list and k!='experiment'}
    matched_prms = {k:v for k,v in exp_prm.items() if type(v) is tuple and k!='experiment'}
    fixed_prms = {k:v for k,v in exp_prm.items() if type(v) not in [list, tuple] and k!='experiment'}

    variable_net_prms = {k:v for k,v in net_args.items() if type(v) is list and k!='model'}
    matched_net_prms = {k:v for k,v in net_args.items() if type(v) is tuple and k!='model'}
    fixed_net_prms = {k:v for k,v in net_args.items() if type(v) not in [list, tuple] and k!='model'}

    variable_opt_prms = {k:v for k,v in opt_args.items() if type(v) is list }
    matched_opt_prms = {k:v for k,v in opt_args.items() if type(v) is tuple }
    fixed_opt_prms = {k:v for k,v in opt_args.items() if type(v) not in [list, tuple] }



    ############################## handle variable arguments 
    if len(matched_prms)>0: 
        if len(np.unique([len(v) for v in matched_prms.values()]))>1:
            raise ValueError('Tuple arguments must all be the same length you dingus!')
        tup_keys = tuple(matched_prms.keys())
        tup_vals = list(zip(*matched_prms.values()))
    else:
        tup_keys = ()
        tup_vals = [()]
    if len(matched_net_prms)>0: 
        if len(np.unique([len(v) for v in matched_net_prms.values()]))>1:
            raise ValueError('Tuple arguments must all be the same length you dingus!')
        tup_net_keys = tuple(matched_net_prms.keys())
        tup_net_vals = list(zip(*matched_net_prms.values()))
    else:
        tup_net_keys = ()
        tup_net_vals = [()]

    if len(matched_opt_prms)>0: 
        if len(np.unique([len(v) for v in matched_opt_prms.values()]))>1:
            raise ValueError('Tuple arguments must all be the same length you dingus!')
        tup_opt_keys = tuple(matched_opt_prms.keys())
        tup_opt_vals = list(zip(*matched_opt_prms.values()))
    else:
        tup_opt_keys = ()
        tup_opt_vals = [()]

    if len(variable_prms)>0:
        var_k, var_v = list(zip(*variable_prms.items()))
    else:
        var_k = ()
        var_v = ()
    if len(variable_net_prms)>0:
        var_n_k, var_n_v = list(zip(*variable_net_prms.items()))
    else:
        var_n_k = ()
        var_n_v = ()

    if len(variable_opt_prms)>0:
        var_o_k, var_o_v = list(zip(*variable_opt_prms.items()))
    else:
        var_o_k = ()
        var_o_v = ()
    ##############################


    exps = []

    all_vals = []
    for vals in list(itt.product(*var_v, tup_vals)):
        this_dset = dict(zip(var_k+tup_keys, vals[:-1]+vals[-1]), **fixed_prms)

        exp_info = {'experiment':exp_prm['experiment'], 'exp_args':this_dset}

        net_idx = 0
        for net_vals in list(itt.product(*var_n_v, tup_net_vals)):
            this_net = dict(zip(var_n_k+tup_net_keys, net_vals[:-1]+net_vals[-1]), **fixed_net_prms)

            net_info = {'model':net_args['model'], 'model_args':this_net}

            for opt_vals in list(itt.product(*var_o_v, tup_opt_vals)):
                this_opt = dict(zip(var_o_k+tup_opt_keys, opt_vals[:-1]+opt_vals[-1]), **fixed_opt_prms)

                exps.append({'exp_prm':exp_info, 'net_args':net_info, 'opt_args':this_opt})
                all_vals.append(vals[:-1]+vals[-1] + net_vals[:-1]+net_vals[-1] + opt_vals[:-1]+opt_vals[-1])
    
    all_vals = np.array(all_vals, dtype=object)
    params = {}
    for i, k in enumerate(var_k + tup_keys + var_n_k + tup_net_keys + var_o_k + tup_opt_keys):
        if bool_friendly:
            params[k] = np.array([stringify(v) for v in all_vals[:,i]])
        else:
            params[k] = np.array([v for v in all_vals[:,i]])

    return exps, params

def stringify(thing):
    """
    because python is doodoo caca  
    """

    whatisit = type(thing)

    if 'builtin' not in whatisit.__module__:
        if '__name__' in dir(thing):
            return thing.__name__
        else:
            return thing.__class__.__name__
    elif callable(thing):
        return thing.__name__
    elif whatisit in [int, float, str, bool]:
        return thing
