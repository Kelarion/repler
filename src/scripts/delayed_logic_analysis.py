
import socket
import os
import sys

if socket.gethostname() == 'kelarion':
    if sys.platform == 'linux':
        CODE_DIR = '/home/kelarion/github/repler/src'
        SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
    else:
        CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
        SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
    openmind = False
elif socket.gethostname() == 'openmind7':
    CODE_DIR = '/home/malleman/repler/'
    SAVE_DIR = '/om2/user/malleman/abstraction/'
    openmind = True
else:    
    CODE_DIR = '/rigel/home/ma3811/repler/'
    SAVE_DIR = '/rigel/theory/users/ma3811/'
    openmind = False

sys.path.append(CODE_DIR)

import re
import pickle as pkl
import warnings

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import scipy.special as spc
import scipy.linalg as la
import scipy.special as spc
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation as anime
from itertools import permutations, combinations
from sklearn import svm, manifold, linear_model
from tqdm import tqdm

# this is my code base, this assumes that you can access it
import students
import assistants
import util
import recurrent
import experiments as exp
import plotting as dicplt

#%%

N = 50

empty_time = 20

# nonlinearity = 'ReLU'
nonlinearity = 'Tanh'

# which_task = 5
which_task = 5

inp_channels = 1

readout_weights = None
# readout_weights = students.BinaryReadout
# readout_weights = students.PositiveReadout

input_task = util.RandomDichotomies(d=[(0,1,2,3),(0,2,4,6),(0,1,4,5)])
output_task = util.LogicalFunctions(d=[(0,1,2,3),(0,2,4,6),(0,1,4,5)], function_class=which_task) # 3d xor

num_cond = 8

this_exp = exp.delayed_logic(input_channels=inp_channels,
                            task=output_task, 
                            input_task=input_task,
                            SAVE_DIR=SAVE_DIR,
                            time_between=empty_time,
                            nonlinearity=nonlinearity)

this_folder = SAVE_DIR + this_exp.folder_hierarchy()


files = os.listdir(this_folder)
param_files = [f for f in files if ('parameters' in f and '_N%d_%s'%(N,nonlinearity) in f)]

# j = 0
num = len(param_files)
all_metrics = {}
best_net = None
this_arg = None
maxmin = 0

all_nets = []
all_args = []
for j,f in enumerate(param_files):
    rg = re.findall(r"init(\d+)?_N%d_%s"%(N,nonlinearity),f)
    if len(rg)>0:
        init = np.array(rg[0]).astype(int)
    else:
        init = None
        
    this_exp.use_model(N=N, init=init)
    model, metrics, args = this_exp.load_experiment(SAVE_DIR)
    
    if metrics['test_perf'][-1,...].min() > maxmin:    
        maxmin = metrics['test_perf'][-1,...].min()
        best_net = model
        this_arg = args
    
    for key, val in metrics.items():
        if key not in all_metrics.keys():
            # shp = (num,) + np.squeeze(np.array(val)).shape
            # all_metrics[key] = np.zeros(shp)*np.nan
            all_metrics[key] = []
        
        # ugh = np.min([all_metrics[key][j,...].shape[0], np.squeeze(val).shape[0]])
        # all_metrics[key][j,:ugh,...] = np.squeeze(val)[:ugh,...]
        all_metrics[key].append(np.squeeze(val))

        # if (val.shape[0]==1000) or not len(val):
            # continue
        # all_metrics[key][j,...] = val
    all_nets.append(model)
    all_args.append(args)
    
net = best_net
mets = all_metrics
dicts = this_arg


#%%
new_exp = exp.delayed_logic(input_channels=inp_channels, 
                            jitter=False,
                            task=output_task, 
                            input_task=input_task,
                            SAVE_DIR=SAVE_DIR,
                            time_between=empty_time)

# new_inputs = torch.zeros(num_data, empty_time+3)
# new_inputs[np.repeat(range(num_data),3),input_times.flatten()] = 2*input_task(this_exp.train_conditions).flatten()-1

# new_inputs = torch.zeros(num_data, empty_time+3, 2)
# new_inputs[np.repeat(range(num_data),3),input_times.flatten(),np.tile([0,1,0],num_data)] = 2*input_task(this_exp.train_conditions).flatten()-1

new_inputs = new_exp.train_data[0]

# z_ = net.transparent_forward(new_inputs[:,:,None].transpose(0,1))[1].detach().numpy()

which_time = np.repeat(range(inputs.shape[1]), num_data) # which time
which_trial = np.tile(range(num_data), inputs.shape[1]) # which trial
winp = which_inp.T.flatten()

# these_times = winp==2 # in between 2nd and 3rd inputs
# these_times = t_==t_.max() # last timestep
# these_times = winp==3 # after 3rd input
# these_times = winp==1 # before 2nd input
# these_times = t_==1 # 1st hidden time
# these_times = t_==3 # 2nd hidden time
# these_times = t_==5 # 3rd hidden time

max_dichs = 50 # the maximum number of untrained dichotomies to test 
max_sample = 5000 # maximum number of data points

num_cond = len(np.unique(new_exp.train_conditions))

# this_task = input_task
this_task = output_task

all_PS = []
all_CCGP = []
CCGP_out_corr = []
mut_inf = []
all_SD = []
dic_pos = []
for net in all_nets:
    z_ = net.transparent_forward(new_inputs.transpose(0,1))[1].detach().numpy()
    
    seq_PS = []
    seq_CCGP = []
    seq_CCGP_out_corr = []
    seq_mut_inf = []
    seq_SD = []
    for t in tqdm(range(inputs.shape[1])):
        
        these_times = (which_time==t)
        
        z = z_.transpose((1,0,2)).reshape((N,-1))[:,these_times].T 
    
        stim_cond = np.tile(new_exp.train_conditions, inputs.shape[1])[these_times]
        
        N = z.shape[1]
        
        # z = this_exp.train_data[0].detach().numpy()
        # z = linreg.predict(this_exp.train_data[0])@W1.T
        n_compute = np.min([max_sample, z.shape[0]])
        
        idx = np.random.choice(z.shape[0], n_compute, replace=False)
        # idx_tst = idx[::4] # save 1/4 for test set
        # idx_trn = np.setdiff1d(idx, idx_tst)
        
        cond = stim_cond[idx]
        # cond = util.decimal(this_exp.train_data[1][idx,...])
        
        # xor = np.where(~(np.isin(range(num_cond), args['dichotomies'][0])^np.isin(range(num_cond), args['dichotomies'][1])))[0]
        ## Loop over dichotomies
        # D = assistants.Dichotomies(num_cond, args['dichotomies']+[xor], extra=50)
        
        # choose dichotomies to have a particular order
        Q = input_task.num_var
        D_fake = assistants.Dichotomies(num_cond, this_task.positives, extra=7000)
        mi = np.array([this_task.information(p) for p in D_fake])
        midx = np.append(range(Q),np.flip(np.argsort(mi[Q:]))+Q)
        # these_dics = args['dichotomies'] + [D_fake.combs[i] for i in midx]
        D = assistants.Dichotomies(num_cond, [D_fake.combs[i] for i in midx], extra=0)
        
        clf = assistants.LinearDecoder(N, 1, assistants.MeanClassifier)
        gclf = assistants.LinearDecoder(N, 1, svm.LinearSVC)
        dclf = assistants.LinearDecoder(N, D.ntot, svm.LinearSVC)
        # clf = LinearDecoder(this_exp.dim_input, 1, MeanClassifier)
        # gclf = LinearDecoder(this_exp.dim_input, 1, svm.LinearSVC)
        # dclf = LinearDecoder(this_exp.dim_input, D.ntot, svm.LinearSVC)
        
        # K = int(num_cond/2) - 1 # use all but one pairing
        # K = int(num_cond/4) # use half the pairings
        
        PS = np.zeros(D.ntot)
        CCGP = [] #np.zeros((D.ntot, 100))
        out_corr = []
        d = np.zeros((n_compute, D.ntot))
        pos_conds = []
        for i, pos in enumerate(D):
            pos_conds.append(pos)
            # print('Dichotomy %d...'%i)
            # parallelism
            PS[i] = D.parallelism(z[idx,:], cond, clf)
            
            # CCGP
            cntxt = D.get_uncorrelated(100)
            out_corr.append(np.array([[(2*np.isin(p,c)-1).mean() for c in cntxt] for p in this_task.positives]))
            
            CCGP.append(D.CCGP(z[idx,:], cond, gclf, cntxt, twosided=True))
            
            # shattering
            d[:,i] = D.coloring(cond)
            
        # dclf.fit(z[idx_trn,:], d[np.isin(idx, idx_trn),:], tol=1e-5, max_iter=5000)
        dclf.fit(z[idx,:], d, tol=1e-5)
        
        idx2 = np.random.choice(z.shape[0], n_compute, replace=False)
        
        d_tst = np.array([D.coloring(stim_cond[idx2]) for _ in D]).T
        SD = dclf.test(z[idx2,:], d_tst).squeeze()
        
        seq_PS.append(PS)
        seq_CCGP.append(CCGP)
        seq_CCGP_out_corr.append(out_corr)
        seq_SD.append(SD)
        seq_mut_inf.append(mi[midx])
    
    all_PS.append(seq_PS)
    all_CCGP.append(seq_CCGP)
    CCGP_out_corr.append(seq_CCGP_out_corr)
    all_SD.append(seq_SD)
    mut_inf.append(seq_mut_inf)

R = np.repeat(np.array(CCGP_out_corr),2,-1)
basis_dependence = np.array(indep).max(1)
out_MI = np.array(mut_inf)

#%%

pos_conds = [tuple(p.tolist()) for p in all_metrics['pos_conds'][0]]

input_times = np.where(new_inputs[0]!=0)[0]


PS = np.mean(all_metrics['PS'], 0)
SD = np.mean(all_metrics['shattering'], 0)
almost_all_CCGP = np.mean(all_metrics['CCGP'], 0)

output_dics = []
for d in output_task.positives:
    output_dics.append(np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(num_cond),p))==list(d))\
                                for p in pos_conds])[0][0])
input_dics = []
for d in input_task.positives:
    input_dics.append(np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(num_cond),p))==list(d))\
              for p in pos_conds])[0][0])

inps = input_task(np.arange(8))
ctxt = tuple(np.argwhere(~((inps[:,0]==0)^(inps[:,1]==0)))[0].tolist())

rest = np.setdiff1d(range(len(pos_conds)), input_dics + output_dics + [pos_conds.index(ctxt)])

plt.figure()

plt.subplot(1,3,1)
plt.plot(PS[:,np.array(input_dics)], '-', linewidth=2, markersize=10, zorder=10)
plt.plot(PS[:,np.array(output_dics)], '--', linewidth=2, markersize=10, zorder=9)
plt.plot(PS[:,pos_conds.index(ctxt)],'-.', linewidth=2, markersize=10, zorder=8)
plt.plot(PS[:,rest], color=(0.7,0.7,0.7), zorder=3)
plt.xlabel('Time step')
plt.title('Parallelism')
plt.ylim(plt.ylim())
plt.plot([input_times[0],input_times[0]], plt.ylim(),'k--', zorder=0)
plt.plot([input_times[1],input_times[1]], plt.ylim(),'k--', zorder=0)
plt.plot([input_times[2],input_times[2]], plt.ylim(),'k--', zorder=0)

plt.subplot(1,3,2)
plt.plot(almost_all_CCGP[:,np.array(input_dics)], linewidth=2, zorder=9) 
plt.plot(almost_all_CCGP[:,np.array(output_dics)], '--', linewidth=2, zorder=8)
plt.plot(almost_all_CCGP[:,pos_conds.index(ctxt)], '-.', linewidth=2,  zorder=10)
plt.plot(almost_all_CCGP[:,rest], color=(0.7,0.7,0.7), zorder=3)
plt.xlabel('Time step')
plt.title('CCGP')
plt.ylim(plt.ylim())
plt.plot([input_times[0],input_times[0]], plt.ylim(),'k--', zorder=0)
plt.plot([input_times[1],input_times[1]], plt.ylim(),'k--', zorder=0)
plt.plot([input_times[2],input_times[2]], plt.ylim(),'k--', zorder=0)

plt.subplot(1,3,3)
plt.plot(SD[:,np.array(input_dics)], linewidth=2, zorder=9)
plt.plot(SD[:,np.array(output_dics)],'--', linewidth=2, zorder=8)
plt.plot(SD[:,pos_conds.index(ctxt)],'-.', linewidth=2, zorder=10)
plt.plot(SD[:,rest], color=(0.7,0.7,0.7), zorder=3)
plt.xlabel('Time step')
plt.title('Shattering')
plt.ylim(plt.ylim())
plt.plot([input_times[0],input_times[0]], plt.ylim(),'k--', zorder=0)
plt.plot([input_times[1],input_times[1]], plt.ylim(),'k--', zorder=0)
plt.plot([input_times[2],input_times[2]], plt.ylim(),'k--', zorder=0)


#%%
n_compute = 10000
lag = 3

# this_exp.load_other_info(args)
# this_exp.load_data(SAVE_DIR)

# fake_task = util.RandomDichotomies(num_cond,num_var,0)
# fake_task.positives = this_exp.task.positives

input_times = np.zeros((num_data,3))
input_times[:,0] = 0 
input_times[:,1] = 20
input_times[:,2] = 40

# new_inputs = torch.zeros(num_data, empty_time+3)
# new_inputs[np.repeat(range(num_data),3),input_times.flatten()] = 2*input_task(this_exp.train_conditions).flatten()-1

new_inputs = torch.zeros(num_data, empty_time+3, 2)
new_inputs[np.repeat(range(num_data),3),input_times.flatten(),np.tile([0,1,0],num_data)] = 2*input_task(this_exp.train_conditions).flatten()-1

# z_ = net.transparent_forward(new_inputs[:,:,None].transpose(0,1))[1].detach().numpy()
z_ = net.transparent_forward(new_inputs.transpose(0,1))[1].detach().numpy()

which_time = np.repeat(range(inputs.shape[1]), num_data) # which time
which_trial = np.tile(range(num_data), inputs.shape[1]) # which trial
winp = which_inp.T.flatten()

stim_cond = np.tile(this_exp.train_conditions, inputs.shape[1])

colorby = np.isin(np.unique(cond), this_task.positives)

z = z_.transpose((1,0,2)).reshape((N,-1)).T 
idx_svd = np.random.choice(z.shape[0], n_compute, replace=False)

avg = np.stack([[z[(stim_cond==i)&(which_time==t),:].mean(0) \
                 for i in np.unique(stim_cond)] for t in np.unique(which_time)]).reshape((-1,N))


# U, S, _ = la.svd(z[idx_svd,:].T-z[idx_svd,:].mean(0, keepdims=True).T, full_matrices=False)
U, S, _ = la.svd(avg.T-avg.mean(0, keepdims=True).T, full_matrices=False)

emb = z@U[:,:3]

avg = np.stack([[emb[(stim_cond==i)&(which_time==t),:].mean(0) \
                 for i in np.unique(stim_cond)] for t in np.unique(which_time)])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot(111)
plt.margins(0)
scat = ax.scatter(avg[0,:,0],avg[0,:,1],avg[0,:,2],s=50, c=colorby, cmap='bwr')
# scat = ax.scatter(avg[0,:,1],avg[0,:,2],s=50, c=colorby, cmap='bwr')
# scat2 = ax.scatter(avg[:0,:,0].flatten(),
#                    avg[:0,:,1].flatten(),
#                    avg[:0,:,2].flatten(),s=30, alpha=0.4, 
#                    c=np.repeat(colorby,1), 
#                    cmap='bwr')
# scat = ax.scatter(emb[:,0],emb[:,1], emb[:,2], c=colorby

ax.set_xlim3d([avg[...,0].min(), avg[...,0].max()])
ax.set_ylim3d([avg[...,1].min(), avg[...,1].max()])
ax.set_zlim3d([avg[...,2].min(), avg[...,2].max()])

# # ax.set_xlim3d([avg[...,0].min(), avg[...,0].max()])
# ax.set_xlim([avg[...,1].min(), avg[...,1].max()])
# ax.set_ylim([avg[...,2].min(), avg[...,2].max()])

ax.set_title("Abstract: %s"%str(pos_conds[PS[0,:].argmax()]))
# util.set_axes_equal(ax)

def init():
    
    ax.scatter(avg[0,:,0],avg[0,:,1],avg[0,:,2],s=50, marker='*', c=colorby, cmap='bwr')
    # ax.scatter(avg[0,:,1],avg[0,:,2],s=50, marker='*', c=colorby, cmap='bwr')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    
    # util.set_axes_equal(ax)
    
    # plt.xticks([])
    # plt.yticks([])
    # plt.zticks([])
    # plt.legend(np.unique(cond), np.unique(cond))
    # cb = plt.colorbar(scat,
    #                   ticks=np.unique(colorby),
    #                   drawedges=True,
    #                   values=np.unique(colorby))
    # cb.set_ticklabels(np.unique(colorby)+1)
    # cb.set_alpha(1)
    # cb.draw_all()
    
    return fig,

# def init():
    # ax.view_init(30,0)
    # plt.draw()
    # return ax,

def update(frame):

    scat._offsets3d = (avg[frame,:,0],avg[frame,:,1],avg[frame,:,2])
    # scat.set_offsets(avg[frame,:,[1,2]].T)
    
    cols = np.isin(np.unique(cond), pos_conds[PS[frame,:].argmax()])
    
    scat._facecolor3d = cm.bwr(cols/cols.max())
    # scat._facecolor = cm.bwr(cols/cols.max())
    ax.title.set_text(str(pos_conds[PS[frame,:].argmax()]))
    # scat2._offsets3d
    # util.set_axes_equal(ax)
    # plt.draw()
    return fig,

ani = anime.FuncAnimation(fig, update, frames=new_inputs.shape[-1],
                          init_func=init, interval=1, blit=True)
# plt.show()
ani.save(SAVE_DIR+'/vidya/tempmovie.mp4', writer=anime.writers['ffmpeg'](fps=20))
