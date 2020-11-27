
task = util.RandomDichotomies(8,2,0)
# task = util.RandomDichotomiesCategorical(8,2,0)
# task = util.ParityMagnitudeEnumerated()
# task = util.Digits()
# task = util.DigitsBitwise()
# obs_dist = Bernoulli(1)
latent_dist = None
# latent_dist = GausId
nonlinearity = 'ReLU'
# nonlinearity = 'LeakyReLU'

num_layer = 0
# num_layer = 1

good_start = True
# good_start = False
# coding_level = 0.5
coding_level = None

rotation = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

decay = 0.0

H = 100

N = 100

random_decoder = None

# find experiments 
# this_exp = exp.mnist_multiclass(task, SAVE_DIR, 
#                                 z_prior=latent_dist,
#                                 num_layer=num_layer,
#                                 weight_decay=decay,
#                                 decoder=random_decoder,
#                                 good_start=good_start,
#                                 init_coding=coding_level)

nets = [[] for _ in rotation]
all_nets = [[] for _ in rotation]
all_args = [[] for _ in rotation]
mets = [[] for _ in rotation]
dicts = [[] for _ in rotation]
best_perf = []
for i,r in enumerate(rotation):
    this_exp = exp.random_patterns(task, SAVE_DIR, 
                                    num_class=8,
                                    dim=100,
                                    var_means=1,
                                    z_prior=latent_dist,
                                    num_layer=num_layer,
                                    weight_decay=decay,
                                    decoder=random_decoder,
                                    good_start=good_start,
                                    init_coding=coding_level,
                                    rot=r)
    
    this_folder = SAVE_DIR + this_exp.folder_hierarchy()
    
    files = os.listdir(this_folder)
    param_files = [f for f in files if ('parameters' in f and '_N%d_%s'%(N,nonlinearity) in f)]
    
    # j = 0
    num = len(param_files)
    all_metrics = {}
    best_net = None
    this_arg = None
    maxmin = 0
    for j,f in enumerate(param_files):
        rg = re.findall(r"init(\d+)?_N%d_%s"%(n,nonlinearity),f)
        if len(rg)>0:
            init = np.array(rg[0]).astype(int)
        else:
            init = None
            
        this_exp.use_model(N=n, init=init)
        model, metrics, args = this_exp.load_experiment(SAVE_DIR)
        
        if metrics['test_perf'][-1,...].min() > maxmin:    
            maxmin = metrics['test_perf'][-1,...].min()
            best_net = model
            this_arg = args
        
        for key, val in metrics.items():
            if key not in all_metrics.keys():
                shp = (num,) + np.squeeze(np.array(val)).shape
                all_metrics[key] = np.zeros(shp)*np.nan
                
            all_metrics[key][j,...] = np.squeeze(val)
    
            # if (val.shape[0]==1000) or not len(val):
                # continue
            # all_metrics[key][j,...] = val
        all_nets[i].append(model)
        all_args[i].append(args)
        
    nets[i] = best_net
    mets[i] = all_metrics
    dicts[i] = this_arg
    best_perf.append(maxmin)

#%%
netid = 8

# show_me = 'train_loss'
# show_me = 'train_perf' 
# show_me = 'test_perf'
show_me = 'test_PS'
# show_me = 'shattering'
# show_me = 'test_ccgp'
# show_me = 'mean_grad'
# show_me = 'std_grad'
# show_me = 'linear_dim'
# show_me = 'sparsity'

epochs = np.arange(1,mets[netid][show_me].shape[1]+1)

mean = np.nanmean(mets[netid][show_me],0)
error = (np.nanstd(mets[netid][show_me],0))#/np.sqrt(mets[netid][show_me].shape[0]))

if len(mean.shape)>1:
    for dim in range(mean.shape[-1]):
        pls = mean[...,dim]+error[...,dim]
        mns = mean[...,dim]-error[...,dim]
        plt.plot(epochs, mean[...,dim])
        plt.fill_between(epochs, mns, pls, alpha=0.5)
        plt.semilogx()
else:
    plt.plot(epochs, mean)
    plt.fill_between(epochs, mean-error, mean+error, alpha=0.5)
    plt.semilogx()

plt.xlabel('epoch', fontsize=15)
plt.ylabel(show_me, fontsize=15)
plt.title('N=%d'%N)


#%%
# show_me = 'train_loss'
# show_me = 'train_perf' 
# show_me = 'test_perf'
# show_me = 'test_PS'
# show_me = 'shattering'
show_me = 'test_ccgp'
# show_me = 'mean_grad'
# show_me = 'std_grad'
# show_me = 'PR'
# show_me = 'sparsity'


final_ps = []
final_ps_err = []
initial_ps = []
initial_ps_err = []
for i in range(len(mets)):
    final_ps.append(np.nanmean(mets[i][show_me][:,-1,:],0))
    final_ps_err.append(np.nanstd(mets[i][show_me][:,-1,:],0))
    
    initial_ps.append(np.nanmean(mets[i]['test_PS'][:,0,:],0))
    initial_ps_err.append(np.nanstd(mets[i]['test_PS'][:,0,:],0))

final_ps = np.array(final_ps)
final_ps_err = np.array(final_ps_err)

initial_ps = np.array(initial_ps)
initial_ps_err = np.array(initial_ps_err)


# plt.plot(output_type, final_ps[:,:2].mean(1))
# plt.fill_between(rotation, 
#                  final_ps[:,:2].mean(1)-final_ps_err[:,:2].mean(1),
#                  final_ps[:,:2].mean(1)+final_ps_err[:,:2].mean(1),
#                  alpha=0.5)

# plt.plot(output_type, final_ps[:,2:].mean(1))
# plt.fill_between(rotation, 
#                  final_ps[:,2:].mean(1)-final_ps_err[:,2:].mean(1),
#                  final_ps[:,2:].mean(1)+final_ps_err[:,2:].mean(1),
#                  alpha=0.5)

# plt.legend(['Trained Dichotomies','Untrained Dichotomies'])
# plt.ylabel('Final ' + show_me)
# plt.xlabel('Output "simpliciality"')

plt.plot(initial_ps[:,:2].mean(1), final_ps[:,:2].mean(1))
plt.fill_between(initial_ps[:,:2].mean(1), 
                  final_ps[:,:2].mean(1)-final_ps_err[:,:2].mean(1),
                  final_ps[:,:2].mean(1)+final_ps_err[:,:2].mean(1),
                  alpha=0.5)

plt.plot(initial_ps[:,:2].mean(1), final_ps[:,2:].mean(1))
plt.fill_between(initial_ps[:,:2].mean(1), 
                  final_ps[:,2:].mean(1)-final_ps_err[:,2:].mean(1),
                  final_ps[:,2:].mean(1)+final_ps_err[:,2:].mean(1),
                  alpha=0.5)

plt.legend(['Trained Dichotomies','Untrained Dichotomies'])
# plt.ylabel('Final ' + show_me)
# plt.xlabel('Output "simpliciality"')
