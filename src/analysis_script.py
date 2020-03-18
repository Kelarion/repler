

#%%
recon, _, _ = vae(digits.tensors[0][:10,:,:].reshape(-1,784).float()/252)
# recon = recon.reshape((-1,28,28))

pred, _, z = vae(digits.tensors[0].reshape(-1,784).float()/252)
pred = pred.detach().numpy()
z = z.detach().numpy()

#%% do the linear readout
clsfr = svm.LinearSVC # the classifier to use
cfargs = {'tol': 1e-5, 'max_iter':5000}

# train linear decoders
z = vae(stigid.tensors[0].reshape(-1,784).float()/252)[2].detach().numpy()
ans = stigid.tensors[1].detach().numpy()

clf = LinearDecoder(N, Q, clsfr)
clf.fit(z[:,:,None], ans[:,None,:], **cfargs)
    
coefs = clf.coefs
thrs = clf.thrs

# test
z = vae(digits.tensors[0].reshape(-1,784).float()/252)[2].detach().numpy()
ans = digits.tensors[1].detach().numpy()

perf = clf.test(z[:,:,None], ans[:,None,:])
marg = clf.margin(z[:,:,None], ans[:,None,:])
inner = np.einsum('ik...,jk...->ij...', coefs, coefs)
proj = clf.project(z[:,:,None])

plt.figure()
scat = plt.scatter(proj[0,:,0], proj[1,:,0], c=decimal(ans))
plt.xlabel('Parity classifier')
plt.ylabel('Magnitude classifier')
cb = plt.colorbar(scat, 
                  ticks=np.unique(decimal(ans)),
                  drawedges=True,
                  values=np.unique(decimal(ans)))
cb.set_ticklabels(np.unique(decimal(ans)))
cb.set_alpha(1)
cb.draw_all()

plt.figure()
plt.scatter(clf.coefs[0,:,0],clf.coefs[1,:,0])
lims = [min([plt.xlim()[0],plt.ylim()[0]]), max([plt.xlim()[1],plt.ylim()[1]])]
plt.gca().set_xlim(lims)
plt.gca().set_ylim(lims)
plt.gca().set_aspect('equal')
plt.plot(lims, lims, 'k--', alpha=0.2)
plt.plot(lims,[0,0], 'k-.', alpha=0.2)
plt.plot([0,0],lims, 'k-.', alpha=0.2)
plt.xlabel('Parity weight')
plt.ylabel('Magnitude weight')

#%% PCA
# cmap_name = 'nipy_spectral'
colorby = decimal(ans)

U, S, _ = la.svd(z-z.mean(1)[:,None], full_matrices=False)
pcs = z@U[:3,:].T

plt.figure()
plt.loglog((S**2)/np.sum(S**2))
plt.xlabel('PC')
plt.ylabel('variance explained') 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scat = ax.scatter(pcs[:,0],pcs[:,1],pcs[:,2], c=colorby, alpha=0.1)
ax.set_xlabel('pc1')
ax.set_ylabel('pc2')
ax.set_zlabel('pc3')

cb = plt.colorbar(scat, 
                  ticks=np.unique(colorby),
                  drawedges=True,
                  values=np.unique(colorby))
cb.set_ticklabels(np.unique(colorby))
cb.set_alpha(1)
cb.draw_all()

#%%
idx = 1

plt.subplot(1,2,1)
plt.imshow(digits.data[idx,...].detach().numpy())

plt.subplot(1,2,2)
plt.imshow(recon[idx,...].detach().numpy())


#%% 
wa = np.meshgrid(sts.norm.ppf(np.linspace(0.01,0.99,20)),sts.norm.ppf(np.linspace(0.01,0.99,20)))
z_q = np.append(wa[0].flatten()[:,None], wa[1].flatten()[:,None],axis=1)