CODE_DIR = '/home/matteo/Documents/github/repler/src/'
SAVE_DIR = '/home/matteo/Documents/uni/columbia/bleilearning/'

import os, sys
import pickle
sys.path.append(CODE_DIR)

import torch
import torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gsp
import matplotlib.colors as mpc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation as anime
from cycler import cycler
import scipy
import scipy.linalg as la
from scipy.spatial.distance import pdist, squareform
from itertools import permutations, combinations
import itertools as itt

import students
import assistants
import util

#%%

class AnimeCollection(object):
    def __init__(self, *animes):
        self.animes = list(animes)

        self.fig = self.animes[0].fig

        self.ani = anime.FuncAnimation(self.fig, self.update, interval=self.animes[0].interval, 
            init_func=self.setup_plot, blit=False, frames=self.animes[0].frames)

    def setup_plot(self):

        self.patches = []
        for an in self.animes:
            self.patches.append(an.setup_plot())

        return self.patches

    def update(self, i):
        for an in self.animes:
            an.update(i)

        return self.patches

    def save(self,save_dir, fps=30):
        self.ani.save(save_dir, writer=anime.writers['ffmpeg'](fps=fps))


class ScatterAnime(object):
    """ 
    Makes a scatter plot, and animates it to update along a particular axis 

    i hate matplotlib animation so much 
    """
    def __init__(self, x, y, auto_zoom=False, trail=False, interval=10, **scat_args):
        ''' 
        x and y are shape (num_data, num_time) 
        '''

        self.x = x
        self.y = y
        self.auto_zoom = auto_zoom
        self.leave_trail = trail
        self.kwargs = scat_args

        self.frames = x.shape[-1]
        self.interval = interval
        
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        # self.ani = anime.FuncAnimation(self.fig, self.update, interval=interval, 
        #     init_func=self.setup_plot, blit=False, frames=x.shape[-1])

    def setup_plot(self):
        """Initial drawing of the scatter plot."""

        self.patches = []

        # self.ax.plot(self.x,self.y)
        self.scat = self.ax.scatter(self.x[...,0], self.y[...,0], **self.kwargs)

        self.patches += [self.scat,]
        if not self.auto_zoom:
            self.ax.set_xlim([1.1*self.x.min(), 1.1*self.x.max()])
            self.ax.set_ylim([1.1*self.y.min(), 1.1*self.y.max()])

        if self.leave_trail:
            self.trail, = self.ax.plot(self.x[...,0], self.y[...,0])
            self.patches += [self.trail,]
    
        return self.patches


    def update(self, i):
        """Update the scatter plot."""

        # Set x and y data...
        self.scat.set_offsets([self.x[...,i], self.y[...,i]])

        if self.auto_zoom and i>0:
            # self.ax.set_xlim([1.1*self.x[...,:i+1].min(), 1.1*self.x[...,:i+1].max()])
            # self.ax.set_ylim([1.1*self.y[...,:i+1].min(), 1.1*self.y[...,:i+1].max()])
            plt.xlim([1.1*self.x[...,:i+1].min(), 1.1*self.x[...,:i+1].max()])
            plt.ylim([1.1*self.y[...,:i+1].min(), 1.1*self.y[...,:i+1].max()])

        if self.leave_trail and i>0:
            self.trail.set_data([self.x[...,:i].flatten(), self.y[...,:i].flatten()])

        return self.patches

class LineAnime(object):
    """ 
    Makes a (multiple) line plot(s), and animates 

    i hate matplotlib animation so much 
    """
    def __init__(self, x, y, auto_zoom=False, interval=10, colors=None,
        fig=None, ax=None, **ln_args):
        ''' 
        x and y are shape (num_lines, num_data, num_time) 
        '''

        self.x = x
        self.y = y
        self.kwargs = ln_args
        self.auto_zoom = auto_zoom

        if fig is None and ax is None:
            self.fig, self.ax = plt.subplots()
        elif ax is not None:
            self.ax = ax
            self.fig = ax.get_figure()
        elif fig is not None:
            self.fig = fig
            self.ax = fig.add_subplot(111)

        if colors is None:
            self.colors = [(0.5,0.5,0.5),]*len(x)
        else:
            self.colors = colors

        self.frames = x.shape[-1]
        self.interval = interval
        
        # Then setup FuncAnimation.
        # self.ani = anime.FuncAnimation(self.fig, self.update, interval=interval, 
        #     init_func=self.setup_plot, blit=False, frames=x.shape[-1])

    def setup_plot(self):
        """Initial drawing of the scatter plot."""

        self.patches = []

        # self.ax.plot(self.x,self.y)
        self.lines = []
        for i in range(len(self.x)):
            ln, = self.ax.plot(self.x[i,...,:1].T, self.y[i,...,:1].T,
                color=self.colors[i], **self.kwargs)
            self.lines.append(ln)

        self.patches += self.lines
        if self.auto_zoom:
            self.ax.set_xlim([1.1*self.x.min(), 1.1*self.x.max()])
            self.ax.set_ylim([1.1*self.y.min(), 1.1*self.y.max()])
    
        return self.patches

    def save(self,save_dir, fps=30):
        self.ani.save(save_dir, writer=anime.writers['ffmpeg'](fps=fps))

    def update(self, i):
        """Update the scatter plot."""

        # Set x and y data...
        for j in range(len(self.lines)):
            self.lines[j].set_data([self.x[j,...,:i].T, self.y[j,...,:i].T])

        if self.auto_zoom and i>0:
            # self.ax.set_xlim([1.1*self.x[...,:i+1].min(), 1.1*self.x[...,:i+1].max()])
            # self.ax.set_ylim([1.1*self.y[...,:i+1].min(), 1.1*self.y[...,:i+1].max()])
            plt.xlim([1.1*self.x[...,:i+1].min(), 1.1*self.x[...,:i+1].max()])
            plt.ylim([1.1*self.y[...,:i+1].min(), 1.1*self.y[...,:i+1].max()])

        return self.patches

class LineAnime3D(object):
    """ 
    Makes a (multiple) line plot(s), and animates 

    i hate matplotlib animation so much 
    """
    def __init__(self, x, y, z, auto_zoom=False, interval=10, 
        rotation_period=1000, colors=None, dot=False, scat_colors=None,
        focus_on=None, focus_period=100,
        frames=None, blit=False, view_period=0,
        init_elev=30, init_azim=30, rot_elev=0, rot_azim=1,
        fig=None, ax=None,
        **kwargs):
        ''' 
        x and y are shape (num_lines, num_data, num_time) 
        '''

        self.x = x
        self.y = y
        self.z = z
        self.kwargs = kwargs
        self.auto_zoom = auto_zoom
        self.view_period = view_period
        
        self.rot_speed = rotation_period
        self.init_elev = init_elev # initial elevation
        self.init_azim = init_azim # initial azimuth
        self.rot_e = rot_elev # how much to rotat3 elevation
        self.rot_a = rot_azim # how much to rotate azimuth

        if scat_colors is None:
            self.scat_colors = colors
        else:
            self.scat_colors = scat_colors

        if colors is None:
            self.colors = [(0.5,0.5,0.5),]*len(x)
        else:
            self.colors = colors

        self.do_scat = dot
        
        if focus_on is None:
            self.focus = False
        else:
            self.focus = True
            self.focus_on = focus_on
            self.focus_period = focus_period

        if frames is None:
            frames = x.shape[-1] + 2*view_period

        # Setup the figure and axes...
        if fig is None and ax is None:
            self.fig = plt.figure()
            self.ax = plt.subplot(111, projection='3d')
        elif ax is not None:
            self.ax = ax
            self.fig = ax.get_figure()
        elif fig is not None:
            self.fig = fig
            self.ax = fig.add_subplot(111, projection='3d')
            
        # Then setup FuncAnimation.
        self.ani = anime.FuncAnimation(self.fig, self.update, interval=interval, 
            init_func=self.setup_plot, blit=blit, frames=frames)

    def save(self,save_dir, fps=30):
        self.ani.save(save_dir, writer=anime.writers['ffmpeg'](fps=fps))

    def setup_plot(self):
        """Initial drawing of the scatter plot."""

        self.patches = []

        if self.focus:
            idx = self.focus_on
        else:
            idx = np.arange(len(self.x))

        # self.ax.plot(self.x,self.y)
        self.lines = []
        for i in idx:
            ln, = self.ax.plot3D(self.x[i,...,:1].T, self.y[i,...,:1].T, self.z[i,...,:1].T, 
                color=self.colors[i], **self.kwargs)
            self.lines.append(ln)

        self.patches += self.lines

        if self.do_scat:
            self.scat = self.ax.scatter(self.x[idx,...,0], self.y[idx,...,0], self.z[idx,...,0], 
                edgecolors=self.scat_colors, facecolors=self.scat_colors)
            self.patches += [self.scat,]

        if not self.auto_zoom:
            self.ax.set_xlim3d([1.1*self.x[idx].min(), 1.1*self.x[idx].max()])
            self.ax.set_ylim3d([1.1*self.y[idx].min(), 1.1*self.y[idx].max()])
            self.ax.set_zlim3d([1.1*self.z[idx].min(), 1.1*self.z[idx].max()])
        else:
            self.ax.set_xlim3d([1.1*self.x[idx,...,0].min(), 1.1*self.x[idx,...,0].max()])
            self.ax.set_ylim3d([1.1*self.y[idx,...,0].min(), 1.1*self.y[idx,...,0].max()])
            self.ax.set_zlim3d([1.1*self.z[idx,...,0].min(), 1.1*self.z[idx,...,0].max()])


        self.ax.view_init(self.init_elev, self.init_azim)
    
        return self.patches

    def update(self, i):
        """Update the scatter plot."""

        self.ax.view_init(self.init_elev + self.rot_e*(i/self.rot_speed)*360,
            self.init_azim + self.rot_a*(i/self.rot_speed)*360)

        if i>self.view_period:
            i -= self.view_period
            # Set x and y data...
            for j in range(len(self.lines)):
                # self.lines[j].set_data_3d(self.x[j,...,:i].T, self.y[j,...,:i].T, self.z[j,...,:i].T)
                self.lines[j].set_data(self.x[j,...,:i], self.y[j,...,:i])
                self.lines[j].set_3d_properties(self.z[j,...,:i])

            if self.do_scat:
                # self.scat.set_offsets([self.x[...,i], self.y[...,i]])
                # self.scat.set_3d_properties(self.z[...,i], 'z')
                self.scat._offsets3d = (self.x[...,i], self.y[...,i],self.z[...,i])

            if self.auto_zoom and i>0:
                # self.ax.set_xlim([1.1*self.x[...,:i+1].min(), 1.1*self.x[...,:i+1].max()])
                # self.ax.set_ylim([1.1*self.y[...,:i+1].min(), 1.1*self.y[...,:i+1].max()])
                self.ax.set_xlim3d([1.1*self.x[...,:i+1].min(), 1.1*self.x[...,:i+1].max()])
                self.ax.set_ylim3d([1.1*self.y[...,:i+1].min(), 1.1*self.y[...,:i+1].max()])
                self.ax.set_zlim3d([1.1*np.min(self.z[...,:i+1]), 1.1*self.z[...,:i+1].max()])

        # self.ax.view_init(30,(i/self.rot_speed)*360)

        return self.patches


class ScatterAnime3D(object):
    """ 
    Makes a (multiple) line plot(s), and animates 

    i hate matplotlib animation so much 
    """
    def __init__(self, x, y, z, auto_zoom=False, interval=10, 
        rotation_period=1000, frames=None, blit=False, view_period=0,
        init_elev=30, init_azim=30, rot_elev=0, rot_azim=1,
        fig=None, ax=None, equalize=False,
        **kwargs):
        ''' 
        x and y are shape (num_lines, num_data, num_time) 
        '''

        self.x = x
        self.y = y
        self.z = z
        self.kwargs = kwargs
        self.auto_zoom = auto_zoom
        self.view_period = view_period
        self.equalize = equalize
        
        self.rot_speed = rotation_period
        self.init_elev = init_elev # initial elevation
        self.init_azim = init_azim # initial azimuth
        self.rot_e = rot_elev # how much to rotat3 elevation
        self.rot_a = rot_azim # how much to rotate azimuth

        if frames is None:
            frames = x.shape[-1] + view_period

        # Setup the figure and axes...
        if fig is None and ax is None:
            self.fig = plt.figure()
            self.ax = plt.subplot(111, projection='3d')
        elif ax is not None:
            self.ax = ax
            self.fig = ax.get_figure()
        elif fig is not None:
            self.fig = fig
            self.ax = fig.add_subplot(111, projection='3d')
            
        # Then setup FuncAnimation.
        self.ani = anime.FuncAnimation(self.fig, self.update, interval=interval, 
            init_func=self.setup_plot, blit=blit, frames=frames)

    def save(self,save_dir, fps=30):
        self.ani.save(save_dir, writer=anime.writers['ffmpeg'](fps=fps))

    def setup_plot(self):
        """Initial drawing of the scatter plot."""

        self.patches = []

        idx = np.arange(len(self.x))

        # self.ax.plot(self.x,self.y)
        self.scat = self.ax.scatter(self.x[...,0].T, self.y[...,0].T, self.z[...,0].T, 
             **self.kwargs)

        self.patches += [self.scat,]

        if not self.auto_zoom:
            self.ax.set_xlim3d([1.1*self.x[idx].min(), 1.1*self.x[idx].max()])
            self.ax.set_ylim3d([1.1*self.y[idx].min(), 1.1*self.y[idx].max()])
            self.ax.set_zlim3d([1.1*self.z[idx].min(), 1.1*self.z[idx].max()])
        else:
            self.ax.set_xlim3d([1.1*self.x[idx,...,0].min(), 1.1*self.x[idx,...,0].max()])
            self.ax.set_ylim3d([1.1*self.y[idx,...,0].min(), 1.1*self.y[idx,...,0].max()])
            self.ax.set_zlim3d([1.1*self.z[idx,...,0].min(), 1.1*self.z[idx,...,0].max()])

        if self.equalize:
            util.set_axes_equal(self.ax)

        self.ax.view_init(self.init_elev, self.init_azim)
    
        return self.patches

    def update(self, i):
        """Update the scatter plot."""

        self.ax.view_init(self.init_elev + self.rot_e*(i/self.rot_speed)*360,
            self.init_azim + self.rot_a*(i/self.rot_speed)*360)

        if i>self.view_period:
            i -= self.view_period
            # Set x and y data...
    
                # self.lines[j].set_data_3d(self.x[j,...,:i].T, self.y[j,...,:i].T, self.z[j,...,:i].T)
            self.scat._offsets3d = (self.x[...,i], self.y[...,i],self.z[...,i])
            # self.scat.set_color(self.color[i])

            if self.auto_zoom and i>0:
                # self.ax.set_xlim([1.1*self.x[...,:i+1].min(), 1.1*self.x[...,:i+1].max()])
                # self.ax.set_ylim([1.1*self.y[...,:i+1].min(), 1.1*self.y[...,:i+1].max()])
                self.ax.set_xlim3d([1.1*self.x[...,:i+1].min(), 1.1*self.x[...,:i+1].max()])
                self.ax.set_ylim3d([1.1*self.y[...,:i+1].min(), 1.1*self.y[...,:i+1].max()])
                self.ax.set_zlim3d([1.1*np.min(self.z[...,:i+1]), 1.1*self.z[...,:i+1].max()])

        # self.ax.view_init(30,(i/self.rot_speed)*360)

        return self.patches

#%%
class PCA3D(object):
    def __init__(self, X, special_vec=None, ax=None, **scat_args):
        ''' assume X is shape (dim_data, num_data) '''

        if special_vec is None:
            X_ = X
        else:
            X_ = X - (special_vec[:,None]@special_vec[None,:]/la.norm(special_vec))@X
            y_ = (special_vec[None,:]/la.norm(special_vec))@X

        U, S, _ = la.svd(X-X.mean(1)[:,None], full_matrices=False)
        pcs = X.T@U[:,:3]

        if special_vec is None:
            self.loadings = U[:,:3]
        else:
            self.loadings = np.concatenate([special_vec[:,None], U[:,:2]])

        if ax is None:
            self.ax = plt.subplot(111, projection='3d')
        else:
            self.ax = ax
        if special_vec is None:
            self.scat = self.ax.scatter(pcs[:,0],pcs[:,1],pcs[:,2], **scat_args)
        else:
            self.scat = self.ax.scatter(y_,pcs[:,0],pcs[:,1], **scat_args)

    def overlay(self, new_X, **scat_args):

        new_pcs = new_X.T@self.loadings
        self.ax.scatter(new_pcs[:,0],new_pcs[:,1],new_pcs[:,2], **scat_args)

#%%
class SliceViewer(object):
    def __init__(self, ax=None):
        if ax is not None:
            self.ax = ax
        else:
            self.ax = plt.axes()

        if hasattr(self.ax,'claimed'):
            if self.ax.claimed:
                self.is_vassal = True
                self.leige = self.ax.leige
            else:
                self.ax.claimed = True
                self.is_vassal = False
                self.ax.leige = self
        else:
            self.ax.claimed = True
            self.is_vassal = False
            self.ax.leige = self

    def on_scroll(self,event):
        if self.is_vassal:
            self.ind = self.leige.ind
        else:
            if event.button == 'up':
                self.ind = (self.ind + 1) % self.slices
            else:
                self.ind = (self.ind - 1) % self.slices
        self.update() 

class ImageSlices(SliceViewer):
    def __init__(self, X, ax=None, **im_args):

        super(ImageSlices,self).__init__(ax)

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = self.ax.imshow(self.X[:, :, self.ind], 
            norm=mpc.Normalize(self.X.min(), self.X.max()), **im_args)
        self.update()

        self.ax.get_figure().canvas.mpl_connect('scroll_event', self.on_scroll)

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

class LineSlices(SliceViewer):
    def __init__(self, X, Y, ax=None, slice_width=0.1, scroll_mag=0.02, num_slice=None, slice_idx=None, **ln_args):
        '''
        Plots X, only showing the parts within a certain range of Y -- scroll to change Y
        '''
        super(LineSlices,self).__init__(ax)

        self.X = X
        self.Y = Y 
        self.slice_width = slice_width

        if slice_idx is None:
            if num_slice is None:
                self.scroll_mag = scroll_mag
            else:
                self.scroll_mag = (Y.max()-Y.min())/(num_slice-1)

            self.slice_idx = np.arange(Y.min(), Y.max()+self.scroll_mag, self.scroll_mag)
        else:
            self.slice_idx = slice_idx
        self.slices = len(self.slice_idx)

        self.ind = self.slices//2
        # self.idx = (self.Y < self.slice_idx[self.ind]+self.slice_width) \
        # & (self.Y >= self.slice_idx[self.ind]-self.slice_width)
        # self.mask_x1 = np.where(self.idx, self.X[...,0], np.nan)
        # self.mask_x2 = np.where(self.idx, self.X[...,1], np.nan)

        self.all_idx = [(self.Y < s+self.slice_width) & (self.Y >= s-self.slice_width) for s in self.slice_idx]
        self.mask_X1 = np.where(self.all_idx, self.X[...,0], np.nan)
        self.mask_X2 = np.where(self.all_idx, self.X[...,1], np.nan)

        self.ln = self.ax.plot(self.mask_X1[self.ind].T, self.mask_X2[self.ind].T, **ln_args)
        print(len(self.ln))
        self.update()

        self.ax.get_figure().canvas.mpl_connect('scroll_event', self.on_scroll)

    def update(self):
        # self.idx = (self.Y < self.slice_idx[self.ind]+self.slice_width) \
        # & (self.Y >= self.slice_idx[self.ind]-self.slice_width)
        # self.mask_x1 = np.where(self.idx,self.X[...,0], np.nan)
        # self.mask_x2 = np.where(self.idx, self.X[...,1], np.nan)
        # print(self.idx)
        # print(self.mask)

        for ln in self.ln:
            ln.set_data([self.mask_X1[self.ind], self.mask_X2[self.ind]])
        self.ax.set_ylabel('slice for y= %.1f' % self.slice_idx[self.ind])
        plt.draw()

class ScatterSlices(SliceViewer):
    def __init__(self, X, Y, ax=None, slice_width=0.1, scroll_mag=0.02,num_slice=None,slice_idx=None, 
        color=None, **scat_args):
        '''
        Plots X, only showing the parts within a certain range of Y -- scroll to change Y
        '''
        super(ScatterSlices,self).__init__(ax)

        if color is None:
            self.col = [(0.5,0.5,0.5),]*len(Y)
        else:
            self.col = color

        self.X = X
        self.Y = Y 
        self.slice_width = slice_width
        if slice_idx is None:
            if num_slice is None:
                self.scroll_mag = scroll_mag
            else:
                self.scroll_mag = (Y.max()-Y.min())/(num_slice-1)

            self.slice_idx = np.arange(Y.min(), Y.max()+self.scroll_mag, self.scroll_mag)
        else:
            self.slice_idx = slice_idx
        self.slices = len(self.slice_idx)

        self.ind = self.slices//2
        self.idx = (self.Y < self.slice_idx[self.ind]+self.slice_width) \
        & (self.Y >= self.slice_idx[self.ind]-self.slice_width)

        self.scat = self.ax.scatter(self.X[self.idx, 0], self.X[self.idx, 1], 
            facecolors=np.array(self.col)[self.idx], **scat_args)
        self.update()

        self.ax.get_figure().canvas.mpl_connect('scroll_event', self.on_scroll)

    def update(self):

        self.idx = (self.Y < self.slice_idx[self.ind]+self.slice_width) \
        & (self.Y >= self.slice_idx[self.ind]-self.slice_width)

        self.scat.set_offsets(self.X[self.idx, :])
        self.scat.set_color(np.array(self.col)[self.idx])
        self.ax.set_ylabel('slice for y= %.3f' % self.slice_idx[self.ind])
        plt.draw()

class QuiverSlices(SliceViewer):
    def __init__(self, X, V, ax=None, **quiv_args):

        super(QuiverSlices,self).__init__()

        if ax is not None:
            self.ax = ax
        else:
            self.ax = plt.axes()

        self.X = X
        self.V = V
        self.slices = V.shape[0]
        self.ind = self.slices//2

        self.quiv = self.ax.quiver(self.X[:,0], self.X[:,1], 
            self.V[self.ind,:,0],  self.V[self.ind,:,1], **quiv_args)
        self.update()

        self.ax.get_figure().canvas.mpl_connect('scroll_event', self.on_scroll)

    def update(self):
        self.quiv.set_UVC(self.V[self.ind, :, 0], self.V[self.ind, :, 1])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.quiv.axes.figure.canvas.draw()

#%%
def hierarchical_labels(row_labels, col_labels, row_names=None, col_names=None, hmarg=0.1,wmarg=0.1, **text_args):
    """
    
    """

    if row_names is None:
        label_rows = False
        row_names = np.arange(len(row_labels))
    else:
        label_rows = True
    if col_names is None:
        label_cols = False
        col_names = np.arange(len(col_labels))
    else:
        label_cols = True

    n_row = len(list(itt.product(*row_labels)))
    n_col = len(list(itt.product(*col_labels)))

    fig, axes = plt.subplots(n_row,n_col)
    axes = np.reshape(axes, (n_row, n_col))

    w_rat = [0.025,]*len(row_names) + [(1-0.025*len(row_names))/n_col,]*n_col
    h_rat = [0.025,]*len(col_names) + [(1-0.025*len(col_names))/n_row,]*n_row

    gs = gsp.GridSpec(n_row+len(col_names), n_col+len(row_names), 
                      width_ratios=w_rat, height_ratios=h_rat)
    gs.update(left=0.1, right=0.9, bottom=0.15, wspace=wmarg, hspace=hmarg)
    fig.subplots_adjust(left=.1)

    for i in range(n_row):
        for j in range(n_col):
            # if n_row == 1:
            #     axes[j].set_subplotspec(gs[i+len(col_names),j+len(row_names)])
            # elif n_col == 1:
            #     axes[i].set_subplotspec(gs[i+len(col_names),j+len(row_names)])
            # else:
            axes[i,j].set_subplotspec(gs[i+len(col_names),j+len(row_names)])

    row_lab_axs = []
    num_lab = n_row
    for i,lab in enumerate(row_names):
        num_lab = num_lab//len(row_labels[i])
        # gs_idx = np.arange(n_row)//num_lab
        
        tmp_axs = []
        for j in range(n_row//num_lab):
            idx0 = num_lab*j + len(col_names)
            idx1 = num_lab*(j+1) + len(col_names)
            tmp_axs.append(fig.add_subplot(gs[idx0:idx1,i]))
        
        lab_idx = np.mod(np.arange(n_row//num_lab),len(row_labels[i]))
        for j,ax in enumerate(tmp_axs):
            if label_rows:
                ax.set_ylabel('%s = %s'%(lab,row_labels[i][lab_idx[j]]), 
                              **text_args)
            else:
                ax.set_ylabel('%s'%(row_labels[i][lab_idx[j]]), 
                              **text_args)
        
        row_lab_axs += tmp_axs

    col_lab_axs = []
    num_lab = n_col
    for i,lab in enumerate(col_names):
        num_lab = num_lab//len(col_labels[i])
        
        tmp_axs = []
        for j in range(n_col//num_lab):
            idx0 = num_lab*j + len(row_names)
            idx1 = num_lab*(j+1) + len(row_names)
            tmp_axs.append(fig.add_subplot(gs[i,idx0:idx1]))
        
        lab_idx = np.mod(np.arange(n_col//num_lab),len(col_labels[i]))
        for j,ax in enumerate(tmp_axs):
            if label_cols:
                ax.set_title('%s = %s'%(lab,col_labels[i][lab_idx[j]]),
                              **text_args)
            else:
                ax.set_title('%s'%(col_labels[i][lab_idx[j]]),
                              **text_args)
        
        col_lab_axs += tmp_axs


    for ax in row_lab_axs:
        ax.tick_params(size=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_facecolor("none")
        for pos in ["right", "top", "bottom"]:
            ax.spines[pos].set_visible(False)
        ax.spines["left"].set_linewidth(3)
        ax.spines["left"].set_color("crimson")

    for ax in col_lab_axs:
        ax.tick_params(size=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_facecolor("none")
        for pos in ["right", "left", "bottom"]:
            ax.spines[pos].set_visible(False)
        ax.spines["top"].set_linewidth(3)
        ax.spines["top"].set_color("crimson")

    return axes

def dichotomy_plot(PS, CCGP, SD, PS_err=None, CCGP_err=None, SD_err=None, 
    mutinfo_cmap='copper', var_cmap='tab10', include_legend=True, include_cbar=True, 
    input_dics=None, output_dics=None, other_dics=None, out_MI=None, s=31, **scatter_args):
    '''
    Plot values for each abstraction metric as scatter plots with x-axis noise.

    Optionally highlight the first "num_special" dichotomies.
    '''
    ndic = len(PS)

    color_by_info = (out_MI is not None)

    # be very picky about offsets
    offset = np.random.choice(np.linspace(-0.15,0.15,10), 3*ndic)
    if (output_dics or input_dics) is not None:
        spec_dics = []
        if output_dics is not None:
            spec_dics += output_dics
        if input_dics is not None:
            spec_dics += input_dics
        if other_dics is not None:
            spec_dics += other_dics

        special_offsets = np.linspace(-0.15,0.15,len(spec_dics))
        for i,n in enumerate(spec_dics):
            offset[n+np.arange(3)*ndic] = special_offsets[i]

    xfoo = np.repeat([0,1,2],ndic).astype(int) + offset # np.random.randn(ndic*3)*0.1
    yfoo = np.concatenate((PS, CCGP, SD))
    if PS_err is None:
        yfoo_err = np.zeros(xfoo.shape)
    else:
        yfoo_err = np.concatenate((PS_err, CCGP_err, SD_err))
    if color_by_info:
        cfoo = np.tile(out_MI,3)

    plt.errorbar(xfoo, yfoo, yerr=yfoo_err, linestyle='None', c=(0.5,0.5,0.5), zorder=0)
    if color_by_info:
        scat = plt.scatter(xfoo, yfoo, s=s, c=cfoo, zorder=10, cmap=mutinfo_cmap, **scatter_args)
    else:
        scat = plt.scatter(xfoo, yfoo, s=s, c=[(0.5,0.5,0.5)], zorder=10, **scatter_args)
    # plt.errorbar(xfoo, yfoo, yerr=yfoo_err, linestyle='None', linecolor=cfoo)
    plt.xticks([0,1,2], labels=['PS', 'CCGP', 'Shattering'])
    plt.ylabel('PS or Cross-validated performance')
    if color_by_info and include_cbar:
        plt.colorbar(scat, label='Mutual information with output')

    # highlight special dichotomies
    anns = []
    leg_text = []
    if input_dics is not None:
        for i, n in enumerate(input_dics):
            anns.append(plt.scatter(xfoo[[n,n+ndic,n+2*ndic]], yfoo[[n,n+ndic,n+2*ndic]], marker='d',
                       s=70, linewidths=3, facecolors='none', edgecolors=cm.get_cmap(var_cmap)(i)))

        if include_legend:
            # plt.legend(anns, ['input %d'%(i+1) for i in range(len(anns))])
            leg_text += ['input %d'%(i+1) for i in range(len(anns))]
    if output_dics is not None:
        for i,n in enumerate(output_dics):
            anns.append(plt.scatter(xfoo[[n,n+ndic,n+2*ndic]], yfoo[[n,n+ndic,n+2*ndic]], marker='s',
                       s=70, linewidths=3, facecolors='none', edgecolors=cm.get_cmap(var_cmap)(i)))

        if include_legend:
            # plt.legend(anns, ['output %d'%(i+1) for i in range(len(anns))])
            leg_text += ['output %d'%(i+1) for i in range(len(output_dics))]
    if other_dics is not None:
        for i,n in enumerate(other_dics):
            anns.append(plt.scatter(xfoo[[n,n+ndic,n+2*ndic]], yfoo[[n,n+ndic,n+2*ndic]], marker='*',
                       s=70, linewidths=3, facecolors='none', edgecolors=cm.get_cmap(var_cmap)(i)))
    
        if include_legend:
            # plt.legend(anns, ['output %d'%(i+1) for i in range(len(anns))])
            leg_text += ['special %d'%(i+1) for i in range(len(other_dics))]

    if len(anns)>0 and include_legend:
        plt.legend(anns, leg_text)


def pca3d(X, special_vec=None, **scat_args):
    '''
    Assume that "X" is N_feat x N_samp, "coloring" is a vector of dim N_samp
    '''

    if special_vec is None:
        X_ = X
    else:
        X_ = X - (special_vec[:,None]@special_vec[None,:]/la.norm(special_vec))@X
        y_ = (special_vec[None,:]/la.norm(special_vec))@X

    U, S, _ = la.svd(X-X.mean(1)[:,None], full_matrices=False)
    pcs = X.T@U[:,:3]

    ax = plt.subplot(111, projection='3d')
    if special_vec is None:
        scat = ax.scatter(pcs[:,0],pcs[:,1],pcs[:,2], **scat_args)
    else:
        scat = ax.scatter(y_,pcs[:,0],pcs[:,1], **scat_args)
    # ax.set_xlabel('pc1')
    # ax.set_ylabel('pc2')
    # ax.set_zlabel('pc3')

    util.set_axes_equal(ax)

def square_axis(ax=None):
    if ax is None:
        ax = plt.gca()
    newlims = [np.min([ax.get_ylim(), ax.get_xlim()]), np.max([ax.get_ylim(), ax.get_xlim()])]

    plt.axis('equal')
    plt.axis('square')
    ax.set_xlim(newlims)
    ax.set_ylim(newlims)

def scatter3d(X, ax=None, **scat_args):
    if ax is None:
        ax = plt.subplot(111, projection='3d')
    scat = ax.scatter(X[...,0],X[...,1],X[...,2], **scat_args)

    util.set_axes_equal(ax)

    return ax

def diverging_clim(ax):
    for im in ax.get_images():
        cmax = np.max(np.abs(im.get_clim()))
        im.set_clim(-cmax,cmax)

def pairwise_lines(X, col=(0.5,0.5,0.5), ax=None):

    if ax is None:
        ax=plt.gca()
    num_dat = X.shape[0]
    for ix in combinations(range(num_dat),2):
        ax.plot(X[ix,0],X[ix,1],X[ix,2],color=col)

def color_cycle(cmap, num_col):
    # return getattr(cm,cmap)(np.linspace(0,1,num_col))
    these_cols = ['r','b','g','y','c','m']
    return [mpc.to_rgb(these_cols[i]) for i in range(num_col)]


