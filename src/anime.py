import os, sys
import pickle

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


#####################################

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
    def __init__(self, x, y, auto_zoom=False, trail=False, interval=10, 
        fig=None, ax=None, **scat_args):
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
        if fig is None and ax is None:
            self.fig, self.ax = plt.subplots()
        elif ax is not None:
            self.ax = ax
            self.fig = ax.get_figure()
        elif fig is not None:
            self.fig = fig
            self.ax = fig.add_subplot(111)
        # Then setup FuncAnimation.
        self.ani = anime.FuncAnimation(self.fig, self.update, interval=interval, 
            init_func=self.setup_plot, blit=False, frames=x.shape[-1])

    def save(self,save_dir, fps=30):
        self.ani.save(save_dir, writer=anime.writers['ffmpeg'](fps=fps))

    def setup_plot(self):
        """Initial drawing of the scatter plot."""

        self.patches = []

        # self.ax.plot(self.x,self.y)
        self.scat = self.ax.scatter(self.x[...,0], self.y[...,0], **self.kwargs)

        self.patches += [self.scat,]
        # if not self.auto_zoom:
        #     self.ax.set_xlim([1.1*self.x.min(), 1.1*self.x.max()])
        #     self.ax.set_ylim([1.1*self.y.min(), 1.1*self.y.max()])

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

class RotationAnime(object):
    """ 
    Simply rotate a 3d plot 

    i hate matplotlib animation so much 
    """
    def __init__(self, ax, interval=10, rotation_period=100, 
        frames=None, blit=False,
        init_elev=30, init_azim=30, rot_elev=0, rot_azim=1):
        ''' 
        x and y are shape (num_lines, num_data, num_time) 
        '''

        self.rot_speed = rotation_period
        self.init_elev = init_elev # initial elevation
        self.init_azim = init_azim # initial azimuth
        self.rot_e = rot_elev # how much to rotat3 elevation
        self.rot_a = rot_azim # how much to rotate azimuth

        self.ax = ax
        self.fig = ax.get_figure()

        if frames is None:
            frames = rotation_period*2
        
        # Then setup FuncAnimation.
        self.ani = anime.FuncAnimation(self.fig, self.update, interval=interval, 
            init_func=self.setup_plot, blit=blit, frames=frames)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""

        self.ax.view_init(self.init_elev, self.init_azim)

    def save(self,save_dir, fps=30):
        self.ani.save(save_dir, writer=anime.writers['ffmpeg'](fps=fps))

    def update(self, i):
        """Update the scatter plot."""
        self.ax.view_init(self.init_elev + self.rot_e*(i/self.rot_speed)*360,
                            self.init_azim + self.rot_a*(i/self.rot_speed)*360)

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
        x, y and z are shape (num_lines, num_data, num_time) 
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
    def __init__(self, x, y, z, auto_zoom=False, interval=10, blit=False,
        rotation_period=1000, after_period=0, view_period=0,
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
        self.after_period = after_period
        self.equalize = equalize
        
        self.rot_speed = rotation_period
        self.init_elev = init_elev # initial elevation
        self.init_azim = init_azim # initial azimuth
        self.rot_e = rot_elev # how much to rotat3 elevation
        self.rot_a = rot_azim # how much to rotate azimuth

        self.tot_frames = x.shape[-1] + view_period + after_period

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
            init_func=self.setup_plot, blit=blit, frames=self.tot_frames)

    def save(self, save_dir, fps=30):
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
            set_axes_equal(self.ax)

        self.ax.view_init(self.init_elev, self.init_azim)
    
        return self.patches

    def update(self, i):
        """Update the scatter plot."""

        self.ax.view_init(self.init_elev + self.rot_e*(i/self.rot_speed)*360,
            self.init_azim + self.rot_a*(i/self.rot_speed)*360)

        if i>self.view_period and i<(self.tot_frames-self.after_period):
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

