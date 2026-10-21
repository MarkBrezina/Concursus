import pandas as pd

import random
import math
import numpy as np

import seaborn as sns
from scipy.integrate import odeint

from IPython import display
from IPython.display import HTML

import matplotlib as mpl
from matplotlib import cm
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.animation as FuncAnimation
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

%matplotlib notebook
%matplotlib inline

from scipy.ndimage import gaussian_filter


class particle:

    def __init__(self, size, x, y, v):

        # components of inner energy
        self.size = size
        self.rotation = random.uniform(0, 1)

        self.position = [x, y]
        self.velocity = [v[0], v[1]]
        # inner_energy?
        # outer_energy?

        #connections
        self.associations = []

    def update(self):
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

    def interaction(self, particle):
        size_coef = max(particle.size / self.size,1)
        dist_coef = (np.sqrt((particle.position[0] - self.position[0])**2 + (particle.position[1] - self.position[1])**2)**2)
        r1 = self.rotation * (2*np.pi)
        r2 = particle.rotation * (2*np.pi)
        rot_coef = round((np.cos(r2) * np.cos(r1)) + (np.sin(r2) * np.sin(r1)),4)
        force_coef = size_coef * rot_coef
        self.velocity[0] = self.velocity[0] + force_coef * (particle.position[0] - self.position[0])/(dist_coef**1.5)
        self.velocity[1] = self.velocity[1] + force_coef * (particle.position[1] - self.position[1])/(dist_coef**1.5)

        # self-destruct and divide
        # absorb
        # emission



class p_field:

    def __init__(self, x, y, p):

        # Take coordinates and shape them into a mesh-map
        self.width_x = x
        self.width_y = y
        self._field = np.zeros((self.width_x, self.width_y))

        self.x_coords = np.linspace(0-x, 0+x, self.width_x)
        self.y_coords = np.linspace(0-y, 0+y, self.width_y)
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)

        # Given the coordinate map, update with appropriate curvature
        for _p in p:
            strength = np.sign(np.cos(_p.rotation*2*np.pi))
            spread = _p.size * 200
            X_c = _p.position[0] #np.random.normal(abs(_p.position[0]), /2, 1)[0]
            Y_c = _p.position[1] #np.random.normal(abs(_p.position[0]), abs(_p.position[1])/2, 1)[0]

            self._field += strength * np.exp(-((self.X - X_c)**2 + (self.Y - Y_c)**2) / spread**2)   # Central well

        self._field = gaussian_filter(self._field, sigma=0.5)


        # expansion rate
        # time


p = []
n = 5
for _ in range(n):
    size = abs(np.random.normal(0, 1, 1)[0])
    x = np.random.normal(0, 500, 1)[0]
    y = np.random.normal(0, 500, 1)[0]
    v_x = 0 #np.random.normal(0, 10, 1)[0]
    v_y = 0 #np.random.normal(0, 10, 1)[0]

    a = particle(size, x, y,[v_x,v_y])
    p.append(a)

width_x = 2000
width_y = 2000

def update(frame):
    
    ax.clear()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim(-width_x, width_x,)
    ax.set_ylim(-width_y, width_y)
    ax.set_aspect('equal', adjustable='box') # Ensure square pixels

    field = p_field(width_x, width_y, p)    
    ax.imshow(field._field, origin='lower', cmap='coolwarm', alpha=0.3,extent=[-width_x, width_x, -width_y, width_y])
    for _p in p:

        if np.cos(_p.rotation*(2*np.pi)) < 0:
            inter = abs(np.cos(_p.rotation*(2*np.pi)))
            clr = 'blue'
        else:
            inter = abs(np.cos(_p.rotation*(2*np.pi)))
            clr = 'red'

        ax.scatter(_p.position[0], _p.position[1], color = clr, alpha = inter)
        _p.update()

        for _p2 in p:
            if _p != _p2:
                _p.interaction(_p2)
    
fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig=fig, func=update, frames=1000, interval=100)
video = ani.to_html5_video()
html = display.HTML(video)
display.display(html)
plt.close()
