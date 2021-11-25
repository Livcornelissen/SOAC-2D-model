# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 11:41:16 2021

@author: ellen
"""

import numpy as np
import math as M


#parameters
v=10
c_s=1.5e3 #speed of sound in water m/s
dx=0.1 
rho_0=1 
L=10
dt=dx/(c_s*np.sqrt(3))
tau=1/2+3*v*dt/(dx**2)
c=np.sqrt(3)*c_s # or c=dx/dt delivers same value
p=c_s**2/np.sqrt(3)
x=10 #number of x gridpoints
y=10 #number of y gridpoints

#intialise domain
D=np.zeros([x,y])
#movement vector 9 x 2 array
ei=np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]]) 

#defines each direction that is the corresponding opposite direction to
#each direction of ei
#can define it in a few ways
#ej=np.array([ei[0],ei[3],ei[4],ei[1],ei[2],ei[7],ei[8],ei[5],ei[6]])
j=[0,3,4,1,2,7,8,5,6]
ej=np.array(ei[j])



#weight vector
w=np.array([[4/9],[1/9],[1/9],[1/9],[1/9],[1/36],[1/36],[1/36],[1/36]])
#opposing direction vector

#j_vec=np.array([[0],[3],[4],[1],[2],[7],[8],[5],[6]])

#initialise density
#is this also eqm density then?
fi=np.zeros([x,y,9]) #for each gridpoint there are 9 velocity layers
#initial density has no flow 
#create empty rho to sum over 

rho=0
#create empty velocity vector, one for each gridpoint, two possible 
#directions

u=np.zeros([x,y,2])
for i in range(9):
    fi[:,:,i]=rho_0*w[i]
    #this gives the density by summing over velocity layers, 
    #assume same at all points? so just select a point?
    rho=rho+fi[:,:,i]
    #u=1/rho*(c*u)
    
    #for j in range(x):
    #    vel_i=ei[j]*f[]
    #    u=u+c*ei[i]*fi[i,i,i]
    
#or just define density, creates 2d array from 3d array
rho=np.sum(fi,2)    


#consindering the case of the first gripoint
vel_vec=np.zeros([9,2])
u=np.zeros([x,y,2])

#vel_vec=ei[1]*fi[1,1,1]
#ei_s=np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1/2,1/2],[-1/2,1/2],[-1/2,-1/2],[1/2,-1/2]])


for n in range(x):
    for p in range(y):
        for i in range(9):
            #multiply each flow by each direction, takes on sign of direction
            vel_vec[i]=ei[i]*fi[n,p,i]
    
        for i in range(1):
            #sum up the flows in each direction
            u[n,p,i]=np.sum(vel_vec[:,i])
            #returns all zeros, is this because we are considering a system
            #at rest first? 



nx=10
ny=10
#given by Jan
Xupshift = np.linspace(-1,nx-2,nx,dtype=int)
Xdwshift = np.mod(np.linspace(1,nx,nx,dtype=int),nx)
Yupshift = np.linspace(-1,ny-2,ny,dtype=int)
Ydwshift = np.mod(np.linspace(1,ny,ny,dtype=int),ny)



    
    
    
 

    
    
    
    
    





