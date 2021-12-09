import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

def sys_var(dx,cs,v):
    c = cs*np.sqrt(3)
    dt = dx/c
    return c, dt, 3*dt*v/dx**2+1/2

def calc_u(f,e,c,rho):
    u = np.zeros([N+1,N+1,9,2])
    u[:,:,:,0] = np.multiply(np.tile(e[:,0],[N+1,N+1,1]),f)
    u[:,:,:,1] = np.multiply(np.tile(e[:,1],[N+1,N+1,1]),f)
    #pre = c*np.ones([N+1,N+1])/rho
    for k in range(9):
        u[:,:,k,0] = c/rho*u[:,:,k,0]
        u[:,:,k,1] = c/rho*u[:,:,k,1]
    return np.sum(u,axis=2)

def calc_s(e,u,c,w):
    s = np.zeros([N+1,N+1,9])
    for i in range(N+1):
        for j in range(N+1):
            s[i,j,:] = w*(3*np.dot(u[i,j],np.transpose(e))/c+4.5*(np.dot(u[i,j],np.transpose(e))**2/c**2)-1.5*np.dot(u[i,j],u[i,j])*np.ones(9)/c**2)
    return s

def calc_f0(w,rho,s):
    f0 = np.zeros([N+1,N+1,9])
    for k in range(9):
        f0[:,:,k] = w[k]*rho+rho*s[:,:,k]
    return f0

def timestep(f,w,e,c,ej,tau,eb):
    
    rho = np.sum(f,axis=2)

    u = calc_u(f,e,c,rho)

    s = calc_s(e,u,c,w)

    f0 = calc_f0(w,rho,s)

    fnew = np.zeros(np.shape(f))
    for i in range(N+1):
        for j in range(N+1):
            coord = (np.tile(np.array([i,j]),(9,1))+ej) % (N+1)
            for k in range(9):
                if boundary[coord[k][0],coord[k][1]]:
                    fnew[i,j,k] = f[i,j,eb[k]] #simply take the reflection. 
                else:
                    fnew[i,j,k] = (1 - 1/tau)*f[coord[k][0],coord[k][1],k] + 1/tau*f0[coord[k][0],coord[k][1],k]
    return u, fnew, rho

#Parameters
dx = 0.1
N = 30
cs = 1.5e3
v = 10
rho0 = 1e3

#Calculate other parameters
c, dt, tau = sys_var(dx,cs,v)
X, Y = np.meshgrid(dx*np.arange(0,N+1),dx*np.arange(0,N+1))

#Unit vectors
e = np.array([[0,0],\
              [1,0],\
              [0,1],\
              [-1,0],\
              [0,-1],\
              [1,1],\
              [-1,1],\
              [-1,-1],\
              [1,-1]])

ej = np.array([[0,0],\
               [-1,0],\
               [0,-1],\
               [1,0],\
               [0,1],\
               [-1,-1],\
               [1,-1],\
               [1,1],\
               [-1,1]])

eb = np.array([0,3,4,1,2,7,8,5,6]) #when it bounces of the boundary

#Weights
w = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])

#Choose boundary
boundary = np.zeros([N+1,N+1])
boundary[:,0] = 1
boundary[:,-1] = 1
boundary[0,:] = 1
boundary[-1,:] = 1

#Initial condition
f = rho0*w*np.ones([N+1,N+1,9])
f[int(N/2):int(N/2)+2,int(N/2):int(N/2)+2,:] = 2.1*rho0*w*np.ones(9)
rho = np.sum(f,axis=2)

##fig = plt.figure()
##camera = Camera(fig)

for i in range(21):
    u, f, rho = timestep(f,w,e,c,ej,tau)
    
##    U = u[:,:,0]
##    V = u[:,:,1]
##    uabs = U**2+V**2
##    plt.imshow(uabs)
##    plt.quiver(X,Y,U,V)
##    camera.snap()

##animation = camera.animate()
##animation.save('animation.gif')

U = u[:,:,0]
V = u[:,:,1]
uabs = U**2+V**2

plt.figure()
plt.imshow(uabs)

plt.figure()
plt.imshow(rho)

##plt.figure()
##plt.quiver(X,Y,V,-U)
plt.show()
