import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

def sys_var(dx,cs,v):
    c = cs*np.sqrt(3)
    dt = dx/c
    return c, dt, 3*dt*v/dx**2+1/2

def calc_u(f,e,c,rho):
    u = np.zeros([M+1,N+1,9,2])
    u[:,:,:,0] = np.multiply(np.tile(e[:,0],[M+1,N+1,1]),f)
    u[:,:,:,1] = np.multiply(np.tile(e[:,1],[M+1,N+1,1]),f)
    for k in range(9):
        u[:,:,k,0] = c/rho*u[:,:,k,0]
        u[:,:,k,1] = c/rho*u[:,:,k,1]
    return np.sum(u,axis=2)

def calc_s(e,u,c,w):
    s = np.zeros([M+1,N+1,9])
    for k in range(9):
        s[:,:,k] = w[k]*(3*np.sum(e[k]*u,axis=2)/c+4.5*np.sum(e[k]*u,axis=2)**2/c**2-1.5*np.sum(u*u,axis=2)/c**2)
    return s

def calc_f0(w,rho,s):
    f0 = np.zeros([M+1,N+1,9])
    for k in range(9):
        f0[:,:,k] = w[k]*rho+rho*s[:,:,k]
    return f0

def timestep(f,w,e,c,eb,tau):
    
    rho = np.sum(f,axis=2)

    u = calc_u(f,e,c,rho)
    u[:,:,1] = u[:,:,1] + dp/rho
    
    s = calc_s(e,u,c,w)

    f0 = calc_f0(w,rho,s)

    fnew = np.zeros(np.shape(f))
    for k in range(9):
        fnew[:,:,k] = np.where(np.roll(boundary,e[k],axis=[0,1]),f[:,:,eb[k]],(1 - 1/tau)*np.roll(f[:,:,k],e[k],axis=[0,1]) + 1/tau*np.roll(f0[:,:,k],e[k],axis=[0,1]))

    return u, fnew, rho

#Parameters
dx = 0.1
M = 20
N = 200
cs = 1.5e3
v = 10
rho0 = 1e3
T = 5000
dp = 1000

#Calculate other parameters
c, dt, tau = sys_var(dx,cs,v)
X, Y = np.meshgrid(dx*np.arange(0,M+1),dx*np.arange(0,N+1))

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

eb = np.array([0,3,4,1,2,7,8,5,6]) #when it bounces of the boundary

#Weights
w = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])

#Choose boundary
boundary = np.zeros([M+1,N+1])
boundary[0,:] = 1
boundary[-1,:] = 1
boundary[1,int(N/4):int(N/4+3)] = 1
boundary[2,int(N/4+1):int(N/4+3)] = 1
boundary[3,int(N/4+2):int(N/4+3)] = 1

boundary = boundary.astype(int)

#Initial condition
f = rho0*w*np.ones([M+1,N+1,9])
rho = np.sum(f,axis=2)

fig = plt.figure()
camera = Camera(fig)

cross = np.zeros(T)

for i in range(T):
    u, f, rho = timestep(f,w,e,c,eb,tau)

    cross[i] = u[10,150,1]

    if i%(T/10)==0:
        print(str(i/T*100)+'%')
    
    if i%1==0:
        plt.imshow(rho)
        U = u[:,:,0]
        V = u[:,:,1]
        uabs = U**2+V**2
        plt.imshow(uabs)
        camera.snap()

animation = camera.animate()
animation.save('animation.gif')

##U = u[:,:,0]
##V = u[:,:,1]
##uabs = U**2+V**2

##plt.figure()
##plt.imshow(uabs)

##plt.figure()
##plt.imshow(rho)

##plt.figure()
##plt.quiver(X,Y,V*10,-U*10)

##plt.figure()
##plt.plot(U[:,150])
##plt.figure()
##plt.plot(V[:,150])

##print(dx*M*u[10,150,1]/v)

##plt.figure()
##plt.plot(cross)
plt.show()
