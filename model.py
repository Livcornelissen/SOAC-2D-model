import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

#Parameters
dx = 0.1        #Grid point size
M = 20          #Height of domain
N = 200         #Width of domain
cs = 1.5e3      #Speed of sound
v = 10          #Viscocity
rho0 = 1e3      #Density
T = 5000        #Amount of time steps
dp = 100        #Pressure gradient
tube = True     #Flow in a tube
bump = True     #Bump on side of tube
obst = 0        #0 = none, 1 = square, 2 = circle
c_x = 0         #x center of obstacle
c_y = 0         #y center of obstacle
r = 0           #radius of obstacle
visual = 2      #0 = none, 1 = animation, 2 = velocity at one point

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

def draw_boundary(tube,bump,obst,c_x,c_y,r):
    boundary = np.zeros([M+1,N+1])

    if tube:
        boundary[0,:] = 1
        boundary[-1,:] = 1

    if bump:
        boundary[1,int(N/5):int(N/5+3)] = 1
        boundary[2,int(N/5+1):int(N/5+3)] = 1
        boundary[3,int(N/5+2):int(N/5+3)] = 1

    if obst == 1:
        boundary[c_x-r:c_x+r,c_y-r:c_y+r] = 1
    elif obst == 2:
        x, y = numpy.mgrid[:M+1,:N+1]
        boundary[np.where((x-c_x)**2+(y-c_y)**2<=r**2)] = 1

    return boundary.astype(int)

#Calculate other parameters
c, dt, tau = sys_var(dx,cs,v)
boundary = draw_boundary(tube,bump,obst,c_x,c_y,r)

#Create grid
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

#When it bounces of the boundary
eb = np.array([0,3,4,1,2,7,8,5,6]) 

#Weights
w = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])

#Initial condition
f = rho0*w*np.ones([M+1,N+1,9])
rho = np.sum(f,axis=2)

if visual == 1:
    fig = plt.figure()
    camera = Camera(fig)
elif visual == 2:
    point = np.zeros(T)

for i in range(T):
    u, f, rho = timestep(f,w,e,c,eb,tau)

    if i%(T/10)==0:
        print('Simulation at '+str(round(i/T*100))+'%')

    if visual == 1:
        if i%10==0:
            U = u[:,:,0]
            V = u[:,:,1]
            uabs = U**2+V**2
            plt.imshow(uabs)
            camera.snap()
    elif visual == 2:
        point[i] = u[10,150,1]

print('Simulation at 100%')

print('Re = '+str(round(dx*M*np.max(u[:,:,1])/v,2)))

if visual == 1:
    animation = camera.animate()
    animation.save('animation.gif')
    plt.show()
elif visual == 2:
    plt.figure()
    plt.plot(point)
    plt.show()
