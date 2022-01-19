import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

#Parameters
dx = 0.1         #Grid point size
M = 200          #Height of domain
N = 400          #Width of domain
cs = 0.5e3       #Speed of sound
v = 10           #Viscocity
rho0 = 1e3       #Density
T = 10000        #Amount of time steps
dp = 250         #Pressure gradient
tube = False     #Flow in a tube
bump = False     #Bump on side of tube
obst = 2         #0 = none, 1 = square, 2 = circle
c_x = int(N/5)   #x center of obstacle
c_y = int(M/2)   #y center of obstacle
r = 10           #radius of obstacle
visual = 1       #0 = none, 1 = animation, 2 = velocity at one point, 3 = Ellens calculations
Re = 10           #Reynolds number
mov_ball = False            #Want the ball moving?
u0 = np.array([0,-0.1])     #Speed of ball

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
    u[:,:10,1] = np.exp(1-np.arange(1,11,1))*np.mean(u[:,-10:,1],axis=0)+(1-np.exp(1-np.arange(1,11,1)))*u[:,-10:,1]
    u[:,:10,0] = (1-np.exp(1-np.arange(1,11,1)))*u[:,-10:,0]
    u[:,-10:,1] = np.exp(np.arange(1,11,1)-10)*np.mean(u[:,-10:,1],axis=0)+(1-np.exp(np.arange(1,11,1)-10))*u[:,-10:,1]
    u[:,-10:,0] = (1-np.exp(1-np.arange(1,11,1)))*u[:,-10:,0]
    U = u[:,:,0]
    V = u[:,:,1]
    uabs = np.sqrt(U**2+V**2)
    if np.mean(uabs) < Re/dx/2/r*v:
        u[:,:,1] = u[:,:,1] + dp/rho
    
    s = calc_s(e,u,c,w)

    f0 = calc_f0(w,rho,s)

    fnew = np.zeros(np.shape(f))
    if mov_ball:
        for k in range(9):
            fnew[:,:,k] = np.where(np.roll(boundary,e[k],axis=[0,1]),f[:,:,eb[k]],(1 - 1/tau)*np.roll(f[:,:,k],e[k],axis=[0,1]) + 1/tau*np.roll(f0[:,:,k],e[k],axis=[0,1]) + (6*rho0/c)*w[eb[k]]*np.dot(e[eb[k]],u0))
    else:
        for k in range(9):
            fnew[:,:,k] = np.where(np.roll(boundary,e[k],axis=[0,1]),f[:,:,eb[k]],(1 - 1/tau)*np.roll(f[:,:,k],e[k],axis=[0,1]) + 1/tau*np.roll(f0[:,:,k],e[k],axis=[0,1]))

    return u, fnew, rho

def draw_boundary(tube,bump,obst,c_x,c_y,r):
    boundary = np.zeros([M+1,N+1])
    if tube:
        boundary[0,:] = 1
        boundary[-1,:] = 1

    if bump:
        boundary[1,int(N/5-3):int(N/5)] = 1
        boundary[2,int(N/5-2):int(N/5)] = 1
        boundary[3,int(N/5-1):int(N/5)] = 1

    if obst == 1:
        boundary[c_x-r:c_x+r,c_y-r:c_y+r] = 1
    elif obst == 2:
        y, x = np.mgrid[:M+1,:N+1]
        boundary[np.where((x-c_x)**2+(y-c_y)**2<=r**2)] = 1
        boundary[c_y+r+1,c_x] = 1

    return boundary.astype(int)

#Calculate other parameters
c, dt, tau = sys_var(dx,cs,v)
boundary = draw_boundary(tube,bump,obst,c_x,c_y,r)

#Create grid
X, Y = np.meshgrid(dx*np.arange(0,N+1),dx*np.arange(0,M+1))

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
    fig = plt.figure(figsize=(10,8),dpi=200)
    plt.gca().invert_yaxis()
    camera = Camera(fig)
elif visual == 2:
    point = np.zeros(T)

for i in range(T):
    u, f, rho = timestep(f,w,e,c,eb,tau)

    if i%(T/10)==0:
        print('Simulation at '+str(round(i/T*100))+'%')

    if visual == 1:
        if i%100==0:
            U = u[:,:,0]
            V = u[:,:,1]
            vort = (-U+np.roll(U,1,axis=1))/dx - (V-np.roll(V,1,axis=0))/dx
            cp = plt.contourf(X,Y,vort,cmap='bwr',levels=np.arange(-Re*5,Re*5,Re/10))
            plt.quiver(X[::5,::5],Y[::5,::5],V[::5,::5],-U[::5,::5], pivot='mid', width=0.001)
            camera.snap()
    elif visual == 2:
        point[i] = u[10,150,1]
    elif visual == 3:
        U = u[:,:,0]
        sumU=sumU+U
        sqU=sqU+np.square(U)
        SDtU[:,:,i]=(sqU-np.square(sumU)/i)/i
        avgU=sumU/i
        U_pert[i,:,:]=U-avgU
        #U_pert=U-avgU #for animation
        
        V = u[:,:,1]
        sumV=sumV+V
        sqV=sqV+np.square(V)
        SDtV[:,:,i]=(sqV-np.square(sumV)/i)/i
        avgV=sumV/i
        V_pert[i,:,:]=V-avgV

        Kurl =(-U+np.roll(U,1,axis=1))/dx - (V-np.roll(V,1,axis=0))/dx
        
        uabs = np.sqrt(U**2+V**2)
        sumUabs=sumUabs+uabs
        sqUabs=sqUabs+np.square(uabs)
        SDtUabs[:,:,i]=(sqUabs-np.square(sumUabs)/i)/i
        avgUabs=sumUabs/i
        Uabs_pert[i,:,:]=uabs-avgUabs

        #plt.imshow(1-boundary,cmap='gray',extent=[-0.05,N*dx-0.05,M*dx-0.05,-0.05],vmax=50000)
        #plt.imshow(avgU)
        plt.imshow(U_pert)
        #plt.imshow(uabs,vmin=0,vmax=400)
        #plt.imshow(Kurl)
        camera.snap()

print('Simulation at 100%')

print('Re = '+str(round(dx*2*r*np.mean(np.sqrt(U**2+V**2))/v,2)))

if visual == 1:
    plt.colorbar(cp)
    plt.axis('equal')
    animation = camera.animate(interval=10)
    animation.save('animation.gif')
    #plt.show()
elif visual == 2:
    plt.figure()
    plt.plot(point)
    plt.show()
elif visual == 3:
    plt.colorbar()
    plt.axis('equal')
    animation = camera.animate(interval=10)
    animation.save('animation.gif')
    plt.show()
    
    #avgU=avgU/T #average over timesteps 
    SDu=(sqU-np.square(sumU)/T)/T #array of final SD for each gridpoint #this seems way too big, but maybe that is why it breaks down! 
    SDv=(sqV-np.square(sumV)/T)/T
    SDuabs=(sqUabs-np.square(sumUabs)/T)/T

    'can ammend to be for V or Uabs'
    #plt.plot(SDtU[5,5,:]) #can now pick any point on the grid and see the evolution of the standard deviation
    #plt.plot(SDtU[c_y+r,c_x+r,:]) #plot of SD for a point near ball (top right)
    #plt.plot(SDtU[20,45,:]) #plot of SD over time for a point beneath the ball

    'plot the perturbations of a given point over time' #not convinced works properly in the loop
    #plt.plot(U_pert[:,10,10])

    'this plots a plot for the final differences between velocity and the average velocity up to that point'
    #plt.plot(U-avgU)

    #plot of difference between vel and avg vel averaged across the entire grid
    #plt.plot(u_diff)

    'shows a countour plot of the SD at each gridpoint'
    #plt.contourf(SDu)
    #plt.contourf(SDv)
    #plt.contourf(SDuabs)
    #plt.colorbar()

    'if we plot the final curl value it gives a nice plot'
    #but need to get it to save every time
    #same problem with the pertubrations
    #plt.contourf(curl(Ncurl,Mcurl,U,V))

    'plot the curl of a given point over time?'
    #plt.plot(Kurl[:,10,10])

    #sdSD=(np.sum(np.square(SD))-(np.square(np.sum(SD)))/(M+N))/(M+N) #standard deviation across space

    #avgSDu=np.mean(SDu)
    #plt.show()
