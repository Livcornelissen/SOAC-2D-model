#Choose boundary
boundary = np.zeros([M+1,N+1])
boundary[:,0] = 1
boundary[:,-1] = 1
boundary[0,:] = 1
boundary[-1,:] = 1

## add shapes to boundary to simulate objects blocking flow path
 

##square object, pick centrepoint on x axis and then length of square
#N_cen=int(N/5)
#M_cen=int(M/2)
#s_len=6 #length of square is actually s_len+1
#rad=int(s_len/2)
#boundary[M_cen-rad:M_cen+rad,N_cen-rad:N_cen+rad]=1

# circle object
h_cen=25 #N axis centre of circle
k_cen=25 #M axis centre of circle
rad=10


#a=[]
#b=[]
for j in range(1,N):
    for i in range(1,M):
        if (j-h_cen)**2+(i-k_cen)**2 <= rad**2:
            boundary[i,j]=1
            #a.append(i)
            #b.append(j)
    
#plot shape to confirm it is as desired    
#plt.plot(a,b)       
        
boundary = boundary.astype(int)
