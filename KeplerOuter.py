import numpy as np 
class KeplerOuter: 
    """
    KeplerOuter: classe con metodi di 
    integrazione per le equazioni del moto
    """
    def __init__(self,N=1000, h=0.1e-5):
        """
        Inseriamo i valori delle costanti 
        e le condizioni iniziali fornite
        """
        self.G = 2.95912208286e-4
        
        # masse dei corpi, nell'ordine: sole (0), giove (1), saturno (2), urano (3), nettuno (4), plutone (5)
    
        self.m = np.array([1.00000597682, 0.000954786104043, 0.000285583733151,\
                           0.0000437273164546,0.0000517759138449, 1/(1.3e+8)])
        
        # posizioni iniziali dei corpi, nell'ordine: sole (0), giove (1), saturno (2), urano (3), nettuno (4), plutone (5)
        
        self.qx = np.zeros((N+1,6))
        self.qy = np.zeros((N+1,6))
        self.qz = np.zeros((N+1,6))
        
        self.px = np.zeros((N+1,6))
        self.py = np.zeros((N+1,6))
        self.pz = np.zeros((N+1,6))
        
        self.qxi = np.array([0, -3.5023653, 9.0755314, 8.3101420, 11.4707666, -15.5387357])
        self.qyi = np.array([0, -3.8169847, -3.0458353, -16.2901086, -25.7294829, -25.2225594])
        self.qzi = np.array([0, -1.5507963, -1.6483708, -7.2521278, -10.8169456, -3.1902382])
        
        # velocità iniziali dei corpi, nell'ordine: sole (0), giove (1), saturno (2), urano (3), nettuno (4), plutone (5)
        
        vxi = np.array([0, 0.00565429, 0.00168318, 0.00354178, 0.00288930, 0.00276725])
        vyi = np.array([0, -0.00412490, 0.00483525, 0.00137102, 0.001145279, -0.00170702])
        vzi = np.array([0, -0.001905893, 0.00192462, 0.00055029, 0.00039677, -0.00136504])
        self.pxi = self.m[:]*vxi
        self.pyi = self.m[:]*vyi
        self.pzi = self.m[:]*vzi
        
        self.ek = np.zeros(N+1)        
        self.ep = np.zeros(N+1)
        self.ang= np.zeros(N+1)
        
        self.qxt = np.zeros(6)
        self.qyt = np.zeros(6)
        self.qzt = np.zeros(6)
        
        self.pxt = np.zeros(6)
        self.pyt = np.zeros(6)
        self.pzt = np.zeros(6)
        
        self.qxd = np.zeros(6)
        self.qyd = np.zeros(6)
        self.qzd = np.zeros(6)
        
        self.pxd = np.zeros(6)
        self.pyd = np.zeros(6)
        self.pzd = np.zeros(6)
        
        self.h  = h
        self.N  = N
        
        self.qx[0,:] = self.qxi[:]
        self.qy[0,:] = self.qyi[:]
        self.qz[0,:] = self.qzi[:]
        
        self.px[0,:] = self.pxi[:]
        self.py[0,:] = self.pyi[:]
        self.pz[0,:] = self.pzi[:]
        
    #def kineng(self,it=0):
    #    self.ek[it] = 1/(2*self.m[:])*(self.px[it,:]**2 * self.py[it,:]**2 * self.pz[it,:]**2)
                                
  
    def poteng(self,it=0):
        """
        Calcolo energia potenziale del sistema
        """
        for j in range(1,6):
            for k in range(j-1):
                mod = np.sqrt((self.q[:][j])**2+ (self.q[:][k])**2)
                self.ep = -self.G*self.m[j]*self.m[k]/mod 
                           
    
    def keplerdot(self):
        """
        Calcolo derivate con equazioni di Hamilton
        """
        self.qxd[:] = self.pxt[:]/self.m[:]
        self.qyd[:] = self.pyt[:]/self.m[:]
        self.qzd[:] = self.pzt[:]/self.m[:]
        self.pxd[:] = 0.
        self.pyd[:] = 0.
        self.pzd[:] = 0.
        for j in range(6):
            for k in range(6):
                if (j != k):
                    dist_terza = np.sqrt((self.qxt[j]-self.qxt[k])**2 +\
                                  (self.qyt[j]-self.qyt[k])**2 +\
                                  (self.qzt[j]-self.qzt[k])**2)**3
                    self.pxd[j] -= (self.G*self.m[j]*self.m[k]*(self.qxt[j]-self.qxt[k]))/dist_terza
                    self.pyd[j] -= (self.G*self.m[j]*self.m[k]*(self.qyt[j]-self.qyt[k]))/dist_terza
                    self.pzd[j] -= (self.G*self.m[j]*self.m[k]*(self.qzt[j]-self.qzt[k]))/dist_terza
    
    
    def forward_euler(self,it, neuler):
        """
        Implementazione metodo di Eulero all'avanti
        """
        self.qxt[:] = self.qx[it,:]
        self.qyt[:] = self.qy[it,:]
        self.qzt[:] = self.qz[it,:]
        
        self.pxt[:] = self.px[it,:]
        self.pyt[:] = self.py[it,:]
        self.pzt[:] = self.pz[it,:]
        
        for j in range(neuler):
            # calcolo derivate
            self.keplerdot()
            # iterazione
            self.qxt += self.h*self.qxd
            self.qyt += self.h*self.qyd
            self.qzt += self.h*self.qzd
            
            self.pxt += self.h*self.pxd
            self.pyt += self.h*self.pyd
            self.pzt += self.h*self.pzd
            
        it += 1
        
        self.qx[it,:] = self.qxt[:]
        self.qy[it,:] = self.qyt[:]
        self.qz[it,:] = self.qzt[:]
        
        self.px[it,:] = self.pxt[:]
        self.py[it,:] = self.pyt[:]
        self.pz[it,:] = self.pzt[:]
        
        
    def RK4(self,it,nrk4):
        """
        Implementazione del metodo Runge-Kutta 4
        """
        self.qxt[:] = self.qx[it,:]
        self.qyt[:] = self.qy[it,:]
        self.qzt[:] = self.qz[it,:]
        self.pxt[:] = self.px[it,:]
        self.pyt[:] = self.py[it,:]
        self.pzt[:] = self.pz[it,:]
        for j in range(nrk4): 
            qxn = self.qxt.copy()
            qyn = self.qyt.copy()
            qzn = self.qzt.copy()
            pxn = self.pxt.copy()
            pyn = self.pyt.copy()
            pzn = self.pzt.copy()
            # iterazione K1
            self.keplerdot()
            k1qx=self.qxd.copy()
            k1qy=self.qyd.copy()
            k1qz=self.qzd.copy()
            k1px=self.pxd.copy()
            k1py=self.pyd.copy()
            k1pz=self.pzd.copy()
            # iterazione K2
            self.qxt = qxn + 0.5*self.h*self.qxd
            self.qyt = qyn + 0.5*self.h*self.qyd
            self.qzt = qzn + 0.5*self.h*self.qzd
            self.pxt = pxn + 0.5*self.h*self.pxd
            self.pyt = pyn + 0.5*self.h*self.pyd
            self.pzt = pzn + 0.5*self.h*self.pzd
            self.keplerdot()
            k2qx=self.qxd.copy()
            k2qy=self.qyd.copy()
            k2qz=self.qzd.copy()
            k2px=self.pxd.copy()
            k2py=self.pyd.copy()
            k2pz=self.pzd.copy()
            # iterazione K3
            self.qxt = qxn + 0.5*self.h*self.qxd
            self.qyt = qyn + 0.5*self.h*self.qyd
            self.qzt = qzn + 0.5*self.h*self.qzd
            self.pxt = pxn + 0.5*self.h*self.pxd
            self.pyt = pyn + 0.5*self.h*self.pyd
            self.pzt = pzn + 0.5*self.h*self.pzd
            self.keplerdot()
            k3qx=self.qxd.copy()
            k3qy=self.qyd.copy()
            k3qz=self.qzd.copy()
            k3px=self.pxd.copy()
            k3py=self.pyd.copy()
            k3pz=self.pzd.copy()
            # iterazione K4
            self.qxt = qxn + self.h*self.qxd
            self.qyt = qyn + self.h*self.qyd
            self.qzt = qzn + self.h*self.qzd
            self.pxt = pxn + self.h*self.pxd
            self.pyt = pyn + self.h*self.pyd
            self.pzt = pzn + self.h*self.pzd
            self.keplerdot()
            k4qx=self.qxd.copy()
            k4qy=self.qyd.copy()
            k4qz=self.qzd.copy()
            k4px=self.pxd.copy()
            k4py=self.pyd.copy()
            k4pz=self.pzd.copy()
            #  step RK4
            self.qxt = qxn + self.h/6.*(k1qx + 2.*k2qx + 2.*k3qx + k4qx)
            self.qyt = qyn + self.h/6.*(k1qy + 2.*k2qy + 2.*k3qy + k4qy)
            self.qzt = qzn + self.h/6.*(k1qz + 2.*k2qz + 2.*k3qz + k4qz)
            self.pxt = pxn + self.h/6.*(k1px + 2.*k2px + 2.*k3px + k4px)
            self.pyt = pyn + self.h/6.*(k1py + 2.*k2py + 2.*k3py + k4py)
            self.pzt = pzn + self.h/6.*(k1pz + 2.*k2pz + 2.*k3pz + k4pz)
        it += 1
        self.qx[it,:]= self.qxt[:]
        self.qy[it,:]= self.qyt[:]
        self.qz[it,:]= self.qzt[:]
        self.px[it,:]= self.pxt[:]
        self.py[it,:]= self.pyt[:]
        self.pz[it,:]= self.pzt[:]
        
    def RK2(self,it,nrk2):
        """
        Implementazione del metodo di Runge-Kutta 2
        """
        self.qxt[:] = self.qx[it,:]
        self.qyt[:] = self.qy[it,:]
        self.qzt[:] = self.qz[it,:]
        self.pxt[:] = self.px[it,:]
        self.pyt[:] = self.py[it,:]
        self.pzt[:] = self.pz[it,:]
        for j in range(nrk2):
            qxn = self.qxt.copy()
            qyn = self.qyt.copy()
            qzn = self.qzt.copy()
            pxn = self.pxt.copy()
            pyn = self.pyt.copy()
            pzn = self.pzt.copy()
            #iterazione K1
            self.keplerdot()
            k1qx = self.qxd.copy()
            k1qy = self.qyd.copy()
            k1qz = self.qzd.copy()
            k1px = self.pxd.copy()
            k1py = self.pyd.copy()
            k1pz = self.pzd.copy()
            #calcolo punto intermedio
            self.qxt[:] = qxn[:] + self.h*self.qxd[:]
            self.qyt[:] = qyn[:] + self.h*self.qyd[:]
            self.qzt[:] = qzn[:] + self.h*self.qzd[:]
            self.pxt[:] = pxn[:] + self.h*self.pxd[:]
            self.pyt[:] = pyn[:] + self.h*self.pyd[:]
            self.pzt[:] = pzn[:] + self.h*self.pzd[:]
            #iterazione K2
            self.keplerdot()
            k2qx = self.qxd.copy()
            k2qy = self.qyd.copy()
            k2qz = self.qzd.copy()
            k2px = self.pxd.copy()
            k2py = self.pyd.copy()
            k2pz = self.pzd.copy()
            #calcolo valori successivi delle coordinate
            self.qxt[:] = qxn[:] + self.h/2.*(k1qx[:] + k2qx[:])
            self.qyt[:] = qyn[:] + self.h/2.*(k1qy[:] + k2qy[:])
            self.qzt[:] = qzn[:] + self.h/2.*(k1qz[:] + k2qz[:])
            self.pxt[:] = pxn[:] + self.h/2.*(k1px[:] + k2px[:])
            self.pyt[:] = pyn[:] + self.h/2.*(k1py[:] + k2py[:])
            self.pzt[:] = pzn[:] + self.h/2.*(k1pz[:] + k2pz[:])
        it += 1
        self.qx[it,:]= self.qxt[:]
        self.qy[it,:]= self.qyt[:]
        self.qz[it,:]= self.qzt[:]
        self.px[it,:]= self.pxt[:]
        self.py[it,:]= self.pyt[:]
        self.pz[it,:]= self.pzt[:]
        
    def misto(self,it,nmisto):
        """
        Implementazione del metodo di Eulero "misto":
        il calcolo del p_n+1 è calcolato col metodo all'avanti
        e il q_n+1 con quello all'indietro
        """
        self.qxt[:] = self.qx[it,:]
        self.qyt[:] = self.qy[it,:]
        self.qzt[:] = self.qz[it,:]
        self.pxt[:] = self.px[it,:]
        self.pyt[:] = self.py[it,:]
        self.pzt[:] = self.pz[it,:]
        
        for j in range(nmisto):
            # calcolo derivate
            self.keplerdot()
            # iterazione
            self.pxt += self.h*self.pxd
            self.pyt += self.h*self.pyd
            self.pzt += self.h*self.pzd
            self.keplerdot()
            self.qxt += self.h*self.qxd
            self.qyt += self.h*self.qyd
            self.qzt += self.h*self.qzd
            
        it += 1
        
        self.qx[it,:] = self.qxt[:]
        self.qy[it,:] = self.qyt[:]
        self.qz[it,:] = self.qzt[:]
        
        self.px[it,:] = self.pxt[:]
        self.py[it,:] = self.pyt[:]
        self.pz[it,:] = self.pzt[:]