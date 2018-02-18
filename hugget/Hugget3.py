# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:59:05 2018

@author: uctpsp1
"""

import sys 
sys.path.insert(0,'../')

from copy import deepcopy
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy import sparse as sp
import time
import matplotlib.pyplot as plt

class HuggetModel:
    
   def __init__(self, BoroLmt, CRRA, DiscFac, Wage,
                 NumGridAsset, NumGridIncome,
                 tolerance, rMax, rMin,
                 aMin, aMax, GridIncome, MarkovMat):
       
       self.BoroLmt = BoroLmt
       self.CRRA = CRRA 
       self.DiscFac =DiscFac
       self.Wage = Wage
       self.NumGridAsset = NumGridAsset
       self.NumGridIncome = NumGridIncome
       self.tolerance = tolerance
       self.rMax = rMax
       self.rMin = rMin
       self.aMin = aMin
       self.aMax = aMax
       self.GridIncome = GridIncome
       self.MarkovMat = MarkovMat
       self.GridAsset = np.linspace(self.BoroLmt, self.aMax, self.NumGridAsset)
       
       
   def printAttr(self):
       aSum = self.aMax + self.aMin
       return aSum
   
   def IndSolve(self, rfree):
       GridAsset = self.GridAsset
       GridIncome = self.GridIncome
       meshs, mesha = np.meshgrid(GridIncome,GridAsset,indexing='ij')
       C = self.Wage*meshs + rfree*mesha
       mutil = lambda x : 1/x**(self.CRRA)
       invmutil = lambda x : 1/x**(1/self.CRRA)
       
       Cold = C
       dist = 9999
       #Aprime = mesha
       while (dist > self.tolerance):
           mu = mutil(C)
           emu = self.MarkovMat.dot(mu)
           Cstar = invmutil(self.DiscFac * (1+rfree) * emu)
           Astar = (Cstar + mesha - self.Wage*meshs)/(1+rfree)
           
           for s in range(0,self.NumGridIncome):
               Savingsfunc = interp1d(Astar[s,:], GridAsset, fill_value='extrapolate')
               Aprime[s] = Savingsfunc(GridAsset)
               Consumptionfunc = interp1d(Astar[s,:], Cstar[s,:], fill_value='extrapolate')
               C[s] = Consumptionfunc(GridAsset)
           
           #print(mesha)
           BorrowConstrained = mesha <  np.tile(np.vstack(Astar[:,0]),(1,self.NumGridAsset))
           C[BorrowConstrained] = (1+rfree) * mesha[BorrowConstrained] + self.Wage*meshs[BorrowConstrained] - self.BoroLmt
           dista = abs(C-Cold)
           dist = dista.max()   
           Cold = C
                      
       Aprime[BorrowConstrained]=self.BoroLmt           
       return {'Aprime': Aprime, 'C': C}
       
       
   def transMat(self, Aprime):
       GridAsset = self.GridAsset
       idx = np.digitize(Aprime, GridAsset)-1
       idx[Aprime <= GridAsset[0]] = 0
       idx[Aprime >= GridAsset[-1]] = self.NumGridAsset-2
       distance = Aprime - GridAsset[idx]
       weightright = np.minimum(distance/(GridAsset[idx+1]-GridAsset[idx]) ,1)
       weightleft = 1-weightright
       ind1, ind2 = np.meshgrid(range(0,self.NumGridIncome),range(0, self.NumGridAsset),indexing='ij')
       row = np.ravel_multi_index([ind1.flatten(order='F'),ind2.flatten(order='F')],(self.NumGridIncome,self.NumGridAsset),order='F')
       rowindex=[]
       colindex=[]
       value=[]

       for s in range(0,self.NumGridIncome):
           pi = np.tile(self.MarkovMat[:,s],(self.NumGridAsset,1))
           rowindex.append(row)
           col = np.ravel_multi_index([[s]*(self.NumGridAsset*self.NumGridIncome),idx.flatten(order='F')],
                                       (self.NumGridIncome,self.NumGridAsset),order='F')
           colindex.append(col)
           rowindex.append(row)
           col = np.ravel_multi_index([[s]*(self.NumGridAsset*self.NumGridIncome),idx.flatten(order='F')+1],
                                       (self.NumGridIncome,self.NumGridAsset),order='F')
           colindex.append(col)
           value.extend((pi.flatten()*weightleft.flatten(order='F'), pi.flatten()*weightright.flatten(order='F') )) 

       value=np.asarray(value)
       rowindex=np.asarray(rowindex)
       colindex=np.asarray(colindex)
       Transition = sp.coo_matrix((value.flatten(), (rowindex.flatten(), colindex.flatten())), shape=(self.NumGridIncome*self.NumGridAsset,self.NumGridIncome*self.NumGridAsset) )
       return Transition

   def unitEigen(self,Transition):
       eigen, distr = sp.linalg.eigs(Transition.transpose(), k=1, which='LM')
       distr = distr/distr.sum()
       distr = distr.reshape(self.NumGridIncome,self.NumGridAsset,order='F').copy()
       distr = distr.real
       return distr
            
   def ExcessA(self,Aprime,distr):
       Aprime = Aprime.flatten(order='F')
       ExcessA = (Aprime.transpose()).dot(distr.flatten(order='F'))
       return ExcessA
   
   def EquilSolve(self):
       rmax = self.rMax
       rmin = self.rMin
       r = (self.rMax+self.rMin)/2
       init = 5
       
       while abs(init) > self.tolerance:
             Result = self.IndSolve(r)
             TRM = self.transMat(Result['Aprime'])
             distr=self.unitEigen(TRM)
             ExcessA=self.ExcessA(Result['Aprime'],distr)
             print ExcessA
             if ExcessA > 0:
                rmax = (r + rmax)/2
             else:
                rmin = (r + rmin)/2
             
             init = rmax-rmin   
             print 'Starting Iteration for r. Difference remaining:     ',  init
             
             r = (rmax + rmin)/2
             print rmax, rmin
            
       return {'r': r, 'Aprime': Result['Aprime'], 'TransitionMtx': TRM, 'dist': distr  }

# -----------------------------------------------------------------------------
# --- Define all of the parameters for Hugget model ------------
# -----------------------------------------------------------------------------

BoroLmt = -2                        # Borrowing limit
CRRA = 2.0                          # Coefficient of relative risk aversion
DiscFac = 0.99                      # Intertemporal discount factor
Wage = 1                            # Wage multiplier

NumGridAsset = 10
NumGridIncome = 5
tolerance = 10**(-5)
rMax = 1/DiscFac -1
rMin = -0.017
aMin = BoroLmt
aMax = 10
GridIncome = np.array([0.6177, 0.8327, 1, 1.2009, 1.6188])
MarkovMat = np.array([[0.7497,  0.2161, 0.0322, 0.002,  0],
                   [0.2161,  0.4708, 0.2569, 0.0542, 0.002],
                   [0.0322,  0.2569, 0.4218, 0.2569, 0.0322],
                   [0.002,   0.0542, 0.2569, 0.4708, 0.2161],
                   [0,       0.002,  0.0322, 0.2161, 0.7497]])

# Make a dictionary to specify model parameters


hugget_param = { 'BoroLmt' : BoroLmt, 
                 'CRRA' : CRRA,
                 'DiscFac' : DiscFac,
                 'Wage' : Wage,
                 'NumGridAsset' : NumGridAsset,
                 'NumGridIncome' : NumGridIncome,
                 'tolerance' : tolerance,
                 'rMax' : rMax,
                 'rMin' : rMin,
                 'aMin' : aMin,
                 'aMax' : aMax,
                 'GridIncome' : GridIncome,
                 'MarkovMat' : MarkovMat}            
       
       
EX       = HuggetModel(**hugget_param)

start_time = time.clock()
result1=EX.IndSolve((rMax+rMin)/2)
#resultr=EX.EquilSolve()
end_time = time.clock()
print 'Elapsed time is ',  (end_time-start_time), ' seconds.'
#print(resultr['r'])

#plt.plot(EX.GridAsset,resultr['Aprime'][1,:])
#plt.legend(['Income 0'])
#plt.xlabel('a')
#plt.ylabel('a prime')       