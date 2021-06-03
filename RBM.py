import numpy as np
import MetropolisSampling as ms
import minresQLP2 as mqlp
import StochasticReconfiguration
import recenter as rc
import random


gamma = 0.01
n_vis = 5
n_hid = 20

weights = np.array([[complex(random.uniform(0,1)*1e-2,random.uniform(0,1)*1e-2) for i in range(n_vis)] for j in range(n_hid)])
print("init")
print(np.shape(weights))


totParams = n_vis*n_hid

updatedParams = weights.flatten()
updatedParams = np.reshape(updatedParams,totParams)

spins = np.array([-1.,1.,-1.,1.,1.])
for i in range(100):
  gamma = 0.1

  ### SamplingData - OFull, OAvg, EFull, EAvg, spins
  Ns = 250
  OFull,OAvg,EAvg,EFull,spins = ms.MetropolisHastings(500, Ns, weights,spins)
  xCenter, eCenter = rc.recenter(OAvg,OFull,EAvg,EFull,Ns,totParams)
  F, S = rc.ForceVec(xCenter,eCenter)

  #XFunReal = lambda x: ([np.matmul(np.conj(xCenter.real.T),np.matmul(xCenter.real,x))])
  #XFunImag = lambda x: ([np.matmul(np.conj(xCenter.imag.T),np.matmul(xCenter.imag,x))])
  XFunReal = lambda x: ([xCenter.real.T @ xCenter.real @ x])
  XFunImag = lambda x: ([np.conj(xCenter.imag.T) @ xCenter.imag @ x])
  print(np.shape(S.real))
  NuReal = mqlp.MinresQLP(np.array([S.real]),F.real,1e-6,100)
  #NuReal = mqlp.MinresQLP(XFunReal,F.real,1e-6,100)
  NuReal = NuReal[0]
  NuImag = mqlp.MinresQLP(np.array([S.imag]),F.imag,1e-6,100)
  #NuImag = mqlp.MinresQLP(XFunImag,F.imag,1e-6,100)
  NuImag = NuImag[0]
  print("Iteration:", i)
  print(NuReal)
  #print(len(NuReal))
  updatedParams = updatedParams - gamma*(NuReal[:][0] + NuImag[:][0])

  weights = np.array([updatedParams[i] for i in range(n_vis*n_hid)])
  weights = weights.reshape((n_hid,n_vis))
  #print("updated")
  #print(type(weights))
  #visBias = np.array([updatedParams[i + n_vis*n_hid] for i in range(n_vis)])
  #print(visBias)
  #hidBias = np.array([updatedParams[i + n_vis*n_hid + n_vis] for i in range(n_hid)])
  #print(hidBias)