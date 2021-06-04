import numpy as np
import jax.numpy as jnp
import cmath


def thetaCalc(spins, weights):
	theta = jnp.dot(weights,spins)
	#print(theta)
	return theta

def LocalEnergy(spins, updatedSpins, weights):
	numHid = 10

	#ELoc = np.array([complex(0.,0.) for i in range(len(spins))])
	ELoc_zz = [1. for i in range(5)]
	for i in range(5):
		for s in range(5):
			ELoc_zz[i] *= spins[(i+s)%5]*spins[(i+s+1)%5]

	### Step 2 & 3:
	ELocx = [1. for i in range(len(spins))]
	ELocy = [1. for i in range(len(spins))]
	theta = thetaCalc(spins,weights)
	for j in range(len(spins)):
		for s in range(len(spins)):
			for i in range(numHid):
				numerx = np.cosh(theta[i] - 2* weights[i,(j+s)%5]*spins[(j+s)%5])
				numery = 1.j * (-1)**((1+spins[(j+s)%5])/2)*numerx
				denom = np.cosh(theta[i])
				ELocx[j] *= numerx/denom
				ELocy[j] *= numery/denom

	ELoc = np.sum(ELocx) + np.sum(ELocy) + np.sum(ELoc_zz)

	#preFact = jnp.exp(-2*jnp.multiply(visBias,spins))
	'''for i in range(len(spins)):
					multArray = jnp.cosh(theta - 2 * weights[:,i]*spins[i])/jnp.cosh(theta)
					#print(multArray)
					#print(preFact)
					ELoc[i] += jnp.prod(multArray)
					ELoc[i] += 1j*(-1)**((1+spins[i])/2)
				ELoc = jnp.sum(ELoc) + ELocJ'''
	#print(ELoc)
	return ELoc

def O_Deriv(spins, weights):

	numHid = 10
	numVis = 5

	#print(O_W)
	O_a = spins
	theta = thetaCalc(spins, weights)
	O_b = jnp.tanh(theta)
	#print(O_b)
	O_W = [[1. for i in range(numVis)] for j in range(numHid)]
	preFact = 1
	for s in range(len(spins)):
		for i in range(numHid):
			for j in range(len(spins)):
				O_W[i][j] *= O_b[i] * spins[(j+s)%5]
	#print(O_W)

	StackDev = np.array(O_W)
	StackDev = StackDev.flatten()
	#print(StackDev)

	return StackDev

