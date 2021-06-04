import numpy as np
import jax.numpy as jnp
import networkx as nx
import StochasticReconfiguration as SR
from copy import deepcopy
from random import randint
import random

def HilbertBuild(edges):
	return edges

def ratioFunk(weights, spins, spinSite):

	### theta = b_j + sum W_ij * v_j
	spinUpdate = deepcopy(spins)

	#preFact = jnp.exp(-2*visBias[spinSite]*spins[spinSite])
	theta = SR.thetaCalc(spinUpdate, weights)
	numer = jnp.prod(jnp.cosh(theta - 2 * weights[:,spinSite]*spins[spinSite]))
	denom = jnp.prod(jnp.cosh(theta))
	multFact = numer/denom
	#print(multFact)
	r = random.uniform(0,1)
	if np.argmin([1., abs(multFact)**2]) >= r:
		#print("flip")
		if spinUpdate[spinSite] == 1.:
			spinUpdate[spinSite] = -1.
		else:
			spinUpdate[spinSite] = 1.
	return spinUpdate, multFact

def MetropolisHastings(steps, sampling, weights, spins):
	''' Hilbert: Graph with the given connections'''
	numHid = 10
	numVis = 5
	totParam = numHid*numVis

	OFull = np.array([[complex(0.,0.) for i in range(totParam)] for j in range(steps-sampling)])
	O = np.array([complex(0.,0.) for i in range(totParam)])
	print("O:",len(O))
	EFull = np.array([complex(0.,0.) for i in range(steps-sampling)])
	EAvg = complex(0.,0.)
	count = 0.
	updatedSpins = deepcopy(spins)
	for i in range(steps):
		spinSite = randint(0,numVis-1)
		updatedSpins = ratioFunk(weights, updatedSpins, spinSite)
		updatedSpins = updatedSpins[0]

		ExpectedEnergy = 0.

		if i >= sampling:
			count += 1.

			O += SR.O_Deriv(updatedSpins,weights)
			OFull[i-sampling] = O

			ELoc = SR.LocalEnergy(spins,updatedSpins,weights)
			#print(ELoc)
			EAvg += ELoc

			EFull[i-sampling] = ELoc

			ExpectedEnergy += HamiltonianExpectation(1, 1, weights, spins)
	ExpectedEnergy = ExpectedEnergy/count
	print(spins)
	print(updatedSpins)
	print("Expected Energy:",ExpectedEnergy)

	OAvg = O/count
	EAvg = EAvg/count

	print(EAvg)
	#print(OAvg)
	return OFull, OAvg, EAvg, EFull, updatedSpins

def HamiltonianExpectation(A, B, weights, spins):
	EnergyA = 0.
	EnergyB = 0.
	hidSpins = 10
	theta = SR.thetaCalc(spins, weights)
	for s in range(len(spins)):
		for i in range(len(spins)):
			EnergyB += B*spins[(i+s)%5]*spins[(i+s+1)%5]
			EnergyAA = 1.
			for j in range(hidSpins):
				EnergyAA *= A*np.cosh(theta[i] - 2* weights[j,(i+s)%5]*spins[(i+s)%5])/np.cosh(theta[i])
			EnergyA += EnergyAA
		operand = ratioFunk(weights, spins, i)
	Energy = -EnergyA - EnergyB
	return Energy