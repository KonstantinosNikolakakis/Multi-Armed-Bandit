### Identifying the arm with the greatest mean value. The epsilon-greedy approach is used for exploration.
### The distribution of the rewards are Gaussian with vartiance 1 and means: [1,2,3,4].
### Output: A plot of the moving average of the data and the estimated mean of the best arm

import numpy as np
import matplotlib.pyplot as plt

class Bandit:
	def __init__(self,m,N):
		self.m=m #true mean
		self.est_mean=0 #initialize the estimated mean
		self.number_of_pulls_so_far=1 #initialize the number of pulls
		self.mean_vec=np.zeros(N) #store the values of the mean estimates

	def pull(self): #pull the corresponding arm
		return np.random.randn()+ self.m #Gaussian Distribution

	def update_mean(self,sample):
		self.number_of_pulls_so_far=self.number_of_pulls_so_far+1 #increase the number of pulls
		self.est_mean = (self.number_of_pulls_so_far-1)/self.number_of_pulls_so_far * self.est_mean + sample/self.number_of_pulls_so_far #update the estimated mean
		self.mean_vec[self.number_of_pulls_so_far]=self.est_mean #store the new mean estimate

def experiment(mean1,mean2,mean3,mean4,eps,N): #epsilon greedy approach
	Bandits=[Bandit(mean1,N),Bandit(mean2,N),Bandit(mean3,N),Bandit(mean4,N)] #initialize the bandits
	data=np.empty(N) #Initialize the data

	for i in range(N):
		print([b.number_of_pulls_so_far for b in Bandits])#print the number of pulls for each arm
		k=np.argmax([b.est_mean + np.sqrt(2*np.log(i)/b.number_of_pulls_so_far) for b in Bandits]) #Choose the next arm based on its confidence bound
		sample=Bandits[k].pull() #sample the the arm k
		Bandits[k].update_mean(sample) #update the mean of the arm k

		data[i]=sample #store the rewards

	moving_average=np.cumsum(data)/(np.arange(N)+1) #Find the moving average

	######## Plot #################################################################################

	k=np.argmax([b.est_mean for b in Bandits])
	Bandits[k].mean_vec[np.argwhere(Bandits[k].mean_vec == 0)] = Bandits[k].est_mean #Fill the remaining zeros in Bandits[k].mean_vec with the last non zro value

	plt.rcParams.update({'font.size': 14})
	MA, = plt.plot(moving_average,label='Moving Average')
	BA, = plt.plot(Bandits[np.argmax([b.est_mean for b in Bandits])].mean_vec,label='Best Arm Estimation Progress')
	plt.plot(np.ones(N)*mean1)
	plt.plot(np.ones(N)*mean2)
	plt.plot(np.ones(N)*mean3)
	plt.plot(np.ones(N)*mean4)
	plt.xscale('log')
	plt.ylabel('Avearages')
	plt.xlabel('Number of Samples')
	plt.legend(bbox_to_anchor=(1.05, 1), loc='down', borderaxespad=0.)
	plt.show()
	###############################################################################################

	return np.argmax([b.est_mean for b in Bandits]) #Return the best arm


if __name__=='__main__':
	x=experiment(0,-3,8,2,0.1,10000) #Run an experiment for 4 Bandits with means 0, -3, 8, 2, eps=0.1 and 10000 number of pulls in total
	print('The arm with the largest mean is the arm:', x+1)#Return the index of the best arm
