### Identifying the arm with the greatest mean value. The epsilon-greedy approach is used for exploration.
### The distribution of the rewards are Gaussian with vartiance 1 and means: [1,2,3,4]. 
### Output: A plot of the moving average of the data and the estimated mean of the best arm 

import numpy as np
import matplotlib.pyplot as plt 

class Bandit:
	def __init__(self,m,N):
		self.m=m #true mean
		self.est_mean=0 #initialize the estimated mean 
		self.number_of_pulls_so_far=0 #initialize the number of pulls
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
		p=np.random.random()
		if p<eps: #Explore with probability eps
			k=np.random.choice(4) #Choose randomly an arm for exploration
		else:
			k=np.argmax([b.est_mean for b in Bandits]) #With probability 1-eps choose the arm with greatest estimated mean 
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

	return np.argmax([Bandits[0].est_mean,Bandits[1].est_mean,Bandits[2].est_mean,Bandits[3].est_mean]) #Return the best arm
	

if __name__=='__main__':
	x=experiment(1,2,3,4,0.1,10000) #Run an experiment for 4 Bandits with means 1, 2, 3, 4, eps=0.1 and 10000 number of pulls in total
	print('The arm with the largest mean is the arm:', x+1)#Return the index of the best arm
	

