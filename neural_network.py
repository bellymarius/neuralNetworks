import numpy as np
import random as rdm
import pickle


"""

Several auxiliaries functions

"""

def sigmoid(M):
	"""float numpy array -> float numpy array
	return M' such that M.shape = M'.shape and
	forall i,j, M'[i][j] = sigmoid(M[i][j])"""
	return np.reciprocal((1 + np.exp(-M)))



def derivative_sigmoid(M):
	"""float numpy array -> float numpy array
	return M' such that M.shape = M'.shape and
	forall i,j, M'[i][j] = dsigmoid/dx (M[i][j])"""
	sigmoid_M = sigmoid(M)
	return sigmoid_M * (1-sigmoid_M) #derivative_sigmoid(x) = d(sigmoid)/dx = exp(-x)/(1+exp(-x))**2



def least_squares(outputs,label) :
	"""float numpy array ,float numpy array -> float
	output and label's shape is (_,1) 
	Return the half of the sum of the squares of the residuals"""
	n = len(outputs)
	cost = 0.
	for i in range(n):
		cost += (outputs[i][0]-label[i][0])**2
	return cost/2

def derivative_least_squares(outputs,label):
	"""float numpy array ,float numpy array -> float numpy array
	output and label's shape is (_,1) 
	Return M' such that M.shape = label.shape = output.shape and
	forall i, M'[i][0] = d_least_squares/d_output[i][0] (output,label)"""
	return outputs - label

def vect_to_int(X):
	"""
	float numpy array -> int
	X's shape is (_,1)
	Return the index of X's maximum
	"""
	res = 0
	for i in range(len(X)):
		if X[i][0]>X[res][0] :
			res = i
	return res


def list_to_int(t):
	"""
	float list -> int
	Return the index of t's maximum
	"""
	res = 0
	for i in range(len(t)):
		if t[i]>t[res]:
			res = i
	return res



class NeuralNetwork():
	"""
Neural Networks class

Constructor :

	NeuralNetwork(structure, activation_function=sigmoid, derivative_activation_function=derivative_sigmoid, cost_function = least_square, partial_derivative_cost_function=derivative_least_square
    \inf=-1 sup=1)

		structure : int numpy array
			The structure of the network. The network has len(structure) layers. The layer i has structure[i] neurons.

		The weights and biases matrix are randomly initialized with real numbers between inf and sup

		activation_function : float numpy array -> float numpy array
			The activation function - sometimes called loss function.  (Default : sigmoid)

		derivative_activation_function : float numpy array -> float numpy array
	  	The derivative of the activation function. (Default : derivative_sigmoid)

		cost_function : float numpy array , float numpy array -> float
			The function used to compare the output of the network with the expected output (Default : least_squares)

		partial_derivative_cost_function : float numpy array , float numpy array -> float numpy array
	  		The derivative of cost_function (Default : derivative_least_squares)



Attributes :

	structure : int array
		The structure of the network. The network has len(structure) layers.
		Forall i in [O, len(structure) - 1] The layer i has structure[i] neurons.

        
	poids : float numpy array list
		forall k, weights[k] has shape (n,p) where p is the number of neurons in the layer k and n in the layer k+1
		forall k, weights[k] is a matrix which contains the weights of the neurons from the layer k on those of the layer k+1
		forall k,i,j poids k[i][j] is the weight of the j-th neuron of the k-layer on the i-th neuron of the layer k+1
		In the documentation, wij[k] (or wij) shall be understood as weights[k][i][j]
        
	biais : float numpy array list
		forall k, biases[k] has shape (n,1) where n is the number of neurons in the layer k+1
		forall k, biases[k] is a vector which contains the biases in the calculus of the activation of the neurons in layer k+1
		forall k,i,j poids k[i][j] is the weight of the j-th neuron of the k-layer on the i-th neuron of the layer k+1
		In the documentation, bi[k] (or bi) shall be understood as biases[k-1][i][0]


	pre_activations : float numpy array list
		forall k, pre_activations[k] has shape (n,1) where n is the number of neurons in the layer k+1
		pre_activations[k][i][0] is the only z such that activation_function(z) = ai[k+1]
		In the documentation, zi[k] (or zi) shall be understood as pre_activations[k-1][i][0]
        
	activation_function : float np.array -> float np.array
		The activation function- sometimes called loss function.
		Default : sigmoide : Mij-> 1/(1 + exp(-Mij))

	derivative_activation_function : float np.array -> float np.array
		The derivative of the activation function. () -> derivee de la fonction d'activation
		Default : derivative_sigmoid : Mij -> sigmoide(Mij) * (1 - sigmoide(Mij))
        
	cost_function :  np.array x np.array -> float
		The function used to compare the output of the network with the expected output
		Default : sum_square_error : (output, expected) -> sum((output[i] - expected[i])²)

	partial_derivative_cost_function : np.array x np.array -> np.array
		The derivative of cost_function by the i-th coordonnate (Default : derivative_sum_square_error)
		Default : derivative_sum_square_error : (output, expected) ->  (2 * (output[j] - expected[j])) 0<j<n
  """



	def __init__(self, structure,activation_function=sigmoid ,derivative_activation_function=derivative_sigmoid ,cost_function=least_squares,partial_derivative_cost_function=derivative_least_squares,inf=-1 ,sup=1):

		print("Creating a new network...")
		self.structure = structure
		self.nb_layers = len(structure)
		self.activations = [np.zeros((structure[k],1)) for k in range(self.nb_layers)]
		self.pre_activations = [np.zeros((structure[k],1)) for k in range(1,self.nb_layers)]
		self.derivated_pre_activations = [np.zeros((structure[k],1)) for k in range(1,self.nb_layers)]
		self.weights = [np.random.uniform(inf, sup, (structure[k+1],structure[k])) for k in range(self.nb_layers - 1)]
		self.biases = [np.random.uniform(inf, sup, (structure[k],1)) for k in range(1,self.nb_layers)]
		self.activation_function = activation_function
		self.derivative_activation_function = derivative_activation_function
		self.cost_function = cost_function
		self.partial_derivative_cost_function = partial_derivative_cost_function
		print("New network created\n")




	def __fill_derivated_pre_activations(self):
		"""
		None -> None
		For all k in range(len(structure)), for all i in len(structure[k]), fill derivated_pre_acivations[k][i] with derivative_activation_function(pre_activations[k])"""
		for layer in range(self.nb_layers - 1) :
			self.derivated_pre_activations[layer] = self.derivative_activation_function(self.pre_activations[layer])
	
        
	def compute(self,features):
		"""
		float numpy array ->  float numpy array
		features.shape = (self.structure[0],1), self.structure[0] is the number of neurons in the first layer of the network.
		Return the output of the network when the vector features is given as input.
		Update activations and pre_activations of the network
		"""
		self.activations[0] = features
		for k in range(self.nb_layers - 1):
			self.pre_activations[k] = np.dot(self.weights[k],self.activations[k]) + self.biases[k]
			self.activations[k+1] = self.activation_function(self.pre_activations[k])
		return self.activations[-1]

    
	def gradient(self,features,label):
		"""
		float numpy array , float numpy array -> float numpy array * float numpy array
		features.shape = (self.structure[0],1), self.structure[0] is the number of neurons in the first layer of the network.
		label.shape = (self.structure[-1],1), self.structure[-1] is the number of neurons in the last layer of the network.
		Return the gradient vector of the cost function as a linear map in the weights and biases space for a given point in the features' space and its label.
		The gradient is returned in a pair (grad_w,grad_b) :
		- grad_w is a float numpy array list such that len(grad_w) = list(weights) and grad_w[i].shape =  weights[i].shape
		forall k,i,j grad_w[k][i][j] = dcost/Wij(k) (features,label)
		- grad_b is a float numpy array list such that len(grad_w) = list(weights) and grad_w[i].shape =  biases[i].shape
		forall k,i grad_b[k][i][0] = dcost/bi(k) (features,label)"""
		output = self.compute(features)	#compute the output of the network
		self.__fill_derivated_pre_activations() #stock all the derivative_activation(pre_activations[k][i]) in an array
		weights_grad = [np.zeros((self.structure[k+1],self.structure[k])) for k in range(self.nb_layers - 1)]
		biases_grad = [np.zeros((self.structure[k],1)) for k in range(1,self.nb_layers)] #initialise weights and biases gradients
		#Compute the gradients in the last layer
		biases_grad[self.nb_layers-2] = self.partial_derivative_cost_function(output,label) * self.derivated_pre_activations[self.nb_layers-2] 
		weights_grad[self.nb_layers-2] = np.dot(biases_grad[self.nb_layers-2],self.activations[self.nb_layers-2].T)
		#Compute the gradients in the previous layers
		for layer in range(self.nb_layers -3,-1,-1): 
			biases_grad[layer] = np.dot(self.weights[layer+1].T,biases_grad[layer+1]) * self.derivated_pre_activations[layer]
			#weights_grad[layer] = np.dot(self.weights[layer],biases_grad[layer])
			weights_grad[layer] = np.dot(biases_grad[layer],self.activations[layer].T)
		return np.array(weights_grad), np.array(biases_grad)


	def train(self,training_features,training_labels,batch_size,learning_rate = 0.05,display_progress = True):
		"""float array numpy list, float array numpy list, int, float,bool -> None
		len(training_features) = len(training_labels and
		forall i, training_features[i].shape = (structure[0],1) and training_labels[i].shape = (structure[-1],0)
		train the network using stochastic gradient descent.
		if display_progress , the function displays the advancement of the training every 1000-elements"""
		training_set_size = len(training_features)
		assert len(training_labels) == training_set_size
		grad_w_list = [] #Gradients are stocked in numpy array
		grad_b_list = []
		q = training_set_size // batch_size
		r = training_set_size % batch_size
		for q0 in range(q): # At each end of this loop, a gradient descent is done
			for r0 in range(batch_size):
				i = q0*batch_size+r0
				feature = training_features[i]
				label = training_labels[i]
				w,b = self.gradient(feature,label)
				grad_w_list.append(w)
				grad_b_list.append(b)
				if display_progress and i%1000 ==0: #Displaying the advancement every 1000-elements
					print("Batch number {}".format(q0))                                                
			grad_w = np.mean(grad_w_list,axis=0)
			grad_b = np.mean(grad_b_list,axis=0)
			for layer in range(self.nb_layers - 1) :
				self.weights[layer] = self.weights[layer] - learning_rate * grad_w[layer]
				self.biases[layer] = self.biases[layer] - learning_rate * grad_b[layer]
				grad_w_list = [] #Gradients are stocked in numpy array
				grad_b_list = []


	def test(self,testing_features,testing_labels,control = True):
		"""
		float array numpy list, float array numpy list,bool -> None
		len(training_features) = len(training_labels and
		forall i, training_features[i].shape = (structure[0],1) and training_labels[i].shape = (structure[-1],0)
		if control , the function also tests a control agent which predicts labels using occurences of previous answers
		"""
		occurences = [0 for i in range(10)]
		testing_set_size = len(testing_features)
		n = len(testing_labels[0])
		assert len(testing_labels) == testing_set_size
		score = 0
		score_control = 0
		#avg_certainty = 0
		for i in range(testing_set_size):
			feature = testing_features[i]
			label = testing_labels[i]
			answer = vect_to_int(label)
			occurences[answer] = occurences[answer] + 1
			prediction = self.predict(feature)
			if prediction == answer:
				score += 1
			if control:
				control_prediction = list_to_int(occurences)
				if control_prediction == answer:
					score_control += 1
		avg_score = score/(testing_set_size)*100
		avg_score_control = score_control/(testing_set_size)*100
		#print("AI average certainty  : {}%".format(avg_certainty/testing_set_size*100))
		print("IA success rate : {} ({} in {})".format(avg_score,score,testing_set_size))
		if control :
			print("Controller success rate : {} ({} in {})".format(avg_score_control,score_control,testing_set_size))
		return avg_score



	def predict(self,features):
		"""
		float array numpy list -> int
		features.shape = (structure[0],1)
		Return the prediction of the network given the features - the index of the maximum of the output
		"""
		output = self.compute(features)
		return vect_to_int(output)

    
    
	def save(self,filename):
		"""save the network in the file filename.data"""
		with open(filename,'wb') as f:
			record = pickle.Pickler(f)
			record.dump(self)
    








		"""
		 The Maths :
		 output shall be understood as compute(features)

		If L is the last layer and yi the i-th coordinate of the expected output (label)
		then dcost/dai[L](input,label) = dcost/doutput[i](output,label) =  derivative_cost_function[i] (output,label)
		et dcost/zi(L) = dcost/ai(L) * dai(L)/dzi(L) =  derivative_cost_function[i] (output,label) * derivative_activation_function(zi(L))

		If k is a layer different from the first one
		dcost/dbj(k) = dcost/dzj(k) * dzj(k)/dbj(k) = dcost/dzj(k) and
		dcost/dzj(k-1) = dcost/daj(k-1) * daj(k-1)/dzj(k-1) = sum(dcost/dzi(k) * dzi(k)/daj(k-1)) * daj(k-1)/dzj(k-1))
		That's to say, dcost/dzj(k-1) = sum(dcost/zi(k) * wij * derivative_activation_function(zj(k-1))
		

		Finally,
		dcost/dbj(k-1) = dcost/dzj(k-1)
		dcost/dWij(k) =  dcost/dzi(k+1) * dzi(k+1)/dWij(k) = aj * dcost/dzi(k+1)
"""
