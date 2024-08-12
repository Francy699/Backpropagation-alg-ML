
Backpropgation .ipynb
Backpropgation .ipynb_
BACKPROPAGATION Backpropagation is an algorithm used to train artificial neural networks. It works by computing the gradient of the loss function with respect to the weights of the network.


[ ]
#Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

[ ]
#sigmoid activation funtion
def sigmoid(x):
    return 1/(1+np.exp(-x))

[ ]
#derivative of the sigmoid funtion
def sigmoid_derivative(x):
    return x*(1-x)

[ ]
#neural network class
class NeuralNetwork:
  def __init__(self,input_size,hidden_size,output_size):
    self.input_size=input_size
    self.hidden_size=hidden_size
    self.output_size=output_size

    #initialize weights
    self.weights_input_hidden=np.random.randn(self.input_size,self.hidden_size)
    self.weights_hidden_output=np.random.randn(self.hidden_size,self.output_size)

    #initialize the bias
    self.bias_hidden=np.zeros((1,self.hidden_size))
    self.bias_output=np.zeros((1,self.output_size))

  #forward propagation
  def forward_propagation(self,x):
    self.hidden_input=np.dot(x,self.weights_input_hidden)+self.bias_hidden
    self.hidden_output=sigmoid(self.hidden_input)
    self.final_input=np.dot(self.hidden_output,self.weights_hidden_output)+self.bias_output
    self.final_output=sigmoid(self.final_input)
    return self.final_output

  #backward propagation
  def backward_propagation(self,x,y,output,learning_rate):
    #calculate error
    output_error=y-output
    output_delta=output_error*sigmoid_derivative(output)

    hidden_error=output_delta.dot(self.weights_hidden_output.T)
    hidden_delta=hidden_error*sigmoid_derivative(self.hidden_output)

    #update weights and biases
    self.weights_hidden_output+=self.hidden_output.T.dot(output_delta)*learning_rate
    self.bias_output+=np.sum(output_delta,axis=0,keepdims=True)*learning_rate
    self.weights_input_hidden+=x.T.dot(hidden_delta)*learning_rate
    self.bias_hidden+=np.sum(hidden_delta,axis=0,keepdims=True)*learning_rate

  #train the model
  def train(self,x,y,learning_rate,epochs):
    for epoch in range(epochs):
      output=self.forward_propagation(x)
      self.backward_propagation(x,y,output,learning_rate)
      if (epoch+1)%1000==0:
        loss=np.mean(np.square(y-output))
        print(f"Epoch: {epoch+1}, Loss: {loss}")

#load the iris dataset
iris=load_iris()
X=iris.data
y=iris.target

  # Convert the target to one-hot encoding
y = np.eye(len(np.unique(y)))[y]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the neural network
input_size = X_train.shape[1]
hidden_size = 5
output_size = y_train.shape[1]

nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(X_train, y_train, epochs=10000, learning_rate=0.01)

# Make predictions on the test set
predictions = nn.forward_propagation(X_test)
predictions = np.argmax(predictions, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Evaluate the model
accuracy = accuracy_score(y_test_labels, predictions)
print(f"Test set accuracy: {accuracy}")
Epoch: 1000, Loss: 0.017042711164388255
Epoch: 2000, Loss: 0.012602216669131333
Epoch: 3000, Loss: 0.011404649140443484
Epoch: 4000, Loss: 0.010878628464532343
Epoch: 5000, Loss: 0.010587405757821895
Epoch: 6000, Loss: 0.010401617001974131
Epoch: 7000, Loss: 0.010270204342640041
Epoch: 8000, Loss: 0.010168920379878533
Epoch: 9000, Loss: 0.010084685566543211
Epoch: 10000, Loss: 0.010009708861690444
Test set accuracy: 1.0
Colab paid products - Cancel contracts here
