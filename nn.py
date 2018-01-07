import numpy
import scipy.special
import matplotlib.pyplot


class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.learningrate = learningrate

        # weights from input to hidden
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes,-0.5),(self.hnodes,self.inodes)))
        # weights from hidden to output
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5),(self.onodes,self.hnodes)))

        self.activation_function= lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):

        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T

        # Calculate input * weight and use sigma function to get output
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Same for hidden -> output
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # calculate error (target - final_output which is the current value)
        output_errors = targets - final_outputs

        # hidden layer error. weights^T from hidden_output * errors_output
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update values between hidden and output layer
        self.who += self.learningrate * numpy.dot((output_errors*final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # update the input -> hidden layer weights
        self.wih += self.learningrate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),numpy.transpose(inputs))
        pass

    def query(self, inputs_list):
        # Convert the input data into an array
        inputs = numpy.array(inputs_list,ndmin=2).T

        # calculate signals into hidden layer (Input for Hidden layer)
        hidden_inputs = numpy.dot(self.wih,inputs)

        # calculate hidden output
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate final input
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # calculate output
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


input_nodes = 784 # jede Zahl
hidden_nodes = 150
output_nodes = 10

learning_rate = 0.1

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("mnist-dataset/mnist_train.csv","r")
training_data_list = training_data_file.readlines()
training_data_file.close()
# training
for e in range(5):
    for record in training_data_list:
        all_values = record.split(",")
        # scale and shift
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass
test_data_File = open("mnist-dataset/mnist_test.csv","r")
test_data_list = test_data_File.readlines()
test_data_File.close()

scorecard = []

for record in test_data_list:
    all_values = record.split(",")
    correct_label = int(all_values[0])
    print(correct_label, "correct label")
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99)+0.1

    outputs = n.query(inputs)

    label = numpy.argmax(outputs)
    print(label, "networks answer")

    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

scorecard_array = numpy.asarray(scorecard)
print("Performance = ", scorecard_array.sum() / scorecard_array.size)
