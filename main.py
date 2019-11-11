# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:52:22 2019

@author: Piotr Sowiński
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime

from NSS_NN import interpretations as ip, toolbox as tb


def run_on_interset(lognet_spec, interset, epochs):
    today = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results = []
    for interpretation in interset:
        l = LogNet(lognet_spec)
        l.fit_interpretation(interpretation, epochs=epochs)
        results.append(l.test_on_subsets())
    with open('results_'+today+'.txt', 'a') as file:
        for result, interpret in zip(results, interset):
            file.write(str([interpret.specification, result])+',')
    return results


def valuation(x, a_min):
    if (type(x) == list) or (type(x) == np.ndarray):
        return [valuation(elem, a_min) for elem in x]
    else:
        if x <= (-1 * a_min):
            return -1
        elif x >= a_min:
            return 1
        return 0


def idem(x):
    """
    :param x: any number (Float)
    :return: exactly the same number (Float)
    """
    return x


def const(x):
    """
    :param x: any number (Float)
    :return: 1 (Float)
    """
    return 1.0


def tanh(x):
    """
    :param x: any number (Float)
    :return: Bipolar sigmoid function of x (Float)

    Note: this is not a hyperbolic tangent function. Its similar but slightly different
    """
    return (2 / (1 + math.exp(-x))) - 1


def tanh_d(x):
    """
    :param x: any number (Float)
    :return: Derivative of bipolar sigmoid function of x (Float)

    Note: this is not a derivative of hyperbolic tangent function. Its similar but slightly different.
    """
    return (2 * math.exp(-x)) / ((math.exp(-x) + 1) ** 2)


def to_mask(table):
    """
    :param table: 2D Array with weights between layers (List of List of Float)
    :return: 2D Array with boolean values of elements in input array (numpy.array of Bool)

    Takes array and returns the corresponding array with value False in places where there was 0, True otherwise.
    """
    return np.array([[bool(x) for x in row] for row in table])


def show_results(*lines):
    """
    :param lines: Results (usually Error vector) (Tuple with List and name (Str))
    :return: None

    Display chart with lines shown.
    """
    plt.figure(num=None, figsize=(13, 6), dpi=80, facecolor='w', edgecolor='k')
    for line, name in lines:
        plt.plot([i for i in range(len(line))], line, label=name)
    plt.legend(loc='upper left')
    plt.show()


def read_hs(filename):
    """
    :param filename: path to file with recipe for Neural Network (Str)
    :return: Evaluated file (Dict)

    Takes path of file and returns evaluated dictionary.
    """
    with open(filename) as file:
        to_return = eval(file.read().replace('=', ':'))
        if type(to_return) == dict:
            return to_return
        else:
            raise TypeError


class Layer:
    """
    Basic layer class with all its parameters.
    """

    def __init__(self, specification):
        """
        :param specification: 2D array
        specification parameter should look as follows:

        specification = [[______designates______],
                         [_activation functions_],
                         [________biases________],
                         [________labels________]]

        """
        self.specification = specification
        self.designates = np.array(specification[0])
        self.labels = np.array(specification[3])
        self.len = len(self.labels)
        self.aggregated = np.array([0 for i in range(self.len)])
        self.activation = np.array([eval(f) for f in specification[1]])
        self.output = np.array([0 for i in range(self.len)])
        self.bias = np.array(specification[2]).astype(float)

    def forward(self, inputs, weights=None):
        """

        :param inputs: values from previous layer (or inputs to network) (List or 1D numpy.array)
        :param weights: 2D array with weights where number of rows correspond to number of inputs and
                        number of columns correspond to number of neurons in current layer (List or 2D numpy.array)
        :return: None

        Takes input vector and matrix of weights then calculates and changes layer output vector.
        """

        if weights is not None:
            self.output = np.array(
                [self.activation[i](value) for i, value in enumerate(np.dot(inputs, weights) - self.bias)])
        else:
            self.output = np.array([self.activation[i](value) for i, value in enumerate(inputs - self.bias)])

    def order(self):
        """
        :return: Haskell module readable list of atoms in layer
        """
        o = [int(tb.digits(idx)) if len(tb.digits(idx)) else -1 for idx in self.labels]
        maxidx = max(o)
        phrase = []
        for elem in o:
            if elem >=0:
                phrase.append('A {idx = ' + str(elem) + ', label = []}')
            else:
                phrase.append("A {idx = " + str(maxidx+1) + ", label = ['t','r','u','t','h']}")
        return '[' + ', '.join(phrase) + ']'


class LogNet:
    """
    Main class of neural network
    """
    def __init__(self, architecture):
        """

        :param architecture: Specification of neurons and connections (Dict)
        parameter architecture should look as follows:

        architecture = {inpLayer: [(designate, activation function, bias, label), ... ],
                        hidLayer: [(designate, activation function, bias, label), ... ],
                        outLayer: [(designate, activation function, bias, label), ... ],
                        inpToHidConnections: [(label of input neuron, label of output neuron, weight), ... ],
                        hidToOutConnections: [(label of input neuron, label of output neuron, weight), ... ],
                        recConnections: [(label of input neuron, label of output neuron, weight of connection), ... ]}

        """

        self.architecture = architecture
        self.inp = Layer(np.array([list(spec) for spec in architecture["inpLayer"]]).T.tolist())
        self.hid = Layer(np.array([list(spec) for spec in architecture["hidLayer"]]).T.tolist())
        self.out = Layer(np.array([list(spec) for spec in architecture["outLayer"]]).T.tolist())

        self.i2h = np.array(self.set_weights(architecture['inpToHidConnections'], self.inp.labels, self.hid.labels))
        self.h2o = np.array(self.set_weights(architecture['hidToOutConnections'], self.hid.labels, self.out.labels))
        self.o2i = np.array(self.set_weights(architecture['recConnections'], self.out.labels, self.inp.labels))

        self.i2h_b = to_mask(self.i2h)
        self.h2o_b = to_mask(self.h2o)
        self.o2i_b = to_mask(self.o2i)

        self.model = []

    def set_weights(self, connections, inp, out):
        """
        :param connections: List with connections between layers (List of Tuples)
        :param inp: Labels of neurons in input layer (List of Strings)
        :param out: Labels of neurons in output layer (List of Strings)
        :return: Matrix with connections (List of List)

        Takes specification of connections, input labels and output labels and returns matrix with connections, where:
            >> number of rows correspond to number of input neurons
            >> number of columns correspond to number of output neurons
            >> at the intersection of input row and output column there is a weight of connection

        """
        weights = [[0 for o in range(len(out))] for i in range(len(inp))]
        inp_dict = dict(zip(inp, [i for i in range(len(inp))]))
        out_dict = dict(zip(out, [i for i in range(len(out))]))

        for inpLab, outLab, weight in connections:
            weights[inp_dict[inpLab]][out_dict[outLab]] = weight

        return weights

    def stabilization(self, amin, maxiter=2000, report=False, soft_input=None):
        """
        :param amin: minimal value of neuron to be considered True (value > amin) or False (value < -amin) (Float)
        :param maxiter: maximal number of iterations of stabilization (Int)
        :param report: if number of iterations will be printed (Bool)
        :param soft_input: vector of values assigned as values of input layer neurons (List of Floats)
        :return: None or List of Float

        Stabilizes network by simulating iterations of Tp operator. Stops when neurons in output layer doesn't change
        their value.
        """
        self.inp.output = np.array([-1.0 for i in range(self.inp.len)])
        if soft_input:
            self.inp.output = soft_input

        self.inp.output[-1] = 1.0

        ite = 0
        for i in range(maxiter):
            values = valuation(self.out.output, amin)
            self.hid.forward(self.inp.output, self.i2h)
            self.out.forward(self.hid.output, self.h2o)
            self.inp.forward(self.out.output, self.o2i)
            ite += 1
            if valuation(self.out.output, amin) == values:
                break

        if report:
            print("Stabilization done in", ite, "iterations")
        if soft_input:
            return self.out.output

    def predict(self, x):
        """
        :param x: Input values (List of Floats)
        :return: Output values (List of Floats)

        Takes list of input values and calculates output of the neural network.
        """

        self.inp.forward(x)
        self.hid.forward(self.inp.output, self.i2h)
        self.out.forward(self.hid.output, self.h2o)
        return self.out.output

    def backprop(self, y, n=0.1):
        """

        :param y: Expected output values (List of Floats)
        :param n: Learning rate (Float)
        :return: Error vector (List of Floats)

        Takes expected values then calculates and propagates error through network.
        """
        error = (y - self.out.output)
        d_weights2 = [[n * error[o] * self.hid.output[i] * tanh_d(self.out.aggregated[o]) for o in range(self.out.len)]
                      for i in range(self.hid.len)]

        error2 = np.dot(error, self.h2o.T)

        d_weights1 = [[n * error2[o] * self.inp.output[i] * tanh_d(self.hid.aggregated[o]) for o in range(self.hid.len)]
                      for i in range(self.inp.len)]
        self.h2o += d_weights2 * self.h2o_b
        self.i2h += d_weights1 * self.i2h_b
        return (error*error) / 2

    def fit(self, y, epochs, amin, report=True):
        """
        :param y: Expected values (List or numpy.array)
        :param epochs: Number of learning epochs (Int)
        :param amin: minimal value of neuron to be considered True (value > amin) or False (value < -amin) (Float)
        :param report: if True it print progressbar (Bool)
        :return: None

        Repeats stabilization and backpropagation for number of epochs.
        """
        if y == 'ones':
            y = [1 for i in range(self.out.len)]
        errors = []
        if report:
            tb.printProgressBar(0, epochs, prefix='Progress:', length=50)
        for e in range(epochs):
            self.stabilization(amin)
            errors.append(self.backprop(y))
            if report:
                tb.printProgressBar(e + 1, epochs, prefix='Progress:', length=50)
        if report:
            show_results(([np.average([error]) for error in errors], 'mean squarred error'))

    def test_on_subsets(self):
        """
        :return: Pairs of output and input layer values (List of tuples)

        For every possible combination of inputs calculates the output by predict() function.
        """
        subs = tb.subsets(self.inp.len)
        res = []
        for i in range(len(subs)):
            res.append((subs[i], [round(x, 2) for x in self.predict(subs[i]).tolist()]))
        return res
    """
    def fit_interpretation(self, interpretation, epochs, report=True):
        errors = []

        input_vector = ip.by_vector(interpretation.input, self.inp.designates)
        output_vector = ip.by_vector(interpretation.output, self.out.designates)

        if report:
            tb.printProgressBar(0, epochs, prefix='Progress:', length=50)

        for e in range(epochs):
            self.predict(input_vector)
            errors.append(self.backprop(output_vector))
            if report:
                tb.printProgressBar(e + 1, epochs, prefix='Progress:', length=50)

        if report:
            show_results(([np.average([error]) for error in errors], 'mean squarred error'))

        return [np.average([error]) for error in errors]

    """
    def fit_interpretations(self, interpretations, epochs, report=True):
        """
        :param interpretations: list with interpretations (List of interpretations.Interpretation)
        :param epochs: number of learning epochs (Int)
        :param report: if True it print progressbar (Bool)
        :return: error vector (List)

        Fits interpretations by predict and backpropagation.
        """
        errors = []
        len_ip = len(interpretations)

        for e in range(epochs):

            str_epoch = "epoch " + str(e) + '/' + str(epochs)
            if report:
                tb.printProgressBar(0, len_ip, prefix='Progress of '+str_epoch, length=50)

            for var_ip, interpretation in enumerate(interpretations):
                if report:
                    tb.printProgressBar(var_ip+1, len_ip, prefix='Progress of '+str_epoch, length=50)

                input_vector = ip.by_vector(interpretation.input, self.inp.designates)
                output_vector = ip.by_vector(interpretation.output, self.out.designates)

                self.predict(input_vector)
                errors.append(self.backprop(output_vector))

        if report:
            show_results(([np.average([error]) for error in errors], 'mean squarred error'))

        return [np.average([error]) for error in errors]

    def show_values(self, a_min=None):
        """
        :param a_min: minimal value of neuron to be considered True (value > amin) or False (value < -amin) (Float)
        :return: values of neurons in output layer (Dict)
        """
        if a_min is not None:
            return dict(zip(self.out.designates, [valuation(x, a_min) for x in self.out.output]))
        else:
            return dict(zip(self.out.designates, self.out.output))

    def show_model(self, a_min=None):
        """
        :param a_min: minimal value of neuron to be considered True (value > amin) or False (value < -amin) (Float)
        :return: values of neurons in output layer (Dict)
        """

        values = self.show_values(a_min)
        model = {'True': [],
                 'False': []}

        designates = list(values.keys())
        values = list(values.values())

        for i, value in enumerate(values):
            if value < 0:
                model['False'].append(designates[i])
            if value > 0:
                model['True'].append(designates[i])

        return model

