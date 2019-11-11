from NSS_NN.toolbox import with_quotes


def read_interpretations(filename):
    """
    :param filename: path to file with interpretations (Str)
    :return: evaluated raw code of interpretations (List)
    """
    with open(filename) as file:
        return [Interpretation(spec) for spec in eval(with_quotes(file.read()))]


def by_vector(interpretation_dict, nn_order):
    """
    :param interpretation_dict: Interpretation.input or Intepretation.output (dict)
    :param nn_order: Order of neurons labels in layer (List of Strings)
    :return: list with values as values for Label.predict or Label.stabilization (List of Floats)

    Creates vector of values from intepretation
    """
    nn_order = list(nn_order)
    value_vector = dict(zip(nn_order, [0 for i in nn_order]))

    for true_neuron in interpretation_dict['True']:
        value_vector[true_neuron] = 1

    for false_neuron in interpretation_dict['False']:
        value_vector[false_neuron] = -1

    return list(value_vector.values())


class Interpretation:
    def __init__(self, specification):
        """
        :param specification: Tuple of Tuples of Lists
            ( ( [ A1 ] , [ A4,A6,A7,A5,A2,A3 ] ) , ( [ A6,A5 ] , [ A1,A4,A7,A2,A3 ] ) ) <-- specification
              ( [ A1 ] , [ A4,A6,A7,A5,A2,A3 ] )   ( [ A6,A5 ] , [ A1,A4,A7,A2,A3 ] )   <-- input and output
                [ A1 ]   [ A4,A6,A7,A5,A2,A3 ]       [ A6,A5 ]  [ A1,A4,A7,A2,A3 ]      <-- true and false neurons
        """

        self.specification = specification
        self.input = {'True': specification[0][0],
                      'False': specification[0][1]}
        self.output = {'True': specification[1][0],
                       'False': specification[1][1]}

    def __getitem__(self, io):
        if io == 'input':
            return self.input
        elif io == 'output':
            return self.output


