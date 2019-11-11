from urllib.request import urlopen, Request


def abd_clause(logic_program, abductive_goal, parameters):
    """
    :param logic_program: Logic Program (Str)
    :param abductive_goal: Clause as goal of abduction (Str)
    :param parameters: List with parameters for request (List)
    :return: Formatted request phrase (Str)
    """
    parameters_out = "[" + ",".join(map(str, parameters)) + "]"
    return "\n".join(logic_program + abductive_goal + parameters_out)


def abd_atoms(logic_program, atoms_goal, parameters):
    """
    :param logic_program: Logic Program (Str)
    :param atoms_goal: Atoms as goal of abduction (Str)
    :param parameters: List with parameters for request (List)
    :return: Formatted request phrase (Str)
    """
    parameters_out = "[" + ",".join(map(str, parameters)) + "]"
    atoms_goal_out = "[" + ",".join(map(str, atoms_goal)) + "]"
    return "\n".join(logic_program + atoms_goal_out + parameters_out)


def nn2lp(input_order, output_order, amin, pairs):
    """
    :param input_order: Order of neurons in input layer (List of Str)
    :param output_order: Order of neurons in output layer (List of Str)
    :param amin: minimal value of neuron to be considered True (value > amin) or False (value < -amin) (Float)
    :param pairs: Pairs of output and input layer values (List of tuples)
    :return: Formatted request phrase (Str)
    """
    return '\n'.join(map(str, [input_order, output_order, amin, pairs]))


def connect(f, phrase, url='http://207.154.220.61:10099/api/'):
    """
    :param f: to which function you want to connect (Str)
    :param phrase: request phrase (Str)
    :param url: url of server (Str)
    :return: response (Str)

    Opens url using Request library
    """
    request = Request(url+f, phrase.encode("utf-8"))
    response = urlopen(request)
    html = response.read()
    response.close()
    return html.decode("utf-8")


"""
sample_lp = "[Cl {clHead = A {idx = 2, label = []}, clPAtoms = [A {idx = 1, label = []}], " \
            "clNAtoms = [A {idx = 4, label = []}]}, Cl {clHead = A {idx = 1, label = []}, " \
            "clPAtoms = [A {idx = 3, label = []}], clNAtoms = []}, Fact {clHead = A {idx = 5, label = []}}]"
"""
