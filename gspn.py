
class gspn(object):
    '''

    '''
    def __init__(self):
        '''
        Class constructor: will get executed at the moment
        of object creation
        '''
        self.places = {}
        self.transitions = {}
        self.in_transitions = []

    def add_places(self, name, ntokens=[]):
        '''

        '''
        name.reverse()
        ntokens.reverse()
        while name:
            if ntokens:
                self.places[name.pop()].append(ntokens.pop())
            else:
                self.places[name.pop()].append(0)

        return self.places

    def add_transitions(self, name, type=[], rate=[]):
        '''
        name: list of strings, denoting the name of the transition
        type: list of strings, indicating if the corresponding transition is either immediate ('imm') or exponential ('exp')
        rate: list of floats, representing a static firing rate in an exponential transition and
                a static (non marking dependent) weight in a immediate transition
        '''
        name.reverse()
        type.reverse()
        rate.reverse()
        while name:
            tn = name.pop()
            self.transitions[tn] = []
            if type:
                self.transitions[tn].append(type.pop())
            else:
                self.transitions[tn].append('imm')
            if rate:
                self.transitions[tn].append(rate.pop())
            else:
                self.transitions[tn].append(1.0)

        return self.transitions

    def add_in_connections(self):
        '''
        PxT represents the arc connections from places to transitions
        '''

        return True

    def add_out_connections(self):
        '''
        TxP represents the arc connections from transitions to places
        '''

        return True

    def add_tokens(self, place, ntokens):
        '''
        set initial marking
        '''

        return True

    def execute(self, steps):
        '''

        '''

        return True


if __name__ == '__main__':
    # create a generalized stochastic petri net structure
    my_pn = gspn()
    trans = my_pn.add_transitions(['1', '2', '3', '4'], ['exp', 'imm', 'exp'], [0.5, 0.1])
    print(trans)
    # my_pn.add_places()