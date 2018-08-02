
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
        self.arc_in = {}
        self.arc_out = {}

    def add_places(self, name, ntokens=[]):
        '''

        '''
        name.reverse()
        ntokens.reverse()
        while name:
            if ntokens:
                self.places[name.pop()] = ntokens.pop()
            else:
                self.places[name.pop()] = 0

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

    def add_arc_in(self, arc):
        '''
        PxT represents the arc connections from places to transitions
        '''
        # check how to create a matrix!
        self.arc_in = arc

        return True

    def add_arc_out(self, arc):
        '''
        TxP represents the arc connections from transitions to places
        '''
        self.arc_out = arc

        return True

    def add_tokens(self, place_name, ntokens):
        '''
        add tokens to the current marking
        '''
        if len(place_name) == len(ntokens):
            place_name.reverse()
            ntokens.reverse()
            while place_name:
                p = place_name.pop()
                self.places[p] = self.places[p] + ntokens.pop()

            return True
        else:
            return False

    def get_current_marking(self):
        return self.places

    def get_transitions(self):
        return self.transitions

    def get_in_arcs(self):
        return self.arc_in

    def get_out_arcs(self):
        return self.arc_out

    def execute(self, steps):
        '''

        '''

        return True


if __name__ == '__main__':
    # create a generalized stochastic petri net structure
    my_pn = gspn()
    # places = my_pn.add_places(['p1', 'p2', 'p3', 'p4', 'p5'], [1, 0, 1, 0, 1])
    places = my_pn.add_places(['p1', 'p2', 'p3', 'p4', 'p5'])
    trans = my_pn.add_transitions(['t1', 't2', 't3', 't4'], ['exp', 'exp', 'exp', 'exp'], [1, 1, 0.5, 0.5])
    arc_in = {}
    arc_in['p1'] = ['t1']
    arc_in['p2'] = ['t2']
    arc_in['p3'] = ['t3']
    arc_in['p4'] = ['t4']
    arc_in['p5'] = ['t1', 't3']
    my_pn.add_arc_in(arc_in)

    arc_out = {}
    arc_out['t1'] = ['p2']
    arc_out['t2'] = ['p5', 'p1']
    arc_out['t3'] = ['p4']
    arc_out['t4'] = ['p3', 'p5']
    my_pn.add_arc_out(arc_out)

    print('Places: ' , my_pn.get_current_marking(), '\n')
    print('Trans: ' , my_pn.get_transitions(), '\n')
    print('Arcs IN: ' , my_pn.get_in_arcs(), '\n')
    print('Arcs OUT: ' , my_pn.get_out_arcs(), '\n')

    print(my_pn.add_tokens(['p1', 'p3', 'p5'], [10,5,1]))

    print('Places: ', my_pn.get_current_marking(), '\n')