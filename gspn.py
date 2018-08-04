# import time

class gspn(object):
    '''

    '''
    def __init__(self):
        '''

        '''
        self.places = {}
        self.transitions = {}
        # self.arc_in = {}
        # self.arc_out = {}
        self.arc_in_m = []
        self.arc_out_m = []

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
        Input
        name: list of strings, denoting the name of the transition
        type: list of strings, indicating if the corresponding transition is either immediate ('imm') or exponential ('exp')
        rate: list of floats, representing a static firing rate in an exponential transition and
                a static (non marking dependent) weight in a immediate transition
        Output
        dictionary of transitions
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

    def add_arcs_matrices(self, arc_in_m, arc_out_m):
        self.arc_in_m = arc_in_m
        self.arc_out_m = arc_out_m
        return True

    def add_arcs(self, arc_in, arc_out):
        '''
        Input:
        arc_in -> dictionary mapping the arc connections from places to transitions
        arc_out -> dictionary mapping the arc connections from transitions to places

        example:
        arc_in = {}
        arc_in['p1'] = ['t1']
        arc_in['p2'] = ['t2']
        arc_in['p3'] = ['t3']
        arc_in['p4'] = ['t4']
        arc_in['p5'] = ['t1', 't3']

        arc_out = {}
        arc_out['t1'] = ['p2']
        arc_out['t2'] = ['p5', 'p1']
        arc_out['t3'] = ['p4']
        arc_out['t4'] = ['p3', 'p5']

        Output:
        arc_in_m -> two-dimentional list
        arc_out_m -> two-dimentional list
        '''

        # self.arc_in = arc_in
        # self.arc_out = arc_out

        self.arc_in_m, self.arc_out_m = self.__CreateArcMatrix(self.places, self.transitions)

        # IN ARCS MATRIX
        # replace the zeros by ones in the positions where there is an arc connection from a place to a transition
        temp = list(zip(*self.arc_in_m))
        for place, target in arc_in.items():
            for transition in target:
                self.arc_in_m[temp[0].index(place)][self.arc_in_m[0].index(transition)] = 1

        # OUT ARCS MATRIX
        # replace the zeros by ones in the positions where there is an arc connection from a transition to a place
        temp = list(zip(*self.arc_out_m))
        for transition, target in arc_out.items():
            for place in target:
                self.arc_out_m[temp[0].index(transition)][self.arc_out_m[0].index(place)] = 1


        return self.arc_in_m, self.arc_out_m

    def __CreateArcMatrix(self, places, transitions):
        # create a zeros matrix (# of places + 1) by (# of transitions)
        arc_in_m = []
        for i in range(len(places.keys())+1):
            arc_in_m.append([0]*(len(transitions.keys())))

        # replace the first row with all the transitions names
        arc_in_m[0] = list(transitions.keys())

        # add a first column with all the places names
        first_column = list(places.keys())
        first_column.insert(0, None) # put None in the element (0,0) since it has no use
        arc_in_m = list(zip(*arc_in_m))
        arc_in_m.insert(0, first_column)
        arc_in_m = list(map(list, zip(*arc_in_m)))

        # create a zeros matrix (# of transitions + 1) by (# of places)
        arc_out_m = []
        for i in range(len(transitions.keys())+1):
            arc_out_m.append([0]*(len(places.keys())))

        # replace the first row with all the places names
        arc_out_m[0] = list(places.keys())

        # add a first column with all the transitions names
        first_column = list(transitions.keys())
        first_column.insert(0, None)  # put None in the element (0,0) since it has no use
        arc_out_m = list(zip(*arc_out_m))
        arc_out_m.insert(0, first_column)
        arc_out_m = list(map(list, zip(*arc_out_m)))

        return arc_in_m, arc_out_m


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

    def get_arcs(self):
        return self.arc_in_m, self.arc_out_m

    def execute(self, steps):
        '''

        '''

        # for place, token in self.places.items():


        return True

    # def __column(matrix, i):
    #     return [row[i] for row in matrix]

if __name__ == "__main__":
    # create a generalized stochastic petri net structure
    my_pn = gspn()
    places = my_pn.add_places(['p1', 'p2', 'p3', 'p4', 'p5'], [1, 0, 1, 0, 1])
    # places = my_pn.add_places(['p1', 'p2', 'p3', 'p4', 'p5'])
    trans = my_pn.add_transitions(['t1', 't2', 't3', 't4'], ['exp', 'exp', 'exp', 'exp'], [1, 1, 0.5, 0.5])

    arc_in = {}
    arc_in['p1'] = ['t1']
    arc_in['p2'] = ['t2']
    arc_in['p3'] = ['t3']
    arc_in['p4'] = ['t4']
    arc_in['p5'] = ['t1', 't3']

    arc_out = {}
    arc_out['t1'] = ['p2']
    arc_out['t2'] = ['p5', 'p1']
    arc_out['t3'] = ['p4']
    arc_out['t4'] = ['p3', 'p5']
    a, b = my_pn.add_arcs(arc_in ,arc_out)

    # print('Places: ' , my_pn.get_current_marking(), '\n')
    # print('Trans: ' , my_pn.get_transitions(), '\n')
    # print('Arcs IN: ' , my_pn.get_in_arcs(), '\n')
    # print('Arcs OUT: ' , my_pn.get_out_arcs(), '\n')
    #
    # print(my_pn.add_tokens(['p1', 'p3', 'p5'], [10,5,1]))
    #
    # print('Places: ', my_pn.get_current_marking(), '\n')

    # parset = pnml_tools()
    # mm = parset.import_pnml('pipediag.xml')
    # print(mm)

    # parset = pnml_tools()
    # parset.show_gspn()

    # nets = pn.parse_pnml_file('example.pnml')
    # # print(nets)
    # for net in nets:
    #     print(nets.pop())
    #     print('------------')