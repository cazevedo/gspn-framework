# import time
import gspn_tools as gst

class gspn(object):
    '''
    # TODO: include arc firing with more than one token
    '''
    def __init__(self):
        '''

        '''
        self.__places = {}
        self.__transitions = {}
        # self.arc_in = {}
        # self.arc_out = {}
        self.__arc_in_m = []
        self.__arc_out_m  = []

    def add_places(self, name, ntokens=[]):
        '''

        '''
        name.reverse()
        ntokens.reverse()
        while name:
            if ntokens:
                self.__places[name.pop()] = ntokens.pop()
            else:
                self.__places[name.pop()] = 0

        return self.__places

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
            self.__transitions[tn] = []
            if type:
                self.__transitions[tn].append(type.pop())
            else:
                self.__transitions[tn].append('imm')
            if rate:
                self.__transitions[tn].append(rate.pop())
            else:
                self.__transitions[tn].append(1.0)

        return self.__transitions

    def add_arcs_matrices(self, arc_in_m, arc_out_m):
        self.__arc_in_m = arc_in_m
        self.__arc_out_m  = arc_out_m
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

        self.__arc_in_m, self.__arc_out_m  = self.__CreateArcMatrix(self.__places, self.__transitions)

        # IN ARCS MATRIX
        # replace the zeros by ones in the positions where there is an arc connection from a place to a transition
        temp = list(zip(*self.__arc_in_m))
        for place, target in arc_in.items():
            for transition in target:
                self.__arc_in_m[temp[0].index(place)][self.__arc_in_m[0].index(transition)] = 1

        # OUT ARCS MATRIX
        # replace the zeros by ones in the positions where there is an arc connection from a transition to a place
        temp = list(zip(*self.__arc_out_m ))
        for transition, target in arc_out.items():
            for place in target:
                self.__arc_out_m [temp[0].index(transition)][self.__arc_out_m [0].index(place)] = 1


        return self.__arc_in_m, self.__arc_out_m

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
                self.__places[p] = self.__places[p] + ntokens.pop()

            return True
        else:
            return False

    def get_current_marking(self):
        return self.__places

    def get_transitions(self):
        return self.__transitions

    def get_arcs(self):
        return self.__arc_in_m, self.__arc_out_m

    def get_enabled_transitions(self):
        dict_enabled_transitions = {}
        arcs_in = list(zip(*self.__arc_in_m))
        current_marking = self.__places

        # for each transition get all the places that have an input arc connection
        for row_index in range(1, len(arcs_in)):
            places_in = []
            for column_index in range(1, len(arcs_in[row_index])):
                if arcs_in[row_index][column_index] > 0:
                    places_in.append(arcs_in[0][column_index])
            # print(arcs_in[row_index][0], places_in)

            # check if the transition in question is enabled or not (i.e. all the places that have an input arc to it have one or more tokens)
            enabled_transition = True
            for place in places_in:
                if current_marking.get(place) == 0:
                    enabled_transition = False

            if enabled_transition:
                dict_enabled_transitions[arcs_in[row_index][0]] = places_in

        return dict_enabled_transitions

    def simulate(self, steps):
        # in this case the transition firing is sampled from the temporal distribution and the method actually waits
        # for the time to elapse before firing
        return True

    def execute(self, steps=1):
        '''

        '''
        conflicting_transitions = {}
        enabled_transitions = self.get_enabled_transitions()

        temp = enabled_transitions.copy()
        # check which transitions are in conflict
        for curr_transition in enabled_transitions.keys():
            curr_place_list = temp.pop(curr_transition)
            conflicting_transitions[curr_transition] = []
            for tr, pl_lst in temp.items():
                for pl in pl_lst:
                    if pl in curr_place_list:
                        conflicting_transitions[curr_transition].append(tr)

            # if the list is empty (i.e. there is no conflict with any other transition, just remove the dict entry
            if not conflicting_transitions[curr_transition]:
                del conflicting_transitions[curr_transition]




        # from conflicting enabled transitions delete the ones that are in conflict with immediate ones and are not immediate

        # sample from the probability distribution the ones in conflict and fire the one that was drawn

        # the ones that are not in conflict just fire them and save the sampled times

        # check the arcs out and place the tokens from the fired transitions in the output places

        # return True

    # def __column(matrix, i):
    #     return [row[i] for row in matrix]

if __name__ == "__main__":
    # create a generalized stochastic petri net structure
    # my_pn = gspn()
    # places = my_pn.add_places(['p1', 'p2', 'p3', 'p4', 'p5'], [1, 0, 1, 0, 1])
    # # places = my_pn.add_places(['p1', 'p2', 'p3', 'p4', 'p5'])
    # trans = my_pn.add_transitions(['t1', 't2', 't3', 't4'], ['exp', 'exp', 'exp', 'exp'], [1, 1, 0.5, 0.5])
    #
    # arc_in = {}
    # arc_in['p1'] = ['t1']
    # arc_in['p2'] = ['t2']
    # arc_in['p3'] = ['t3']
    # arc_in['p4'] = ['t4']
    # arc_in['p5'] = ['t1', 't3']
    #
    # arc_out = {}
    # arc_out['t1'] = ['p2']
    # arc_out['t2'] = ['p5', 'p1']
    # arc_out['t3'] = ['p4']
    # arc_out['t4'] = ['p3', 'p5']
    # a, b = my_pn.add_arcs(arc_in ,arc_out)
    #
    # print(my_pn.get_enabled_transitions())
    # a = my_pn.get_enabled_transitions()

    # print('Places: ' , my_pn.get_current_marking(), '\n')
    # print('Trans: ' , my_pn.get_transitions(), '\n')
    # print('Arcs IN: ' , my_pn.get_in_arcs(), '\n')
    # print('Arcs OUT: ' , my_pn.get_out_arcs(), '\n')z
    #
    # print(my_pn.add_tokens(['p1', 'p3', 'p5'], [10,5,1]))
    #
    # print('Places: ', my_pn.get_current_marking(), '\n')

    parset = gst.gspn_tools()
    a = parset.import_pnml('debug/pipediag.xml')
    pn = a[0]

    z = pn.get_enabled_transitions()
    # print(z)

    t = pn.execute(1)