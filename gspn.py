# import time
import numpy as np

class gspn(object):
    '''
    # TODO: include arc firing with more than one token (for that change fire_transition and get_enabled_transitions)
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
        temp = list(zip(*self.__arc_out_m))
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
        first_column.insert(0, '') # put None in the element (0,0) since it has no use
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
        first_column.insert(0, '')  # put None in the element (0,0) since it has no use
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

    def remove_tokens(self, place_name, ntokens):
        '''
        add tokens to the current marking
        '''
        if len(place_name) == len(ntokens):
            place_name.reverse()
            ntokens.reverse()
            while place_name:
                p = place_name.pop()
                self.__places[p] = self.__places[p] - ntokens.pop()

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
        '''
        returns a dictionary with the enabled transitions and the corresponding set of input places
        '''
        enabled_exp_transitions = {}
        random_switch = {}
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
                transition = arcs_in[row_index][0]
                if self.__transitions[transition][0] == 'exp':
                    enabled_exp_transitions[transition] = self.__transitions[transition][1]
                    # enabled_exp_transitions.add(transition)
                else:
                    random_switch[transition] = self.__transitions[transition][1]
                    # random_switch.add(transition)

        return enabled_exp_transitions, random_switch

    # NOT TESTED
    # def get_conflicting_transitions(self):
    #     conflicting_transitions = []
    #     enabled_transitions = self.get_enabled_transitions()
    #
    #     temp = enabled_transitions.copy()
    #     # check which transitions are in conflict
    #     for curr_transition in enabled_transitions.keys():
    #         curr_place_list = temp.pop(curr_transition)
    #         curr_trans_conflicts = {curr_transition}
    #
    #         # print('LAST CONFL: ', conflicting_transitions)
    #         # print('CURRENT TRANS : ', curr_transition)
    #
    #         for tr, pl_lst in temp.items():
    #             for pl in pl_lst:
    #                 if pl in curr_place_list:
    #                     curr_trans_conflicts.add(tr)
    #
    #         # print('CURRENT : ', curr_trans_conflicts)
    #
    #         # if the list is empty (i.e. there is no conflict with any other transition, just remove the dict entry
    #         if len(curr_trans_conflicts) > 1:
    #             inexistent_in_list = True
    #             for conflict in conflicting_transitions:
    #                 if curr_trans_conflicts.issubset(conflict):
    #                     inexistent_in_list = False
    #                     break
    #
    #             if inexistent_in_list:
    #                 conflicting_transitions.append(curr_trans_conflicts)
    #
    #         return conflicting_transitions

    def simulate(self, steps):
        # in this case the transition firing is sampled from the temporal distribution and the method actually waits
        # for the time to elapse before firing
        return True

    def fire_transition(self, transition):
        index_transition = self.__arc_in_m[0].index(transition)
        arc_in_temp = list(zip(*self.__arc_in_m))

        # obtain a list with all the input places of given transition
        list_of_input_places = []
        for i in range(1,len(arc_in_temp[index_transition])):
            if arc_in_temp[index_transition][i] > 0:
                list_of_input_places.append(arc_in_temp[0][i])

        arc_out_temp = list(zip(*self.__arc_out_m))
        index_transition = arc_out_temp[0].index(transition)

        # obtain a list with all the output places of given transition
        list_of_output_places = []
        for k in range(1,len(self.__arc_out_m[index_transition])):
            if self.__arc_out_m[index_transition][k] > 0:
                list_of_output_places.append(self.__arc_out_m[0][k])

        # remove tokens from input places
        self.remove_tokens(list_of_input_places, [1]*len(list_of_input_places))

        # add tokens to output places
        self.add_tokens(list_of_output_places, [1]*len(list_of_output_places))

        return True

    def execute(self, nsteps=1, reporting_step=1):
        '''

        '''
        markings = []
        for step in range(nsteps):
            if (step%reporting_step == 0):
                markings.append(self.get_current_marking())

            enabled_exp_transitions, random_switch = self.get_enabled_transitions()

            if random_switch:
                if len(random_switch) > 1:
                    # normalize the associated probabilities
                    random_switch_prob = list(np.array(list(random_switch.values()), dtype='f')/sum(random_switch.values()))
                    # Draw from all enabled immediate transitions
                    firing_transition = np.random.choice(list(random_switch.keys()), 1, random_switch_prob)
                    # Fire transition
                    self.fire_transition(firing_transition[0])
                else:
                    # Fire the only immediate available transition
                    self.fire_transition(random_switch.keys()[0])
            elif enabled_exp_transitions:
                if len(enabled_exp_transitions) > 1:
                    # normalize the associated probabilities
                    exp_trans_prob = list(np.array(list(enabled_exp_transitions.values()), dtype='f')/sum(enabled_exp_transitions.values()))
                    # print(exp_trans_prob)
                    # Draw from all enabled exponential transitions
                    firing_transition = np.random.choice(list(enabled_exp_transitions.keys()), 1, exp_trans_prob)
                    # Fire transition
                    self.fire_transition(firing_transition[0])
                else:
                    # Fire that transition
                    self.fire_transition(enabled_exp_transitions.keys()[0])

        return markings

    # def __column(matrix, i):
    #     return [row[i] for row in matrix]

# if __name__ == "__main__":
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