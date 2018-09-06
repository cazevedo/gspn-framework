import time
import numpy as np
import gspn_analysis


# TODO : add methods to remove arcs, places and transitions (removing places and trans should remove the corresponding input and output arcs as well)
# TODO: include arc firing with more than one token (for that change fire_transition and get_enabled_transitions)
class GSPN(object):
    """
    """
    def __init__(self):
        """

        """
        self.__places = {}
        self.__initial_marking = {}
        self.__transitions = {}
        self.__arc_in_m = []
        self.__arc_out_m = []

    def add_places(self, name, ntokens=None, set_initial_marking=True):
        """

        """
        if ntokens is None:
            ntokens = []
        else:
            ntokens.reverse()

        name.reverse()
        while name:
            if ntokens:
                self.__places[name.pop()] = ntokens.pop()
            else:
                self.__places[name.pop()] = 0

        if set_initial_marking:
            self.__initial_marking = self.__places.copy()

        return self.__places.copy()

    def add_transitions(self, tname, tclass=None, trate=None):
        """
        Input
        tname: list of strings, denoting the name of the transition
        type: list of strings, indicating if the corresponding transition is either immediate ('imm') or exponential ('exp')
        rate: list of floats, representing a static firing rate in an exponential transition and
                a static (non marking dependent) weight in a immediate transition
        Output
        dictionary of transitions
        """
        if tclass is None:
            tclass = []
        else:
            tclass.reverse()

        if trate is None:
            trate = []
        else:
            trate.reverse()

        tname.reverse()
        while tname:
            tn = tname.pop()
            self.__transitions[tn] = []
            if type:
                self.__transitions[tn].append(tclass.pop())
            else:
                self.__transitions[tn].append('imm')
            if trate:
                self.__transitions[tn].append(trate.pop())
            else:
                self.__transitions[tn].append(1.0)

        return self.__transitions.copy()

    def add_arcs_matrices(self, arc_in_m, arc_out_m):
        self.__arc_in_m = arc_in_m
        self.__arc_out_m = arc_out_m
        return True

    def add_arcs(self, arc_in, arc_out):
        """
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
        """
        self.__arc_in_m, self.__arc_out_m = GSPN.__create_arc_matrix(self.__places.copy(), self.__transitions.copy())

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
                self.__arc_out_m[temp[0].index(transition)][self.__arc_out_m[0].index(place)] = 1

        return self.__arc_in_m, self.__arc_out_m

    @staticmethod
    def __create_arc_matrix(places, transitions):
        # create a zeros matrix (# of places + 1) by (# of transitions)
        arc_in_m = []
        for i in range(len(places.keys())+1):
            arc_in_m.append([0]*(len(transitions.keys())))

        # replace the first row with all the transitions names
        arc_in_m[0] = list(transitions.keys())

        # add a first column with all the places names
        first_column = list(places.keys())
        first_column.insert(0, '')  # put None in the element (0,0) since it has no use
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

        return list(arc_in_m), list(arc_out_m)

    def add_tokens(self, place_name, ntokens, set_initial_marking=False):
        """
        add tokens to the specified places
        """
        if len(place_name) == len(ntokens):
            place_name.reverse()
            ntokens.reverse()
            while place_name:
                p = place_name.pop()
                self.__places[p] = self.__places[p] + ntokens.pop()

            if set_initial_marking:
                self.__initial_marking = self.__places.copy()

            return True
        else:
            return False

    def remove_tokens(self, place_name, ntokens, set_initial_marking=False):
        """
        remove tokens from the specified places
        """
        if len(place_name) == len(ntokens):
            place_name.reverse()
            ntokens.reverse()
            while place_name:
                p = place_name.pop()
                self.__places[p] = self.__places[p] - ntokens.pop()

            if set_initial_marking:
                self.__initial_marking = self.__places.copy()

            return True
        else:
            return False

    def get_current_marking(self):
        return self.__places.copy()

    def set_marking(self, places):
        self.__places = places.copy()
        return True

    def get_initial_marking(self):
        return self.__initial_marking.copy()

    def get_transitions(self):
        return self.__transitions.copy()

    def get_arcs(self):
        return list(self.__arc_in_m), list(self.__arc_out_m)

    def get_enabled_transitions(self):
        """
        returns a dictionary with the enabled transitions and the corresponding set of input places
        """
        enabled_exp_transitions = {}
        random_switch = {}
        arcs_in = list(zip(*self.__arc_in_m))
        current_marking = self.__places.copy()

        # for each transition get all the places that have an input arc connection
        for row_index in range(1, len(arcs_in)):
            places_in = []
            for column_index in range(1, len(arcs_in[row_index])):
                if arcs_in[row_index][column_index] > 0:
                    places_in.append(arcs_in[0][column_index])
            # print(arcs_in[row_index][0], places_in)

            # check if the transition in question is enabled or not (i.e. all the places that have an input arc to it
            #  have one or more tokens)
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

        return enabled_exp_transitions.copy(), random_switch.copy()

    def fire_transition(self, transition):
        index_transition = self.__arc_in_m[0].index(transition)
        arc_in_temp = list(zip(*self.__arc_in_m))

        # obtain a list with all the input places of given transition
        list_of_input_places = []
        for i in range(1, len(arc_in_temp[index_transition])):
            if arc_in_temp[index_transition][i] > 0:
                list_of_input_places.append(arc_in_temp[0][i])

        arc_out_temp = list(zip(*self.__arc_out_m))
        index_transition = arc_out_temp[0].index(transition)

        # obtain a list with all the output places of given transition
        list_of_output_places = []
        for k in range(1, len(self.__arc_out_m[index_transition])):
            if self.__arc_out_m[index_transition][k] > 0:
                list_of_output_places.append(self.__arc_out_m[0][k])

        # remove tokens from input places
        self.remove_tokens(list_of_input_places, [1]*len(list_of_input_places))

        # add tokens to output places
        self.add_tokens(list_of_output_places, [1]*len(list_of_output_places))

        return True

    def simulate(self, nsteps=1, reporting_step=1, simulate_wait=False):
        """
        """
        markings = []
        for step in range(nsteps):
            if step % reporting_step == 0:
                markings.append(self.get_current_marking())

            enabled_exp_transitions, random_switch = self.get_enabled_transitions()

            if random_switch:
                if len(random_switch) > 1:
                    s = sum(random_switch.values())
                    random_switch_id = []
                    random_switch_prob = []
                    # normalize the associated probabilities
                    for key, value in random_switch.items():
                        random_switch_id.append(key)
                        random_switch_prob.append(value/s)

                    # Draw from all enabled immediate transitions
                    firing_transition = np.random.choice(a=random_switch_id, size=None, p=random_switch_prob)
                    # Fire transition
                    self.fire_transition(firing_transition)
                else:
                    # Fire the only available immediate transition
                    self.fire_transition(list(random_switch.keys())[0])
            elif enabled_exp_transitions:
                if len(enabled_exp_transitions) > 1:
                    if simulate_wait:
                        wait_times = enabled_exp_transitions.copy()
                        # sample from each exponential distribution prob_dist(x) = lambda * exp(-lambda * x)
                        # in this case the beta rate parameter is used instead, where beta = 1/lambda
                        for key, value in enabled_exp_transitions.items():
                            wait_times[key] = np.random.exponential(scale=(1.0/value), size=None)

                        firing_transition = min(wait_times, key=wait_times.get)
                        wait = wait_times[firing_transition]
                        time.sleep(wait)

                    else:
                        s = sum(enabled_exp_transitions.values())
                        exp_trans_id = []
                        exp_trans_prob = []
                        # normalize the associated probabilities
                        for key, value in enabled_exp_transitions.items():
                            exp_trans_id.append(key)
                            exp_trans_prob.append(value / s)

                        # Draw from all enabled exponential transitions
                        firing_transition = np.random.choice(a=exp_trans_id, size=None, p=exp_trans_prob)

                    # Fire transition
                    self.fire_transition(firing_transition)
                else:
                    if simulate_wait:
                        wait = np.random.exponential(scale=(1.0 / list(enabled_exp_transitions.values())[0]), size=None)
                        time.sleep(wait)

                    # Fire transition
                    self.fire_transition(list(enabled_exp_transitions.keys())[0])

        return list(markings)

    def reset_simulation(self):
        self.__places = self.__initial_marking.copy()
        return True

    # def prob_having_n_tokens(self, place_id, ntokens):
    #     '''
    #     Computes the probability of having exactly k tokens in a place pi.
    #     :param place_id: identifier of the place for which the probability
    #     :param ntokens: number of tokens
    #     :return:
    #     '''

    # TODO: add throughput for immediate transitions as well
    def transition_throughput_rate(self, exp_transition):
        if self.__transitions[exp_transition][0] != 'exp':
            raise Exception('Transition is not exponential, throughput rate only available for exponential transitions.')

        ct_tree = gspn_analysis.CoverabilityTree(self)
        ct_tree.generate()
        ctmc = gspn_analysis.CTMC(ct_tree)
        ctmc.generate()
        ctmc.compute_transition_rate()
        ctmc_steady_state = ctmc.get_steady_state()

        transition_rate = self.__transitions[exp_transition]
        transition_rate = transition_rate[1]
        states_already_considered = []
        throughput_rate = 0
        for tr in ctmc.transition:
            state = tr[0]
            transitons_id = tr[2]
            transitons_id = transitons_id.replace('/',':')
            transition_id = transitons_id.split(':')
            if (exp_transition in transition_id) and not (state in states_already_considered):
                throughput_rate = throughput_rate + ctmc_steady_state[state] * transition_rate

                states_already_considered.append(state)

        return throughput_rate

    # def expected_number_of_tokens(self, place):

    def prob_of_n_tokens(self, place, ntokens):
        ct_tree = gspn_analysis.CoverabilityTree(self)
        ct_tree.generate()
        ctmc = gspn_analysis.CTMC(ct_tree)
        ctmc.generate()
        ctmc.compute_transition_rate()
        ctmc_steady_state = ctmc.get_steady_state()

        prob_of_n_tokens = 0
        for state_id, marking in ctmc.state.items():
            marking = marking[0]
            for pl in marking:
                if (place == pl[0]) and (ntokens == pl[1]):
                    prob_of_n_tokens = prob_of_n_tokens + ctmc_steady_state[state_id]

        return prob_of_n_tokens

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
