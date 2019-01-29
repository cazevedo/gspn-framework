import time
import numpy as np
import gspn_analysis
import gspn_tools
import pandas as pd


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
        self.__new_arc_in_m = pd.DataFrame()
        self.__new_arc_out_m = pd.DataFrame()
        self.__ct_tree = None
        self.__ctmc = None
        self.__ctmc_steady_state = None
        self.__ct_ctmc_generated = False
        self.__nsamples = {}
        self.__sum_samples = {}

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

    def add_places_dict(self, places_dict, set_initial_marking=True):
        self.__places.update(places_dict.copy())

        if set_initial_marking:
            self.__initial_marking = self.__places.copy()

        return self.__places.copy()

    def add_transitions(self, tname, tclass=None, trate=None):
        """
        Input
        tname: list of strings, denoting the name of the transition
        tclass: list of strings, indicating if the corresponding transition is either immediate ('imm') or exponential ('exp')
        trate: list of floats, representing a static firing rate in an exponential transition and
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
            if tclass:
                self.__transitions[tn].append(tclass.pop())
            else:
                self.__transitions[tn].append('imm')
            if trate:
                self.__transitions[tn].append(trate.pop())
            else:
                self.__transitions[tn].append(1.0)

        return self.__transitions.copy()

    def add_transitions_dict(self, transitions_dict):
        self.__transitions.update(transitions_dict.copy())
        return self.__transitions.copy()

    def add_arcs_matrices(self, new_arc_in, new_arc_out):
        self.__new_arc_in_m = new_arc_in
        self.__new_arc_out_m = new_arc_out
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
        arc_in_m -> Pandas DataFrame where the columns hold the transition names and the index the place names
        arc_out_m -> Pandas DataFrame where the columns hold the place names and the index the transition names
        Each element of the DataFrame preserves the information regarding if there is a connecting arc (value equal to 1)
        or if there is no connecting arc (value equal to 0)
        """

        # store existing arc dataframes
        old_arc_in, old_arc_out = self.__new_arc_in_m, self.__new_arc_out_m

        # create new empty (no arc connections) dataframe with existing places and transitions
        pl_len, tr_len = len(self.__places.keys()), len(self.__transitions.keys())
        new_arc_in = pd.DataFrame(np.zeros((pl_len, tr_len)))
        new_arc_out = pd.DataFrame(np.zeros((tr_len, pl_len)))

        new_arc_in.columns, new_arc_in.index = self.__transitions.keys(), self.__places.keys()
        new_arc_out.columns, new_arc_out.index = self.__places.keys(), self.__transitions.keys()

        # add to the new dataframe the connections in the old dataframe
        for old_place in old_arc_in.index:
            for old_transition in old_arc_in.columns:
                if old_arc_in.loc[old_place][old_transition] > 0:
                    if (old_place in new_arc_in.index) and (old_transition in new_arc_in.columns):
                        new_arc_in.loc[old_place][old_transition] = 1

        for old_place in old_arc_out.columns:
            for old_transition in old_arc_out.index:
                if old_arc_out.loc[old_transition][old_place] > 0:
                    if (old_place in new_arc_in.index) and (old_transition in new_arc_in.columns):
                        new_arc_out.loc[old_transition][old_place] = 1

        for place, target in arc_in.items():
            for transition in target:
                new_arc_in.loc[place][transition] = 1

        for transition, target in arc_out.items():
            for place in target:
                new_arc_out.loc[transition][place] = 1

        self.__new_arc_in_m = new_arc_in
        self.__new_arc_out_m = new_arc_out

        return self.__new_arc_in_m.copy(), self.__new_arc_out_m.copy()

    def add_tokens(self, place_name, ntokens, set_initial_marking=False):
        """
        add tokens to the specified places
        """
        if len(place_name) == len(ntokens):
            place_name.reverse()
            ntokens.reverse()
            while place_name:
                p = place_name.pop()
                tokens_to_add = ntokens.pop()
                if self.__places[p] != 'w':
                    if tokens_to_add == 'w':
                        self.__places[p] = 'w'
                    else:
                        self.__places[p] = self.__places[p] + tokens_to_add

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
                tokens_to_remove = ntokens.pop()
                if tokens_to_remove == 'w':
                    self.__places[p] = 0
                else:
                    if self.__places[p] != 'w':
                        self.__places[p] = self.__places[p] - tokens_to_remove

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
        return self.__new_arc_in_m.copy(), self.__new_arc_out_m.copy()

    def get_arcs_dict(self):
        '''
        Converts the arcs DataFrames to dicts and outputs them.
        :return: arcs in dict form
        '''
        arcs_in = {}
        for place in self.__new_arc_in_m.index:
            for transition in self.__new_arc_in_m.columns:
                if self.__new_arc_in_m.loc[place][transition] > 0:
                    if place in arcs_in:
                        arcs_in[place].append(transition)
                    else:
                        arcs_in[place] = [transition]

        arcs_out = {}
        for place in self.__new_arc_out_m.columns:
            for transition in self.__new_arc_out_m.index:
                if self.__new_arc_out_m.loc[transition][place] > 0:
                    if transition in arcs_out:
                        arcs_out[transition].append(place)
                    else:
                        arcs_out[transition] = [place]

        return arcs_in, arcs_out

    def get_enabled_transitions(self):
        """
        returns a dictionary with the enabled transitions and the corresponding set of input places
        """
        enabled_exp_transitions = {}
        random_switch = {}
        current_marking = self.__places.copy()

        # for each transition get all the places that have an input arc connection
        for transition in self.__new_arc_in_m.columns:
            idd = self.__new_arc_in_m.loc[:][transition].values > 0 # true/false list stating if there is a connection or not
            places_in = self.__new_arc_in_m.index[idd].values # list of input places of the transition in question

            # check if the transition in question is enabled or not (i.e. all the places that have an input arc to it
            #  have one or more tokens)
            enabled_transition = True
            for place in places_in:
                if current_marking.get(place) == 0:
                    enabled_transition = False

            if enabled_transition:
                if self.__transitions[transition][0] == 'exp':
                    enabled_exp_transitions[transition] = self.__transitions[transition][1]
                    # enabled_exp_transitions.add(transition)
                else:
                    random_switch[transition] = self.__transitions[transition][1]
                    # random_switch.add(transition)

        return enabled_exp_transitions.copy(), random_switch.copy()

    def fire_transition(self, transition):
        # true/false list stating if there is an input connection or not
        idd = self.__new_arc_in_m.loc[:][transition].values > 0
        # list with all the input places of the given transition
        list_of_input_places = list(self.__new_arc_in_m.index[idd].values)

        # true/false list stating if there is an output connection or not
        idd = self.__new_arc_out_m.loc[transition][:].values > 0
        # list with all the output places of the given transition
        list_of_output_places = list(self.__new_arc_out_m.columns[idd].values)

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

    def reset(self):
        self.__places = self.__initial_marking.copy()
        self.__nsamples = {}
        self.__sum_samples = {}
        return True

    # def prob_having_n_tokens(self, place_id, ntokens):
    #     '''
    #     Computes the probability of having exactly k tokens in a place pi.
    #     :param place_id: identifier of the place for which the probability
    #     :param ntokens: number of tokens
    #     :return:
    #     '''

    def init_analysis(self):
        self.__ct_tree = gspn_analysis.CoverabilityTree(self)
        self.__ct_tree.generate()
        self.__ctmc = gspn_analysis.CTMC(self.__ct_tree)
        self.__ctmc.generate()
        self.__ctmc.compute_transition_rate()
        self.__ctmc_steady_state = self.__ctmc.get_steady_state()

        self.__ct_ctmc_generated = True

        return True

    def transition_throughput_rate(self, transition):
        '''
        The throughput of an exponential transition tj is computed by considering its firing rate over the probability
        of all states where tj is enabled. The throughput of an immediate transition tj can be computed by considering
        the throughput of all exponential transitions which lead immediately to the firing of transition tj, i.e.,
        without crossing any tangible state, together with the probability of firing transition tj among all the
        enabled immediate transitions.
        :param transition: (string) with the transition id for which the throughput rate will be computed
        :return: (float) with the computed throughput rate
        '''

        if not self.__ct_ctmc_generated:
            raise Exception('Analysis must be initialized before this method can be used, please use init_analysis() method for that purpose.')

        if self.__transitions[transition][0] == 'exp':
            transition_rate = self.__transitions[transition]
            transition_rate = transition_rate[1]
            states_already_considered = []
            throughput_rate = 0
            for tr in self.__ctmc.transition:
                state = tr[0]
                transiton_id = tr[2]
                transiton_id = transiton_id.replace('/',':')
                transiton_id = transiton_id.split(':')
                if (transition in transiton_id) and not (state in states_already_considered):
                    throughput_rate = throughput_rate + self.__ctmc_steady_state[state] * transition_rate

                    states_already_considered.append(state)
        else:
            throughput_rate = 0
            states_already_considered = []
            for tr in self.__ctmc.transition:
                add_state = False
                tangible_init_state = tr[0]

                transitons_id = tr[2]
                transition_id_set = transitons_id.split('/')
                for tr in transition_id_set:

                    # check if transition exists in the current transition
                    exists_transition = False
                    transitioning_list = tr.split(':')
                    for trn in transitioning_list:
                        if transition == trn:
                            exists_transition = True
                            add_state = True
                            break

                    # if the given transition is part of this ctmc edge, multiply the throughput rate of the exponential transition by the prob of immediate transition
                    if exists_transition and not(tangible_init_state in states_already_considered):
                        exp_transition = transitioning_list[0]
                        current_state = tangible_init_state
                        for trans in transitioning_list:
                            current_transition = trans
                            for edge in self.__ct_tree.edges:
                                if (edge[0] == current_state) and (edge[2] == current_transition):
                                    current_state = edge[1]
                                    break

                            if current_transition == transition:
                                transition_prob = edge[3]
                                exp_transition_rate = self.__transitions[exp_transition]
                                exp_transition_rate = exp_transition_rate[1]

                                throughput_rate = throughput_rate + self.__ctmc_steady_state[tangible_init_state] * exp_transition_rate * transition_prob

                if add_state:
                    states_already_considered.append(tangible_init_state)

        return throughput_rate

    def prob_of_n_tokens(self, place, ntokens):
        if not self.__ct_ctmc_generated:
            raise Exception(
                'Analysis must be initialized before this method can be used, please use init_analysis() method for that purpose.')

        prob_of_n_tokens = 0
        for state_id, marking in self.__ctmc.state.items():
            marking = marking[0]
            for pl in marking:
                if (place == pl[0]) and (ntokens == pl[1]):
                    prob_of_n_tokens = prob_of_n_tokens + self.__ctmc_steady_state[state_id]

        return prob_of_n_tokens

    def expected_number_of_tokens(self, place):
        if not self.__ct_ctmc_generated:
            raise Exception(
                'Analysis must be initialized before this method can be used, please use init_analysis() method for that purpose.')

        # compute the maximum possible number of tokens in the inputed place
        maximum_n_tokens = 0
        for state_id, marking in self.__ctmc.state.items():
            marking = marking[0]
            for pl in marking:
                if (pl[0] == place) and (pl[1] > maximum_n_tokens):
                    maximum_n_tokens = pl[1]

        # sum all the probabilities of having exactly n tokens in the given place
        expected_number_of_tokens = 0
        for ntokens in range(maximum_n_tokens):
            expected_number_of_tokens = expected_number_of_tokens + (ntokens+1)*self.prob_of_n_tokens(place,ntokens+1)

        return expected_number_of_tokens

    def transition_probability_evolution(self, period, step, initial_states_prob, state):
        if not self.__ct_ctmc_generated:
            raise Exception(
                'Analysis must be initialized before this method can be used, please use init_analysis() method for that purpose.')

        prob_evo = []
        for time_interval in np.arange(0, period, step):
            prob_all_states = self.__ctmc.get_prob_reach_states(initial_states_prob, time_interval)
            prob_evo.append(prob_all_states[state])
            # self.__ctmc.compute_transition_probability(step)
            # print(np.array(self.__ctmc.transition_probability))
            # prob_evo.append()

        return list(prob_evo)

    # def mean_wait_time(self, place):
    #     if not self.__ct_ctmc_generated:
    #         raise Exception(
    #             'Analysis must be initialized before this method can be used, please use init_analysis() method for that purpose.')
    #
    #     in_tr_m, out_tr_m = self.get_arcs()
    #     place_column = out_tr_m[0].index(place)
    #     out_tr_m = np.array(out_tr_m)
    #
    #     set_output_transitions = []
    #     for index in range(1,len(out_tr_m)):
    #         if int(out_tr_m[index,place_column]) > 0:
    #             set_output_transitions.append(out_tr_m[index,0])
    #
    #     sum = 0
    #     for transition in set_output_transitions:
    #         print(transition, self.transition_throughput_rate(transition))
    #         sum = sum + self.transition_throughput_rate(transition)
    #
    #     print(self.expected_number_of_tokens(place) / sum)

    def mean_wait_time(self, place):
        if not self.__ct_ctmc_generated:
            raise Exception(
                'Analysis must be initialized before this method can be used, please use init_analysis() method for that purpose.')

        in_tr_m, _ = self.get_arcs()

        # true/false list stating if there is an input connection or not between the given place and any transition
        idd = self.__new_arc_in_m.loc[place][:].values > 0
        # list with all the output transitions of the given place
        set_output_transitions = list(self.__new_arc_in_m.columns[idd].values)

        sum = 0
        for transition in set_output_transitions:
            sum = sum + self.transition_throughput_rate(transition)

        return self.expected_number_of_tokens(place) / sum

    def maximum_likelihood_transition(self, transiton, sample):
        '''
        Use maximum likelihood to iteratively estimate the lambda parameter of the exponential distribution that models the inputed transition
        :param transiton: (string) id of the transition that will be updated
        :param sample: (float) sample obtained from a exponential distribution
        :return: the estimated lambda parameter
        '''
        self.__nsamples[transiton] = self.__nsamples + 1
        self.__sum_samples[transiton] = self.__sum_samples[transiton] + sample
        lb = self.__nsamples[transiton] / self.__sum_samples[transiton]

        tr_info = self.__transitions[transiton]
        tr_info[1] = lb
        self.__transitions[transiton] = tr_info

        return lb

if __name__ == "__main__":
    # create a generalized stochastic petri net structure
    my_pn = GSPN()
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

    print(my_pn.get_enabled_transitions())
    a = my_pn.get_enabled_transitions()

    print('Places: ' , my_pn.get_current_marking(), '\n')
    print('Trans: ' , my_pn.get_transitions(), '\n')
    arcs_in , arcs_out = my_pn.get_arcs()
    print('Arcs IN: ' , arcs_in, '\n')
    print('Arcs OUT: ' , arcs_out, '\n')

    print(my_pn.add_tokens(['p1', 'p3', 'p5'], [10,5,1]))

    print('Places: ', my_pn.get_current_marking(), '\n')
