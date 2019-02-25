# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import linalg

# TODO : Extend analysis methods to matrix equation and reduction decomposition approaches (take a look at Murata)
class CoverabilityTree(object):
    def __init__(self, gspn):
        """
        The coverability tree of a GSPN is a two tuple (V; E), where V (nodes) represents the set of all reachable
        markings and E (edges) the set of all possible state changes.
        The coverability tree allows to perform a logical analysis, providing qualitative information on the GSPN, such as
        reachability, boundedness, safety and deadlocks.
        """
        self.__gspn = gspn
        self.nodes = {}
        self.edges = []
        self.deadlock_free = None

    def generate(self):
        self.deadlock_free = True

        # obtain the enabled transitions for the initial marking
        exp_transitions_en, immediate_transitions_en = self.__gspn.get_enabled_transitions()

        # from the enabled transitions get information on the marking type
        if immediate_transitions_en:
            marking_type = 'V'  # vanishing marking
        elif exp_transitions_en:
            marking_type = 'T'  # tangible marking
        else:
            marking_type = 'D'  # deadlock and tangible marking
            self.deadlock_free = False
            # print('NO transitions enabled : deadlock and tangible')

        current_marking_dict = self.__gspn.get_initial_marking()

        marking_index = 0
        current_marking_id = 'M' + str(marking_index)

        # convert marking from a dict structure into a list structure and sort it
        current_marking = []
        for place_id, ntokens in current_marking_dict.items():
            current_marking.append([place_id, ntokens])
        current_marking.sort()

        # add node to the coverability tree with the initial marking
        self.nodes[current_marking_id] = [current_marking, marking_type]

        # add the current marking to the marking stack
        marking_stack = [[current_marking_dict, current_marking_id]]

        # loop through the marking stack
        while marking_stack:
            # pop a marking from the stack using a FIFO methodology
            marking_stack.reverse()
            marking_info = marking_stack.pop()
            marking_stack.reverse()

            current_marking_dict = marking_info[0]
            current_marking_id = marking_info[1]

            # set the current marking as the marking of the GSPN
            self.__gspn.set_marking(current_marking_dict)

            # obtain the enabled transitions for this marking
            exp_transitions_en, immediate_transitions_en = self.__gspn.get_enabled_transitions()

            if immediate_transitions_en:
                enabled_transitions = immediate_transitions_en.copy()
                transition_type = 'I'
            elif exp_transitions_en:
                enabled_transitions = exp_transitions_en.copy()
                transition_type = 'E'
            else:
                enabled_transitions = {}
                # print('NO transitions enabled : deadlock and tangible')

            if enabled_transitions:
                # sum the rates from all enabled transitions, to obtain the transition probabilities between markings
                rate_sum = sum(enabled_transitions.values())
                for tr, rate in enabled_transitions.items():
                    # for each enabled transition of the current marking fire it to land in a new marking
                    self.__gspn.fire_transition(tr)

                    # obtain the enabled transitions for this marking
                    next_exp_trans, next_imm_trans = self.__gspn.get_enabled_transitions()

                    # from the enabled transitions get information on the marking type
                    if next_imm_trans:
                        marking_type = 'V'  # vanishing marking
                    elif next_exp_trans:
                        marking_type = 'T'  # tangible marking
                    else:
                        marking_type = 'D'  # deadlock and tangible marking
                        # print('NO transitions enabled : deadlock and tangible')

                    # get the new marking where it landed
                    next_marking_dict = self.__gspn.get_current_marking()

                    # convert marking from a dict structure into a list structure so it can be easily searchable if this already exists or no in the current directed graph
                    next_marking = []
                    for place_id, ntokens in next_marking_dict.items():
                        next_marking.append([place_id, ntokens])
                    next_marking.sort()

                    # check if the state is unbounded
                    for state_id, state in self.nodes.items():
                        # checks if the new marking is unbounded
                        unbounded_state = True
                        for i in range(len(state[0])):
                            if next_marking[i][1] < state[0][i][1]:
                                unbounded_state = False

                        # Add an w to mark unbounded states
                        if unbounded_state:
                            for i in range(len(state[0])):
                                if next_marking[i][1] > state[0][i][1]:
                                    next_marking[i][1] = 'w'

                    # check if the marking was already added as a node or not
                    marking_already_exists = False
                    for state_id, state in self.nodes.items():
                        if next_marking in state:
                            marking_already_exists = True
                            next_marking_id = state_id
                            break

                    if not marking_already_exists:
                        marking_index = marking_index + 1
                        next_marking_id = 'M' + str(marking_index)
                        self.nodes['M' + str(marking_index)] = [next_marking, marking_type]
                        # marking_stack.append([next_marking_dict, next_marking_id])
                        # if not unbounded_state:
                        marking_stack.append([next_marking_dict, next_marking_id])

                    # add edge between the current marking and the marking to where it just transitioned
                    self.edges.append([current_marking_id, next_marking_id, tr, rate/rate_sum, rate, transition_type])

                    # revert the current marking
                    self.__gspn.set_marking(current_marking_dict)

        self.__gspn.reset_simulation()
        return True

    def convert_states_to_latex(self):
        states = self.nodes.keys()
        states = sorted(states)

        (l, w) = len(states), len(self.nodes[states[0]][0]) + 1
        df = pd.DataFrame(np.zeros((l, w), dtype=np.dtype(object)))

        dataframe_header = ['State/Place']
        for row, state_name in enumerate(states):
            df.at[row, 0] = state_name

            marking = self.nodes[state_name][0]
            for column, place in enumerate(marking):
                place_name = place[0]
                token = place[1]
                df.at[row, column + 1] = token

                if row == 0:
                    dataframe_header.append(place_name)

        df.columns = dataframe_header
        return df.to_latex(header=True, index=False), df

    def boundedness(self):
        bounded_pn = True
        unbounded_places = []
        for marking_id, marking_info in self.nodes.items():
            for marking in marking_info[0]:
                if marking[1] == 'w' and (not marking[0] in unbounded_places):
                    bounded_pn = False
                    unbounded_places.append(marking[0])

        return bounded_pn, list(unbounded_places)


class CTMC(object):
    def __init__(self, reachability_graph):
        """
        Due to the memoryless property of the exponential transitions, it has been shown that the coverability tree of
        a bounded GSPN is isomorphic to a finite Markov Chain. Therefore in this case, we can obtain the equivalent
        continuous time Markov chain (CTMC), that comprises only tangible states.
        This mainly consists in removing all vanishing states, redirecting the arcs and computing the new stochastic
        transition rates from the weights of removed vanishing states.
        A CTMC makes transitions from state to state, independent of the past, according
        to a discrete-time Markov chain, but once entering a state remains in
        that state, independent of the past, for an exponentially distributed amount of
        time before changing state again.
        Thus a CTMC can simply be described by a transition matrix P = (P ij ), describing
        how the chain changes state step-by-step at transition epochs, together with a set of rates
        {a i : i ∈ S}, the holding time rates. Each time state i is visited, the chain spends, on
        average, E(H i ) = 1/a i units of time there before moving on.
        """
        bounded, unbound_pl = reachability_graph.boundedness()

        if not bounded:
            raise Exception('To obtain the equivalent continuous time markov chain the Petri net must be bounded, and this is not the case.')
        else:
            self.state = reachability_graph.nodes.copy()
            self.transition = list(reachability_graph.edges)

        self.__generated = False
        self.__transition_rate = False
        self.transition_probability = pd.DataFrame()
        self.infinitesimal_generator = pd.DataFrame()

    def generate(self):
        """
        Coverts a reachability graph into a continuous time markov chain.
        Populates the state and transition attributes with the information provided by the inputed reachability graph
        :return: (bool) True if successful
        """
        for marking_id, marking_info in self.state.items():
            if marking_id != 'M0':
                marking_info.append(0)
            else:
                marking_info.append(1)

        # get just the vanishing states
        vanishing_state_list = []
        for marking_id, marking_info in self.state.items():
            if marking_info[1] == 'V':
                vanishing_state_list.append([marking_id, marking_info[0], marking_info[1], marking_info[2]])
        vanishing_state_list.sort()

        for state in vanishing_state_list:
            marking_id = state[0]
            # marking = state[1]
            # marking_type = state[2]
            marking_prob = state[3]

            # check if the current marking has input arcs or not
            no_input_arcs = True
            for arc in self.transition:
                if arc[1] == marking_id:
                    no_input_arcs = False
                    break

            arcs_to_remove = []
            if no_input_arcs:

                for output_arc in self.transition:
                    output_state_id = output_arc[1]
                    # output_transition_id = output_arc[2]
                    output_transition_prob = output_arc[3]

                    if output_arc[0] == marking_id:
                        self.state[output_state_id][3] = self.state[output_state_id][3] + marking_prob*output_transition_prob
                        # mark arc to be removed
                        arcs_to_remove.append(output_arc)

            else:
                for output_arc in self.transition:

                    if output_arc[0] == marking_id:  # if this condition is true then it is an output arc
                        output_state = output_arc[1]
                        output_transition_id = output_arc[2]
                        output_transition_prob = output_arc[3]

                        for input_arc in self.transition:

                            if input_arc[1] == marking_id:  # if this condition is true then it is an input arc
                                input_state = input_arc[0]
                                input_transition_id = input_arc[2]
                                input_transition_prob = input_arc[3]
                                input_transition_rate = input_arc[4]
                                input_transition_type = input_arc[5]

                                if input_transition_type == 'I':
                                    if output_transition_id != input_transition_id:
                                        new_transition_id = input_transition_id + ':' + output_transition_id
                                    else:
                                        new_transition_id = input_transition_id
                                    self.transition.append([input_state, output_state, new_transition_id, output_transition_prob*input_transition_prob, None, 'I'])
                                else:
                                    if output_transition_id != input_transition_id:
                                        new_transition_id = input_transition_id + ':' + output_transition_id
                                    else:
                                        new_transition_id = input_transition_id
                                    self.transition.append([input_state, output_state, new_transition_id, output_transition_prob*input_transition_prob, input_transition_rate*output_transition_prob, 'E'])

                                # mark arc to be removed
                                if not (input_arc in arcs_to_remove):
                                    arcs_to_remove.append(input_arc)

                        # mark arc to be removed
                        if not (output_arc in arcs_to_remove):
                            arcs_to_remove.append(output_arc)

            for i in arcs_to_remove:
                self.transition.remove(i)

            del self.state[marking_id]

        temp_trans = list(self.transition)
        duplicated_transitions = []
        for arc1_index in range(len(temp_trans)):
            isduplicated = False
            for arc2_index in range(arc1_index+1, len(temp_trans)):
                if ( (temp_trans[arc1_index][0] == temp_trans[arc2_index][0]) and
                     (temp_trans[arc1_index][1] == temp_trans[arc2_index][1]) and
                     (temp_trans[arc1_index] in self.transition) ):
                    duplicated_transitions.append(temp_trans[arc2_index])
                    isduplicated = True

            if isduplicated:
                duplicated_transitions.append(temp_trans[arc1_index])

                tr = ''
                prob = 0.0
                rate = 0.0
                while duplicated_transitions:
                    dupl = duplicated_transitions.pop()
                    tr = tr + dupl[2] + '/'
                    prob = prob + dupl[3]
                    rate = rate + dupl[4]
                    # delete the duplicated transitions
                    self.transition.remove(dupl)

                tr = tr[0:-1]

                # create a new transition to replace the duplicated where the probabilities of the duplicated are summed
                self.transition.append([temp_trans[arc1_index][0], temp_trans[arc1_index][1], tr, prob, rate, 'E'])

        self.__generated = True
        return True

    def compute_transition_rate(self):
        """
        Computes the transition rate matrix, also called, infinitesimal generator Q.
        Where Qij is the rate of going from state i to state j at time t,
        and Qii represents the rate of leaving state i at time t.
        :return: (bool) True or False depending if it was successful or not
        """
        if self.__generated:
            states_id = self.state.keys()
            states_id = sorted(states_id)
            states_len = len(states_id)

            self.infinitesimal_generator = pd.DataFrame(np.zeros((states_len, states_len)))

            self.infinitesimal_generator.columns = states_id
            self.infinitesimal_generator.index = states_id

            # arc[0] stores the origin state ID (e.g. 'M1' or 'M7')
            # arc[1] stores the next state ID (e.g. 'M2' or 'M15')
            # arc[2] stores the transition ID (e.g. 'T2' or 'T11')
            # arc[3] stores the transition probability (e.g. 0.25 or 0.33)
            # arc[4] stores the firing rate of the transition (e.g. 1.0 or 0.2)
            # arc[4] stores the transition type ('E' for exponential 'I' for immediate)
            for arc in self.transition:
                self.infinitesimal_generator.loc[arc[0]][arc[1]] = arc[4]

            row_sum = self.infinitesimal_generator.sum(axis=1)

            for st in states_id:
                self.infinitesimal_generator.loc[st][st] = -row_sum.loc[st]

            self.__transition_rate = True
            return True
        else:
            return False

    def convert_states_to_latex(self):
        states = self.state.keys()
        states.sort()

        (l, w) = len(states), len(self.state[states[0]][0]) + 1
        df = pd.DataFrame(np.zeros((l, w), dtype=np.dtype(object)))

        dataframe_header = ['State/Place']
        for row, state_name in enumerate(states):
            df.at[row, 0] = state_name

            marking = self.state[state_name][0]
            for column, place in enumerate(marking):
                place_name = place[0]
                token = place[1]
                df.at[row, column + 1] = token

                if row == 0:
                    dataframe_header.append(place_name)

        df.columns = dataframe_header
        return df.to_latex(header=True, index=False), df

    def compute_transition_probability(self, time_interval):
        """
        Populates the square matrix Hij(t) (encoded here as the attribute transition_probability), i.e. the probability that
        the chain will be in state j, t time units from now, given it is in state i now.
        The transition probability matrix (H(t)) is computed from the infinitesimal generator (Q) through the formula:
        H(t) = exp(Q*t), using Pade approximation to solve it
        The computed transition probability can be accessed through the CTMC attribute transition_probability.
        :param time_interval: (float) time units that have elapsed from now
        :return: (bool) True if it was successful, raises an exception otherwise
        """
        if time_interval < 0:
            raise Exception('Time interval must be greater or equal to zero.')

        if self.__transition_rate:
            self.transition_probability = linalg.expm(self.infinitesimal_generator.values * time_interval)

            self.transition_probability = pd.DataFrame(self.transition_probability)
            self.transition_probability.columns = self.state.keys()
            self.transition_probability.index = self.state.keys()

            return True

        else:
            raise Exception('Transition rates are not computed, please use the method compute_transition_rate')

    def compute_transition_probability_old(self, time_interval, precision=6):
        """
        Populates the matrix Hij(t) (encoded here as the attribute transition_probability), i.e. the probability that
        the chain will be in state j, t time units from now, given it is in state i now.
        The transition probability matrix (H(t)) is computed from the infinitesimal generator (Q) through the formula:
        H(t) = exp(Q*t), by approximating it to H(t) ~= (1+Q*t/large_n)^large_n
        The computed transition probability can be accessed through the CTMC attribute transition_probability.
        :param time_interval: (float) time units that have elapsed from now
        :param precision: (int) number of decimal places
        :return: (bool) True if it was successful
        """
        if time_interval < 0:
            raise Exception('Time interval must be greater or equal to zero.')

        if self.__transition_rate:
            n_states = len(self.state.keys())

            error = 1.0 / (10**precision)
            k = 100
            while True:
                large_n = 2 ** k
                self.transition_probability = np.identity(n_states) + (self.infinitesimal_generator * time_interval / large_n)
                self.transition_probability = np.linalg.matrix_power(self.transition_probability.values, large_n)

                row_sum = self.transition_probability.sum(axis=1)

                delta = row_sum.std(axis=0)

                k = k - 1

                if (delta < error) and (round(row_sum.sum(), precision) == n_states):
                    break
                elif k == 0:
                    raise Exception('Algorithm was not able to compute the transition probabilities for the given time and precision. Please consider lowering the precision.')

            self.transition_probability = pd.DataFrame(self.transition_probability)
            self.transition_probability.columns = self.state.keys()
            self.transition_probability.index = self.state.keys()

            return True

        else:
            raise Exception('Transition rates are not computed, please use the method compute_transition_rate')

    def get_prob_reach_states(self, initial_states_prob, time_interval):
        '''
        Computes the probability of reaching all states after a certain amount of time have elapsed.
        :param initial_states_prob: pandas DataFrame where each row is a state and the value corresponds to the probability
        of initially being in that state. Can also be interpreted as the probability mass function of the random variable X at the beginning of time.
        :param time_interval: (float) representing the time elapsed after the inputed initial state.
        :return: (pandas DataFrame) where each row is a different state and the value of that row corresponds to the probability
        of reaching that state after the inputed time has elapsed.
        '''

        self.compute_transition_probability(time_interval)

        final_state_probability = initial_states_prob.T.dot(self.transition_probability)

        return final_state_probability.T

    def get_steady_state(self):
        '''
        Computes the steady-state probability distribution PI by solving the linear system: PI*Q = 0, sum(PI_j) = 1
        Where PI_j is the probability of reaching the state j and PI = [PI_0, PI_1, ...]
        :return: (pandas DataFrame) where each row represents a different state and the value of that row corresponds to
        the steady-state probability of that state
        '''

        if self.__transition_rate:
            a = np.concatenate((self.infinitesimal_generator.values.T, [[1] * len(self.infinitesimal_generator)]))
            b = np.zeros(len(self.infinitesimal_generator))
            b = np.append(b, 1)

            x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)

            steady_state = pd.DataFrame(x)
            steady_state.index = self.infinitesimal_generator.index

            return steady_state.copy()

        else:
            raise Exception('Transition rates are not computed, please use the method compute_transition_rate')

    def get_steady_state_iteratively(self, precision=6):
        '''
        Computes the steady-state probability distribution PI by iteratively increasing the time interval t until PI(t) converges.
        Where PI_j is the probability of reaching the state j and PI = [PI_0, PI_1, ...]
        :return: (pandas DataFrame) where each row represents a different state and the value of that row corresponds to
        the steady-state probability of that state
        '''

        if self.__transition_rate:
            states_id = self.state.keys()
            states_id = sorted(states_id)

            # steady state prob are independent from the initial probability
            # so for simplicity we use an uniform distribution as initial prob distribution
            states_prob = pd.DataFrame(1.0/len(states_id)*np.ones(len(states_id)))
            states_prob.index = states_id

            time_interval = 0.001
            old_steady_state = self.get_prob_reach_states(states_prob, time_interval)

            convergence_std = 1.0 / 10 ** precision
            while True:
                time_interval = time_interval + 1
                steady_state = self.get_prob_reach_states(states_prob, time_interval)

                diff = steady_state-old_steady_state

                if diff.std().values < convergence_std:
                    break
                else:
                    old_steady_state = steady_state.copy()

            return steady_state.copy()

        else:
            raise Exception('Transition rates are not computed, please use the method compute_transition_rate')

    def get_sojourn_times(self):
        if self.__generated:
            states_id = sorted(self.state.keys())

            sojourn_times = {}
            for current_state in states_id:
                state_rate = 0
                for output_state in states_id:
                    for arc in self.transition:
                        if (arc[0] == current_state) and (arc[1] == output_state):
                            state_rate = state_rate + arc[4]

                sojourn_times[current_state] = 1 / state_rate

            return sojourn_times.copy()

        else:
            raise Exception('Continuous time Markov chain is not created. Please use the method generate to obtain the CTMC.')