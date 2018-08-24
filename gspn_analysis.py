# -*- coding: utf-8 -*-

# TODO : Replace nodes by states and edges by arcs
class CoverabilityTree(object):
    def __init__(self, gspn):
        """

        """
        self.__gspn = gspn
        self.nodes = {}
        self.edges = []

    def generate(self):
        # obtain the enabled transitions for the initial marking
        exp_transitions_en, immediate_transitions_en = self.__gspn.get_enabled_transitions()

        # from the enabled transitions get information on the marking type
        if immediate_transitions_en:
            marking_type = 'V'  # vanishing marking
        elif exp_transitions_en:
            marking_type = 'T'  # tangible marking
        else:
            marking_type = 'D'  # deadlock and tangible marking
            print('NO transitions enabled : deadlock and tangible')

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
                print('NO transitions enabled : deadlock and tangible')

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
                        print('NO transitions enabled : deadlock and tangible')

                    # get the new marking where it landed
                    next_marking_dict = self.__gspn.get_current_marking()

                    # convert marking from a dict structure into a list structure so it can be easily searchable if this already exists or no in the current directed graph
                    next_marking = []
                    for place_id, ntokens in next_marking_dict.items():
                        next_marking.append([place_id, ntokens])
                    next_marking.sort()

                    # check if the state is unbounded
                    for state_id, state in self.nodes.items():
                        # checks if the new marking is either unbounded or equal to any of the markings previously added to the nodes
                        unbounded_or_equal = True
                        for i in range(len(state[0])):
                            if next_marking[i][1] < state[0][i][1]:
                                unbounded_or_equal = False

                        # differentiates from equal markings from unbounded markings
                        unbounded_state = False
                        if unbounded_or_equal:
                            for i in range(len(state[0])):
                                if next_marking[i][1] > state[0][i][1]:
                                    unbounded_state = True

                        # Add an w to mark unbounded states
                        if unbounded_state:
                            for i in range(len(state[0])):
                                # next_marking[i][1] = 'w'
                                if next_marking[i][1] > state[0][i][1]:
                                    next_marking[i][1] = 'w'
                            break

                            # add edge between the current marking and the marking that is covered by this new unbounded state
                            # self.edges.append([current_marking_id, state_id, tr])

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

    def boundness(self):
        unbounded_pn = False
        unbounded_places = []
        for marking_id, marking_info in self.nodes.items():
            for marking in marking_info[0]:
                if marking[1] == 'w' and (not marking[0] in unbounded_places):
                    unbounded_pn = True
                    unbounded_places.append(marking[0])

        return unbounded_pn, list(unbounded_places)


class CTMC(object):
    def __init__(self, reachability_graph):
        """
        A CTMC makes transitions from state to state, independent of the past, ac-
        cording to a discrete-time Markov chain, but once entering a state remains in
        that state, independent of the past, for an exponentially distributed amount of
        time before changing state again.
        Thus a CTMC can simply be described by a transition matrix P = (P ij ), describing
        how the chain changes state step-by-step at transition epochs, together with a set of rates
        {a i : i ∈ S}, the holding time rates. Each time state i is visited, the chain spends, on
        average, E(H i ) = 1/a i units of time there before moving on.
        """
        unbound, unbound_pl = reachability_graph.boundness()
        if unbound:
            print("To obtain the equivalent continuous time markov chain the Petri net must be bounded, and this is not the case.")
        else:
            self.state = reachability_graph.nodes.copy()
            self.transition = list(reachability_graph.edges)

        self.__generated = False
        self.transition_probability = []

    def generate(self):
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

    def create_transition_probability_matrix(self, time_index=0):
        if self.__generated:
            states_id = self.state.keys()
            states_id.sort()

            # create a zeros matrix (# of states + 1) by (# of states)
            n_states = len(states_id)
            for i in range(n_states + 1):
                self.transition_probability.append([0] * n_states)

            # replace the first row with all the states names
            self.transition_probability[0] = list(states_id)

            # add a first column with all the states names
            first_column = list(states_id)
            first_column.insert(0, '')  # put None in the element (0,0) since it has no use
            self.transition_probability = list(zip(*self.transition_probability))
            self.transition_probability.insert(0, first_column)
            self.transition_probability = list(map(list, zip(*self.transition_probability)))

            if time_index == 0:
                for row_index in range(1, n_states+1):
                    for column_index in range(1, n_states+1):
                        if row_index == column_index:
                            self.transition_probability[row_index][column_index] = 1


                # for i in self.transition_probability:
                #     print(i)
            # else:

            return self.transition_probability

        # else:
        #     return False
