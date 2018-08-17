
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
        # marking_stack.append([current_marking_dict, current_marking_id])

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
                self.edges.append([current_marking_id, next_marking_id, tr, rate, transition_type])

                # revert the current marking
                self.__gspn.set_marking(current_marking_dict)

        self.__gspn.reset_simulation()
        return self.nodes.copy(), self.edges

    def boundness(self):
        unbounded_pn = False
        unbounded_places = []
        for marking_id, marking_info in self.nodes.items():
            for marking in marking_info[0]:
                if marking[1] == 'w' and (not marking[0] in unbounded_places):
                    unbounded_pn = True
                    unbounded_places.append(marking[0])

        return unbounded_pn, unbounded_places

class CMTC(object):
    def __init__(self, reachability_graph):
        """

        """
        unbound, unbound_pl = reachability_graph.boundness()
        if unbound:
            print("To obtain the equivalent continuous time markov chain the Petri net must be bounded, and this is not the case.")
        else:
            self.state = reachability_graph.nodes.copy()
            self.transition = reachability_graph.edges

    def generate(self):
        for marking_id, marking_info in self.state.items():
            if marking_id != 'M0':
                marking_info.append(0)
            else:
                marking_info.append(1)

        for marking_id, marking_info in self.state.items():
            # get only vanishing markings
            if marking_info[1] == 'V':

                weight_sum = 0
                # obtain the sum of the weights of the corresponding transitions of all output arcs associated with the current marking (marking_id)
                for arc in self.transition:
                    if arc[0] == marking_id:
                        weight_sum = weight_sum + arc[3]

                # compute the transition fire probability for each arc
                for arc_index in range(len(self.transition)):
                    if self.transition[arc_index][0] == marking_id:
                        self.transition[arc_index][3] = self.transition[arc_index][3] / weight_sum

        vanishing_state_list = []
        for marking_id, marking_info in self.state.items():
            if marking_info[1] == 'V':
                vanishing_state_list.append([marking_id, marking_info[0], marking_info[1], marking_info[2]])
        vanishing_state_list.sort()


        # for marking in state_list:
        #     print(marking)
        #
        # print('-----------------------------------------------')
        # print('-----------------------------------------------')
        # print('-----------------------------------------------')

        for state in vanishing_state_list:
            marking_id = state[0]
            marking = state[1]
            marking_type = state[2]
            marking_prob = state[3]

            # print(state)

            # # get only vanishing markings
            # if marking_type == 'V':

            # check if the current marking has input arcs or not
            no_input_arcs = True
            for arc in self.transition:
                if arc[1] == marking_id:
                    no_input_arcs = False
                    break

            if no_input_arcs:
                for output_arc in self.transition:
                    output_state_id = output_arc[1]
                    output_transition_id = output_arc[2]
                    output_transition_prob = output_arc[3]

                    if output_arc[0] == marking_id:
                        output_state = self.state[output_state_id]
                        output_state.insert(0, output_state_id)
                        output_state = [output_state_id, self.state[output_state_id]]
                        # state_list[state_list.index(output_state)][3] = state_list[state_list.index(output_state)][3] + marking_prob*output_transition_prob

                        self.state[output_state_id][3] = self.state[output_state_id][3] + marking_prob*output_transition_prob

                        self.transition.remove(output_arc)

            else:
                for output_arc in self.transition:
                    output_state = output_arc[1]
                    output_transition_id = output_arc[2]
                    output_transition_prob = output_arc[3]

                    if output_arc[0] == marking_id:  # if this condition is true then it is an output arc

                        for input_arc in self.transition:
                            input_state = input_arc[0]
                            input_transition_id = input_arc[2]
                            input_transition_prob = input_arc[3]
                            input_transition_type = input_arc[4]

                            if input_arc[1] == marking_id:  # if this condition is true then it is an input arc

                                if input_transition_type == 'I':
                                    if output_transition_id != input_transition_id:
                                        new_transition_id = input_transition_id + ':' + output_transition_id
                                    else:
                                        new_transition_id = input_transition_id
                                    self.transition.append([input_state, output_state, new_transition_id,output_transition_prob*input_transition_prob, 'I'])
                                else:
                                    if output_transition_id != input_transition_id:
                                        new_transition_id = input_transition_id + ':' + output_transition_id
                                    else:
                                        new_transition_id = input_transition_id
                                    self.transition.append([input_state, output_state, new_transition_id,output_transition_prob*input_transition_prob, 'E'])

                                self.transition.remove(input_arc)

                        self.transition.remove(output_arc)

            del self.state[marking_id]
            # state_list.remove(state)

        for arc in self.transition:
            print(arc)

        # print('-----------------------------------------------')
        # print('-----------------------------------------------')
        # print('-----------------------------------------------')
        #
        for marking_id, marking_info in self.state.items():
            print(marking_id)
            print(marking_info)

        return True