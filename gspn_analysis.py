

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
            elif exp_transitions_en:
                enabled_transitions = exp_transitions_en.copy()
            else:
                enabled_transitions = {}
                print('NO transitions enabled : deadlock and tangible')

            for tr in enabled_transitions.keys():
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
                    unbounded_state = True
                    for i in range(len(state[0])):
                        if next_marking[i][1] < state[0][i][1]:
                            unbounded_state = False
                    # Add an w to mark unbounded states
                    if unbounded_state:
                        for i in range(len(state[0])):
                            next_marking[i][1] = 'w'
                            # if next_marking[i][1] > state[0][i][1]:
                            #     next_marking[i][1] = 'w'

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
                    marking_stack.append([next_marking_dict, next_marking_id])

                # add edge between the current marking and the marking to where it just transitioned
                self.edges.append([current_marking_id, next_marking_id, tr])

                # revert the current marking
                self.__gspn.set_marking(current_marking_dict)

        self.__gspn.reset_simulation()
        return self.nodes.copy(), self.edges

    def boundness(self):
        '''
        The problem of boundedness is easily solved using a coverability tree. A necessary and
        sufficient condition for a Petri net to be bounded is that the symbol ω never appears in its
        coverability tree. Since ω represents an infinite number of tokens in some place, if ω appears
        in place p i , then p i is unbounded. For example, in Fig. 4.14, place Q is unbounded; this is
        to be expected, since there is no limit to the number of customers that may reside in the
        queue at any time instant.
        '''

    def safety(self):
        '''
        Finally, note that if ω does not appear in place p i , then the largest value of x(p i ) for any
        state encountered in the tree specifies a bound for the number of tokens in p i . For example,
        x(I) ≤ 1 in Fig. 4.14. Thus, place I is 1-bounded (or safe). If the coverability (reachability)
        tree of a Petri net contains states with 0 and 1 as the only place markings, then all places
        are guaranteed to be safe, and the Petri net is safe
        '''

    def liveness(self):
        '''
        Definition B.1.2. Given a Petri net with initial state M0, a transition tj is said to be live if,
        for all reachable states Mi, there is a firing sequence starting in Mi, such that tj is fired.
        A Petri net is live if all its transitions are live.
        '''

    def deadlock_free(self):
        '''
        Given a Petri net, a deadlock state corresponds to a reachable state where none of the transitions
        are fireable. A Petri net is deadlock-free if it contains no reachable deadlock state.
        '''