import gspn_tools

class AnalyseGSPN(object):
    def __init__(self, gspn):
        """

        """
        self.__gspn = gspn
        self.__nodes = {}
        self.__edges = {}

        self.pntools = gspn_tools.GSPNtools()

    def coverability_tree(self):
        marking_index = 0
        marking_stack = []
        marking_stack.append([self.__gspn.get_initial_marking(), 'root'])

        while marking_stack:
            # pop a marking from the stack using a FIFO methodology
            marking_stack.reverse()
            marking_info = marking_stack.pop()
            marking_stack.reverse()

            # gather all the marking information in a readable manner
            current_marking_dict = marking_info[0]
            source_marking_id = marking_info[1]
            current_marking_id = 'M'+str(marking_index)
            marking_index = marking_index + 1

            # convert marking from a dict structure into a list structure and sort it
            current_marking = []
            for place_id, ntokens in current_marking_dict.items():
                current_marking.append([place_id, ntokens])
            current_marking.sort()

            # set the current marking as the marking of the GSPN
            self.__gspn.set_marking(current_marking_dict)

            # obtain the enabled transitions for this marking
            exp_transitions_en, immediate_transitions_en = self.__gspn.get_enabled_transitions()

            # from the enabled transitions get information on the marking type
            if immediate_transitions_en:
                enabled_transitions = immediate_transitions_en.copy()
                marking_type = 'V'  # vanishing marking
            elif exp_transitions_en:
                enabled_transitions = exp_transitions_en.copy()
                marking_type = 'T'  # tangible marking
            else:
                enabled_transitions = {}
                marking_type = 'D'  # deadlock and tangible marking
                print('NO transitions enabled : deadlock and tangible')

            self.__nodes[current_marking_id] = [current_marking, marking_type]  # add a node where the key is the marking id and the value is a list with the marking and its type
            self.__edges[source_marking_id] = current_marking_id  # add an edge from the source (previous marking stored when the marking was pushed to the stack) to this marking

            # print(current_marking, marking_type)
            # drawing = self.pntools.draw_gspn(self.__gspn, 'mypn', show=False)
            # self.pntools.draw_enabled_transitions(self.__gspn, drawing, 'mypn_enabled', show=True)
            # raw_input("")

            for tr in enabled_transitions.keys():
                # for each enabled transition of the current marking fire it to land in a new marking
                self.__gspn.fire_transition(tr)

                # get the new marking where it landed
                next_marking_dict = self.__gspn.get_current_marking()

                # convert marking from a dict structure into a list structure so it can be easily searchable if this already exists or no in the current directed graph
                next_marking = []
                for place_id, ntokens in next_marking_dict.items():
                    next_marking.append([place_id, ntokens])
                next_marking.sort()

                # check if the marking was already added as a node or not
                marking_already_exists = False
                for state_id, state in self.__nodes.items():
                    if next_marking in state:
                        marking_already_exists = True
                        break

                # check if the marking is in te marking stack
                for mrk in marking_stack:
                    if next_marking_dict == mrk[0]:
                        marking_already_exists = True
                        break

                    # else:
                    #     for place_index in range(len(state)):
                    #         if  next_marking[place_index][1] > state[place_index][1]:

                # if doesn't exist append it to the marking stack to be handled in the following iterations on FIFO method
                if not marking_already_exists:
                    marking_stack.append([next_marking_dict, current_marking_id])

                # revert the current marking
                self.__gspn.set_marking(current_marking_dict)

        test_repeated = self.__nodes.copy()
        for state_id, state in self.__nodes.items():
            for key, value in test_repeated.items():
                if value[0] == state[0] and key != state_id:
                    print('REPEATED MARKING!!!')
                    print(state_id, state)
                    print(key, value)

        self.__gspn.reset_simulation()
        return self.__nodes.copy(), self.__edges.copy()
