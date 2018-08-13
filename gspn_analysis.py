

class AnalyseGSPN(object):
    def __init__(self, gspn):
        """

        """
        self.__gspn = gspn
        self.__nodes = {}
        self.__edges = {}

    def coverability_tree(self):
        marking_index = 0
        self.__nodes['M'+str(marking_index)] = self.__gspn.get_initial_marking()

        exp_transitions_en, immediate_transitions_en = self.__gspn.get_enabled_transitions()
        if immediate_transitions_en:
            enabled_transitions = immediate_transitions_en.copy()
        elif exp_transitions_en:
            enabled_transitions = exp_transitions_en.copy()
        else:
            print('NO transitions enabled -> do something')

        previous_marking = self.__gspn.get_current_marking()

        # print(enabled_transitions)
        for tr in enabled_transitions.keys():
            self.__gspn.fire_transition(tr)
            marking_index = marking_index + 1
            self.__nodes['M'+str(marking_index)] = self.__gspn.get_current_marking()
            self.__gspn.set_marking(previous_marking)

        self.__gspn.reset_simulation()
        return self.__nodes.copy()


        return True
