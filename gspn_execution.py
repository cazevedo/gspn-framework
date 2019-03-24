import gspn as pn
import gspn_tools as tools

class GSPNexecution(object):
    @staticmethod
    def make_executable(gspn):
        '''
        Returns an executable version of the Petri Net, where action places are unfolded in to two places
        :param gspn: (PN object)
        :return:    (PN object)
        '''
        all_places = gspn.get_current_marking()

        for place in all_places:

            exec_place = 'i.exec.' + place;
            exec_trans = 'ex.t.'   + place;
            end_place  = 'f.end_'  + place;

            sub_places = {}
            sub_places = {exec_place:0, end_place:0}

            sub_trans = {}
            sub_trans[exec_trans] = ['imm', 1]

            arcs_in = {}
            arcs_in[exec_place] = [exec_trans]
            arcs_out = {}
            arcs_out[exec_trans] = [end_place]


            subPN = pn.GSPN()
            subPN.add_places_dict(sub_places)
            subPN.add_transitions_dict(sub_trans)
            subPN.add_arcs(arcs_in, arcs_out)

            gspn = tools.GSPNtools.expand_pn(gspn,subPN,place)

        return gspn









