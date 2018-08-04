import gspn as pn
import xml.etree.ElementTree as et # XML parser
from graphviz import Digraph
import warnings

class pnml_tools(object):
    def __init__(self):
        '''
        Class constructor: will get executed at the moment
        of object creation
        '''

    def import_pnml(self, file):
        list_gspn = []  # list of parsed GSPN objects
        gspn = pn.gspn()

        tree = et.parse(file)  # parse XML with ElementTree
        root = tree.getroot()

        for petrinet in root.iter('net'):

            place_name = []
            place_marking = []
            for pl in petrinet.iter('place'): # iterate over all places of the petri net
                place_name.append(pl.get('id')) # get place name encoded as 'id' in the pnml structure

                text = pl.find('./initialMarking/value').text
                place_marking.append(int(text.split(',')[-1])) # get place marking encoded inside 'initalMarking', as the 'text' of the key 'value'

            # return place_name, place_marking

            gspn.add_places(place_name, place_marking) # add the compiled list of places to the gspn object

            transition_name = []
            transition_type = []
            transition_rate = []
            for tr in petrinet.iter('transition'): # iterate over all transitions of the petri net
                transition_name.append(tr.get('id')) # get transition name encoded as 'id' in the pnml structure

                if (tr.find('./timed/value').text == 'true'): # get the transition type either exponential ('exp') or immediate ('imm')
                    transition_type.append('exp')
                else:
                    transition_type.append('imm')

                transition_rate.append(float(tr.find('./rate/value').text)) # get the transition fire rate or weight

            gspn.add_transitions(transition_name, transition_type, transition_rate)  # add the compiled list of transitions to the gspn object

            arc_in_m, arc_out_m = self.__CreateArcMatrix(gspn.get_current_marking(), gspn.get_transitions())
            temp_arc_in_m = list(zip(*arc_in_m)) # easy way to get the column of a list
            temp_arc_out_m = list(zip(*arc_out_m)) # easy way to get the column of a list
            place_name = gspn.get_current_marking()
            place_name = place_name.keys()
            transition_name = gspn.get_transitions()
            transition_name = transition_name.keys()
            for arc in petrinet.iter('arc'): # iterate over all arcs of the petri net
                src = arc.get('source')
                trg = arc.get('target')
                if src in place_name: # IN arc connection (from place to transition)
                    arc_in_m[temp_arc_in_m[0].index(src)][arc_in_m[0].index(trg)] = 1
                elif src in transition_name: # OUT arc connection (from transition to place)
                    arc_out_m[temp_arc_out_m[0].index(src)][arc_out_m[0].index(trg)] = 1
                else:
                    return False

            gspn.add_arcs_matrices(arc_in_m, arc_out_m)

            list_gspn.append(gspn)

        return list_gspn

    def export_pnml(self, gspn):

        return True

    def show_gspn(self, file='', gspn=''):
        warnings.filterwarnings("ignore")
        # shape
        # style box rect
        # ref: https://www.graphviz.org/documentation/
        # check where to put forcelabels=true and  labelfloat='true'

        gspn_draw = Digraph()

        # places
        gspn_draw.node('P0',shape='circle', label='<&#9899;>', xlabel='P0')
        gspn_draw.node('P1',shape='circle',  label='', xlabel='P1')

        # exponential transitions
        gspn_draw.node('T0', shape='rectangle', color='black', label='', xlabel='T2')

        # immediate transitions
        gspn_draw.node('T1', shape='rectangle', style='filled', color='black', label='', xlabel='T2')
        gspn_draw.node('T2', shape='rectangle', style='filled', color='black', label='', xlabel='T2')

        gspn_draw.edge('P0','T0')
        gspn_draw.edge('P0','T1')
        gspn_draw.edge('P1','T2')
        gspn_draw.edge('T0','P1')
        gspn_draw.edge('T1','P1')
        gspn_draw.edge('T2','P0')

        gspn_draw.render('GV_gspn.gv', view=True)

        # time.sleep(5)

        gspn_draw.node('P0', shape='circle', label='', xlabel='P0')
        gspn_draw.node('P1', shape='circle', label='<&#9899;>', xlabel='P1')

        gspn_draw.render('GV_gspn.gv', view=True)

    def __CreateArcMatrix(self, places, transitions):
        # create a zeros matrix (# of places + 1) by (# of transitions)
        arc_in_m = []
        for i in range(len(places.keys())+1):
            arc_in_m.append([0]*(len(transitions.keys())))

        # replace the first row with all the transitions names
        arc_in_m[0] = list(transitions.keys())

        # add a first column with all the places names
        first_column = list(places.keys())
        first_column.insert(0, None) # put None in the element (0,0) since it has no use
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
        first_column.insert(0, None)  # put None in the element (0,0) since it has no use
        arc_out_m = list(zip(*arc_out_m))
        arc_out_m.insert(0, first_column)
        arc_out_m = list(map(list, zip(*arc_out_m)))

        return arc_in_m, arc_out_m

if __name__ == "__main__":
    # create a generalized stochastic petri net structure
    my_pn = pn.gspn()
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

    # print('Places: ' , my_pn.get_current_marking(), '\n')
    # print('Trans: ' , my_pn.get_transitions(), '\n')
    # print('Arcs IN: ' , my_pn.get_in_arcs(), '\n')
    # print('Arcs OUT: ' , my_pn.get_out_arcs(), '\n')
    #
    # print(my_pn.add_tokens(['p1', 'p3', 'p5'], [10,5,1]))
    #
    # print('Places: ', my_pn.get_current_marking(), '\n')

    parset = pnml_tools()
    a = parset.import_pnml('pipediag.xml')
    # print(mm)

    # parset = pnml_tools()
    # parset.show_gspn()

    # nets = pn.parse_pnml_file('example.pnml')
    # # print(nets)
    # for net in nets:
    #     print(nets.pop())
    #     print('------------')

