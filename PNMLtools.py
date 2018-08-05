import gspn as pn
import xml.etree.ElementTree as et # XML parser
from graphviz import Digraph
import warnings
import time

class gspn_tools(object):
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
        # TODO: EXPORT PNML FROM GSPN
        return True

    def show_enabled_transitions(self, gspn, gspn_draw, file='default'):
        enabled_transitions = gspn.get_enabled_transitions()

        trns = gspn.get_transitions()
        for transition in enabled_transitions:
            transition_type = trns.get(transition)[0]
            if transition_type == 'exp':
                gspn_draw.node(transition, shape='rectangle', color='red', label='', xlabel=transition)
            else:
                gspn_draw.node(transition, shape='rectangle', style='filled', color='red', label='', xlabel=transition)

        gspn_draw.render(file + '.gv', view=True)

        return gspn_draw

    def show_gspn(self, gspn, file='default'):
        warnings.filterwarnings("ignore")
        # shape
        # style box rect
        # ref: https://www.graphviz.org/documentation/
        # check where to put forcelabels=true and  labelfloat='true'

        gspn_draw = Digraph()

        gspn_draw.attr('node', forcelabels='true')

        # draw places and marking
        plcs = gspn.get_current_marking()
        for place, marking in plcs.items():
            if int(marking) == 0:
                gspn_draw.node(place, shape='circle', label='', xlabel=place)
            else:
                lb = '<' + '&#9899;'*int(marking) + '>'
                gspn_draw.node(place, shape='circle', label=lb, xlabel=place)

        # draw transitions
        trns = gspn.get_transitions()
        for transition, value in trns.items():
            if value[0] == 'exp':
                gspn_draw.node(transition, shape='rectangle', color='black', label='', xlabel=transition)
            else:
                gspn_draw.node(transition, shape='rectangle', style='filled', color='black', label='', xlabel=transition)

        # draw edges
        edge_in, edge_out = gspn.get_arcs()

        # draw arcs in connections from place to transition
        for row_index in range(1,len(edge_in)):
            for column_index in range(1,len(edge_in[row_index])):
                if edge_in[row_index][column_index] == 1:
                    gspn_draw.edge(edge_in[row_index][0], edge_in[0][column_index])

        # draw arcs out connections from transition to place
        for row_index in range(1,len(edge_out)):
            for column_index in range(1,len(edge_out[row_index])):
                if edge_out[row_index][column_index] == 1:
                    gspn_draw.edge(edge_out[row_index][0], edge_out[0][column_index])

        # gspn_draw.node_attr.update()
        #     _attr.update(arrowhead='vee', arrowsize='2')
        # gspn_draw.subgraph()

        gspn_draw.render(file+'.gv', view=True)

        return gspn_draw

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

    parset = gspn_tools()
    a = parset.import_pnml('pipediag.xml')
    pn = a[0]

    drawing = parset.show_gspn(pn, 'pipediag')

    time.sleep(2)

    parset.show_enabled_transitions(pn, drawing, 'pipediag')

