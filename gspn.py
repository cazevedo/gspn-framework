import numpy as np
# import pntools as pnt
# from pntools import *
import petrinet as pn
import xml.etree.ElementTree as et # XML parser
# from graphviz import Digraph
import time
import warnings

class gspn(object):
    '''

    '''
    def __init__(self):
        '''

        '''
        self.places = {}
        self.transitions = {}
        self.arc_in = {}
        self.arc_out = {}
        self.arc_in_m = []
        self.arc_out_m = []

    def add_places(self, name, ntokens=[]):
        '''

        '''
        name.reverse()
        ntokens.reverse()
        while name:
            if ntokens:
                self.places[name.pop()] = ntokens.pop()
            else:
                self.places[name.pop()] = 0

        return self.places

    def add_transitions(self, name, type=[], rate=[]):
        '''
        Input
        name: list of strings, denoting the name of the transition
        type: list of strings, indicating if the corresponding transition is either immediate ('imm') or exponential ('exp')
        rate: list of floats, representing a static firing rate in an exponential transition and
                a static (non marking dependent) weight in a immediate transition
        Output
        dictionary of transitions
        '''
        name.reverse()
        type.reverse()
        rate.reverse()
        while name:
            tn = name.pop()
            self.transitions[tn] = []
            if type:
                self.transitions[tn].append(type.pop())
            else:
                self.transitions[tn].append('imm')
            if rate:
                self.transitions[tn].append(rate.pop())
            else:
                self.transitions[tn].append(1.0)

        return self.transitions

    def add_arcs(self, arc_in, arc_out):
        '''
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
        arc_in_m -> two-dimentional list
        arc_out_m -> two-dimentional list

        arc_in_m = P × T : represents the arc connections from places to transitions, such that i lj = 1 if,
        and only if, there is an arc from p l to t j , and i lj = 0 otherwise;

        arc_out_m = T × P represent the arc connections from transition to places, such that o lj = 1 if,
        and only if, there is an arc from t l to p j , and o lj = 0 otherwise;
        '''

        self.arc_in = arc_in
        self.arc_out = arc_out

        # IN ARCS MAP
        self.arc_in_m = []
        for i in range(len(self.places.keys())+1):
            self.arc_in_m.append([0]*(len(self.transitions.keys())))

        first_column = list(self.places.keys())
        first_column.insert(0, None)
        self.arc_in_m[0] = list(self.transitions.keys())

        self.arc_in_m = list(zip(*self.arc_in_m))
        self.arc_in_m.insert(0, first_column)
        self.arc_in_m = list(map(list, zip(*self.arc_in_m)))

        temp = list(zip(*self.arc_in_m))
        for place, target in arc_in.items():
            for transition in target:
                self.arc_in_m[temp[0].index(place)][self.arc_in_m[0].index(transition)] = 1

        # OUT ARCS MAP
        self.arc_out_m = []
        for i in range(len(self.transitions.keys())+1):
            self.arc_out_m.append([0]*(len(self.places.keys())))

        first_column = list(self.transitions.keys())
        first_column.insert(0, None)
        self.arc_out_m[0] = list(self.places.keys())

        self.arc_out_m = list(zip(*self.arc_out_m))
        self.arc_out_m.insert(0, first_column)
        self.arc_out_m = list(map(list, zip(*self.arc_out_m)))

        temp = list(zip(*self.arc_out_m))
        for transition, target in arc_out.items():
            for place in target:
                self.arc_out_m[temp[0].index(transition)][self.arc_out_m[0].index(place)] = 1


        return self.arc_in_m, self.arc_out_m

    def add_tokens(self, place_name, ntokens):
        '''
        add tokens to the current marking
        '''
        if len(place_name) == len(ntokens):
            place_name.reverse()
            ntokens.reverse()
            while place_name:
                p = place_name.pop()
                self.places[p] = self.places[p] + ntokens.pop()

            return True
        else:
            return False

    def get_current_marking(self):
        return self.places

    def get_transitions(self):
        return self.transitions

    def get_arcs(self):
        return self.arc_in, self.arc_out, self.arc_in_m, self.arc_out_m

    def execute(self, steps):
        '''

        '''

        # for place, token in self.places.items():


        return True

    def __column(matrix, i):
        return [row[i] for row in matrix]

class pnml_tools(object):
    def __init__(self):
        '''
        Class constructor: will get executed at the moment
        of object creation
        '''
        self.list_gspn = []  # list of parsed GSPN objects
        self.gspn = gspn()


    def import_pnml(self, file):
        tree = et.parse(file)  # parse XML with ElementTree
        root = tree.getroot()

        for petrinet in root.iter('net'):

            place_name = []
            place_marking = []
            for pl in petrinet.iter('place'): # iterate over all places of the petri net
                place_name.append(pl.get('id')) # get place name encoded as 'id' in the pnml structure

                text = pl.find('./initialMarking/value').text
                place_marking.append(int(text.split(',')[-1])) # get place marking encoded inside 'initalMarking', as the 'text' of the key 'value'

            self.gspn.add_places(place_name, place_marking) # add the compiled list of places to the gspn object

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

            self.gspn.add_transitions(transition_name, transition_type, transition_rate)  # add the compiled list of transitions to the gspn object



        return True

    def export_pnml(self, file):

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

        time.sleep(5)

        gspn_draw.node('P0', shape='circle', label='', xlabel='P0')
        gspn_draw.node('P1', shape='circle', label='<&#9899;>', xlabel='P1')

        gspn_draw.render('GV_gspn.gv', view=True)


if __name__ == "__main__":
    # create a generalized stochastic petri net structure
    my_pn = gspn()
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
    my_pn.add_arcs(arc_in ,arc_out)

    # print('Places: ' , my_pn.get_current_marking(), '\n')
    # print('Trans: ' , my_pn.get_transitions(), '\n')
    # print('Arcs IN: ' , my_pn.get_in_arcs(), '\n')
    # print('Arcs OUT: ' , my_pn.get_out_arcs(), '\n')
    #
    # print(my_pn.add_tokens(['p1', 'p3', 'p5'], [10,5,1]))
    #
    # print('Places: ', my_pn.get_current_marking(), '\n')

    # parset = pnml_tools()
    # mm = parset.import_pnml('pipediag.xml')
    # print(mm)

    # parset = pnml_tools()
    # parset.show_gspn()

    # nets = pn.parse_pnml_file('example.pnml')
    # # print(nets)
    # for net in nets:
    #     print(nets.pop())
    #     print('------------')