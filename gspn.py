import numpy as np
# import pntools as pnt
# from pntools import *
import petrinet as pn
import xml.etree.ElementTree as et # XML parser
from graphviz import Digraph
import time

class gspn(object):
    '''

    '''
    def __init__(self):
        '''
        Class constructor: will get executed at the moment
        of object creation
        '''
        self.places = {}
        self.transitions = {}
        self.arc_in = {}
        self.arc_out = {}
        self.arc_in_m = np.array([[]])
        self.arc_out_m = np.array([[]])

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
        name: list of strings, denoting the name of the transition
        type: list of strings, indicating if the corresponding transition is either immediate ('imm') or exponential ('exp')
        rate: list of floats, representing a static firing rate in an exponential transition and
                a static (non marking dependent) weight in a immediate transition
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
        PxT represents the arc connections from places to transitions
        '''
        # matrix = two dimentional array
        # [
        # [1, 2, 3],
        # [4, 5, 6]
        # ]
        # column -> a[:,1]
        # row -> a[1,:]
        self.arc_in = arc_in
        self.arc_out = arc_out

        list_items = np.array([])
        for source, target in arc_in.items():
            for item in target:
                list_items = np.append(list_items, item)

        self.arc_in_m = np.zeros( (len(arc_in)+1 , len(np.unique(list_items))+1) )
        # self.arc_in_m[0,:] =
        # self.arc_in_m[:,0] =

        # for source, target in arc_in.items():
        #     arc_in_m

        # arc_in = {}
        # arc_in['p1'] = ['t1']
        # arc_in['p2'] = ['t2']
        # arc_in['p3'] = ['t3']
        # arc_in['p4'] = ['t4']
        # arc_in['p5'] = ['t1', 't3']

        # self.arc_in = arc

        # self.arc_in_m = np.array([])

        # self.arc_in_m = [0] * len(arc)
        # for place, token in arc.items():
        #     np.append(self.arc_in_m, [, ])
        #     np.append(self.arc_in_m, [place, ])

        return True

    def add_arc_out(self, arc):
        '''
        TxP represents the arc connections from transitions to places
        '''
        self.arc_out = arc

        return True

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

    def get_in_arcs(self):
        return self.arc_in

    def get_out_arcs(self):
        return self.arc_out

    def execute(self, steps):
        '''

        '''

        # for place, token in self.places.items():


        return True

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
        # shape
        # style box rect
        # ref: https://www.graphviz.org/documentation/
        # check where to put forcelabels=true and  labelfloat='true'

        gspn_draw = Digraph(name='gspn')

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
    mm = my_pn.add_arcs(arc_in ,arc_out)

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

    parset = pnml_tools()
    parset.show_gspn()

    # nets = pn.parse_pnml_file('example.pnml')
    # # print(nets)
    # for net in nets:
    #     print(nets.pop())
    #     print('------------')