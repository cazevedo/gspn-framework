#! /usr/bin/env python3
from gspn_framework_package import gspn as pn
import pandas as pd
import xml.etree.ElementTree as et  # XML parser
from graphviz import Digraph

class GSPNtools(object):

    # TODO : Document CT and CTMC drawing, by describing what shows up in the generated image
    # (states, prob, firing rate, etc)

    @staticmethod
    def import_xml(file):
        list_gspn = []  # list of parsed GSPN objects
        gspn = pn.GSPN()

        tree = et.parse(file)  # parse XML with ElementTree
        root = tree.getroot()

        for petrinet in root.iter('net'):
            n_places = len(list(petrinet.iter('place')))
            place_name = ['']*n_places
            place_marking = [0]*n_places
            id2place = {}

            # iterate over all places of the petri net
            for i, pl in enumerate(petrinet.iter('place')):
                # get place name encoded as 'id' in the pnml structure
                place_name[i] = pl.find('./name/value').text

                text = pl.find('./initialMarking/value').text
                # get place marking encoded inside 'initalMarking', as the 'text' of the key 'value'
                place_marking[i] = int(text.split(',')[-1])

                id2place[pl.get('id')] = place_name[i]

            # add the list of places to the gspn object
            gspn.add_places(name=list(place_name), ntokens=place_marking, set_initial_marking=True)

            n_transitions = len(list(petrinet.iter('transition')))
            transition_name = ['']*n_transitions
            transition_type = ['']*n_transitions
            transition_rate = [0]*n_transitions
            id2transition= {}

            # iterate over all transitions of the petri net
            for i, tr in enumerate(petrinet.iter('transition')):
                # get transition name encoded as 'id' in the pnml structure
                transition_name[i] = tr.find('./name/value').text

                # get the transition type either exponential ('exp') or immediate ('imm')
                if tr.find('./timed/value').text == 'true':
                    transition_type[i] = 'exp'
                else:
                    transition_type[i] = 'imm'

                # get the transition fire rate or weight
                transition_rate[i] = float(tr.find('./rate/value').text)

                id2transition[tr.get('id')] = transition_name[i]

            # add the list of transitions to the gspn object
            gspn.add_transitions(list(transition_name), transition_type, transition_rate)

            arcs_in = {}
            arcs_out = {}

            # iterate over all arcs of the petri net
            for arc in petrinet.iter('arc'):
                source = arc.get('source')
                target = arc.get('target')

                # IN arc connection (from place to transition)
                if source in id2place:
                    pl = id2place[source]
                    tr = id2transition[target]

                    if pl in arcs_in:
                        arcs_in[pl].append(tr)
                    else:
                        arcs_in[pl] = [tr]

                # OUT arc connection (from transition to place)
                elif source in id2transition:
                    tr = id2transition[source]
                    pl = id2place[target]

                    if tr in arcs_out:
                        arcs_out[tr].append(pl)
                    else:
                        arcs_out[tr] = [pl]

            gspn.add_arcs(arcs_in, arcs_out)

            list_gspn.append(gspn)

        return list_gspn

    # @staticmethod
    # def export_xml(gspn, path):
    #     # TODO: EXPORT XML FROM GSPN
    #     return True

    @staticmethod
    def draw_enabled_transitions(gspn, gspn_draw, file='gspn_default', show=True):
        enabled_exp_transitions, random_switch = gspn.get_enabled_transitions()

        if random_switch:
            for transition in random_switch.keys():
                gspn_draw.node(transition, shape='rectangle', style='filled', color='red', label='', xlabel=transition, height='0.2', width='0.6', fixedsize='true')

            gspn_draw.render(file + '.gv', view=show)
        elif enabled_exp_transitions:
            for transition in enabled_exp_transitions.keys():
                gspn_draw.node(transition, shape='rectangle', color='red', label='', xlabel=transition, height='0.2', width='0.6', fixedsize='true')

            gspn_draw.render(file + '.gv', view=show)

        return gspn_draw

    @staticmethod
    def draw_gspn(gspn, file='gspn_default', show=True):

        # ref: https://www.graphviz.org/documentation/
        gspn_draw = Digraph(engine='dot')

        gspn_draw.attr('node', forcelabels='true')

        # draw places and marking
        plcs = gspn.get_current_marking()
        for place, marking in plcs.items():
            if int(marking) == 0:
                gspn_draw.node(place, shape='circle', label='', xlabel=place, height='0.6', width='0.6', fixedsize='true')
            else:
                # places with more than 4 tokens cannot fit all of them inside it
                if int(marking) < 5:
                    lb = '<'
                    for token_number in range(1, int(marking)+1):
                        lb = lb + '&#9899; '
                        if token_number % 2 == 0:
                            lb = lb + '<br/>'
                    lb = lb + '>'
                else:
                    lb = '<&#9899; x ' + str(int(marking)) + '>'

                gspn_draw.node(place, shape='circle', label=lb, xlabel=place, height='0.6', width='0.6', fixedsize='true')

        # draw transitions
        trns = gspn.get_transitions()
        for transition, value in trns.items():
            if value[0] == 'exp':
                gspn_draw.node(transition, shape='rectangle', color='black', label='', xlabel=transition, height='0.2', width='0.6', fixedsize='true')
            else:
                gspn_draw.node(transition, shape='rectangle', style='filled', color='black', label='', xlabel=transition, height='0.2', width='0.6', fixedsize='true')

        # draw edges
        edge_in, edge_out = gspn.get_arcs()
        for iterator, place_index in enumerate(edge_in.coords[0]):
            transition_index = edge_in.coords[1][iterator]
            gspn_draw.edge(gspn.index_to_places[place_index], gspn.index_to_transitions[transition_index])
        for iterator, transition_index in enumerate(edge_out.coords[0]):
            place_index = edge_out.coords[1][iterator]
            gspn_draw.edge(gspn.index_to_transitions[transition_index], gspn.index_to_places[place_index])

        gspn_draw.render(file+'.gv', view=show)

        return gspn_draw

    @staticmethod
    def draw_coverability_tree(cov_tree, file='ct_default', show=True):
        ct_draw = Digraph(engine='dot')
        ct_draw.attr('node', forcelabels='true')

        # draw coverability tree nodes
        for node_id, node_info in cov_tree.nodes.items():
            if node_info[1] == 'T':
                ct_draw.node(node_id, shape='doublecircle', label=node_id, height='0.6', width='0.6', fixedsize='true')
            elif node_info[1] == 'V':
                ct_draw.node(node_id, shape='circle', label=node_id, height='0.6', width='0.6', fixedsize='true')
            elif node_info[1] == 'D':
                ct_draw.node(node_id, shape='doublecircle', label=node_id, height='0.6', width='0.6', fixedsize='true', color="red")

        # draw coverability tree edges
        for edge in cov_tree.edges:
            edge_label = edge[2] + ' (' + str(round(edge[3],2)) + ')'
            ct_draw.edge(edge[0], edge[1], label=edge_label)

        ct_draw.render(file + '.gv', view=show)

        return ct_draw

    @staticmethod
    def draw_ctmc(ctmc, file='ctmc_default', show=True):
        ctmc_draw = Digraph(engine='dot')
        ctmc_draw.attr('node', forcelabels='true')

        # draw cmtc tree nodes
        for node_id, node_info in ctmc.state.items():
            ctmc_draw.node(node_id, shape='doublecircle', label=node_id, height='0.6', width='0.6', fixedsize='true')
        # draw cmtc tree edges
        for edge in ctmc.transition:
            edge_label = str(round(edge[3],2)) + ' (' + str(round(edge[4],2)) + ')'
            ctmc_draw.edge(edge[0], edge[1], label=edge_label)

        ctmc_draw.render(file + '.gv', view=show)

        return ctmc_draw


    @staticmethod
    def expand_pn(parent, child, sym_place):
        '''
        Function that substitutes a place in the parent GSPN with the child GSPN.
        :param parent: A GSPN object, containing a place with the sym_place name, that will be expanded
        :param child: A GSPN object where the input places start with an 'i.' and the output places start with an 'f.'
        :param sym_place: (str) Name of the place to be expanded
        :return: a GSPN object with the expanded Petri net
        '''

        #TODO Find out conflicting transitions, retrieve weights and sample probability

        parent_places = parent.get_current_marking()
        parent_transitions = parent.get_transitions()

        child_places = child.get_current_marking()
        child_transitions = child.get_transitions()

        parent_set = set(parent_places)
        child_set = set(child_places)
        intersection = parent_set.intersection(child_set)

        if len(intersection) != 0:
            raise Exception('Parent and child PNs have places with identical names.')

        parent_set = set(parent_transitions)
        child_set = set(child_transitions)
        intersection = parent_set.intersection(child_set)

        if len(intersection) != 0:
            raise Exception('Parent and child PNs have transitions with identical names.')

        input_places = {}
        output_places = {}

        expanded_pn = pn.GSPN()

        for place in child_places.keys():
            if place.startswith('i.') is True:
                input_places[place] = child_places[place]

            if place.startswith('f.') is True:
                output_places[place] = child_places[place]

        sym_place_marking = parent_places[sym_place]

        for place in input_places:
            child_places[place] = sym_place_marking

        arcs_in, arcs_out = parent.remove_place(sym_place)
        parent_places = parent.get_current_marking()    # Parent places have to be retrieved only after removing place to be expanded

        arc_pin_m, arc_pout_m = parent.get_arcs()
        arc_cin_m, arc_cout_m = child.get_arcs()

        for place in input_places:
            temp = place.replace('i.', '')
            input_places.pop(place)
            input_places[temp] = child_places[place]
            child_places[temp] = child_places.pop(place)
            arc_cin_m.rename(index={place: temp}, inplace=True)
            arc_cout_m.rename(columns={place: temp}, inplace=True)

        for place in output_places:
            temp = place.replace('f.', '')
            output_places.pop(place)
            output_places[temp] = child_places[place]
            child_places[temp] = child_places.pop(place)
            arc_cin_m.rename(index={place: temp}, inplace=True)
            arc_cout_m.rename(columns={place: temp}, inplace=True)

        expanded_pn.add_places_dict(parent_places)
        expanded_pn.add_places_dict(child_places)

        expanded_pn.add_transitions_dict(parent_transitions)
        expanded_pn.add_transitions_dict(child_transitions)

        arc_in_m = pd.concat([arc_pin_m, arc_cin_m],join='outer',sort=False)
        arc_in_m.where(arc_in_m >= 0, 0.0, inplace=True)

        arc_out_m = pd.concat([arc_pout_m, arc_cout_m], join='outer',sort=False)
        arc_out_m.where(arc_out_m >= 0, 0.0, inplace=True)

        for arc in arcs_in:
            for transition in arcs_in[arc]:
                for place in output_places:
                    arc_in_m.loc[place][transition] = 1

        for transition in arcs_out.keys():
            for place in input_places:
                arc_out_m.loc[transition][place] = 1

        expanded_pn.add_arcs_matrices(arc_in_m, arc_out_m)

        return expanded_pn


if __name__ == "__main__":
    import time
    pntools = GSPNtools()
    mypn = pntools.import_xml('rvary_escort_run.xml')[0]

    drawing = pntools.draw_gspn(mypn, 'run_rvary', show=True)
    time.sleep(2)
    pntools.draw_enabled_transitions(mypn, drawing, 'run_rvary', show=True)
