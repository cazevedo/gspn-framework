import gspn as pn
import xml.etree.ElementTree as et  # XML parser
from graphviz import Digraph

class GSPNtools(object):
    @staticmethod
    def import_pnml(file):
        list_gspn = []  # list of parsed GSPN objects
        gspn = pn.GSPN()

        tree = et.parse(file)  # parse XML with ElementTree
        root = tree.getroot()

        for petrinet in root.iter('net'):

            place_name = []
            place_marking = []
            for pl in petrinet.iter('place'):  # iterate over all places of the petri net
                place_name.append(pl.get('id'))  # get place name encoded as 'id' in the pnml structure

                text = pl.find('./initialMarking/value').text
                place_marking.append(int(text.split(',')[-1]))  # get place marking encoded inside 'initalMarking', as the 'text' of the key 'value'

            gspn.add_places(place_name, place_marking, True)  # add the compiled list of places to the gspn object

            transition_name = []
            transition_type = []
            transition_rate = []
            for tr in petrinet.iter('transition'):  # iterate over all transitions of the petri net
                transition_name.append(tr.get('id'))  # get transition name encoded as 'id' in the pnml structure

                if tr.find('./timed/value').text == 'true':  # get the transition type either exponential ('exp') or immediate ('imm')
                    transition_type.append('exp')
                else:
                    transition_type.append('imm')

                transition_rate.append(float(tr.find('./rate/value').text))  # get the transition fire rate or weight

            gspn.add_transitions(transition_name, transition_type, transition_rate)  # add the compiled list of transitions to the gspn object

            place_name = gspn.get_current_marking()
            place_name = place_name.keys()
            transition_name = gspn.get_transitions()
            transition_name = transition_name.keys()

            arcs_in = {}
            arcs_out = {}

            for arc in petrinet.iter('arc'):  # iterate over all arcs of the petri net
                src = arc.get('source')
                trg = arc.get('target')

                if src in place_name:  # IN arc connection (from place to transition)
                    if src in arcs_in:
                        arcs_in[src].append(trg)
                    else:
                        arcs_in[src] = [trg]

                elif src in transition_name:  # OUT arc connection (from transition to place)
                    if src in arcs_out:
                        arcs_out[src].append(trg)
                    else:
                        arcs_out[src] = [trg]

            gspn.add_arcs(arcs_in, arcs_out)

            list_gspn.append(gspn)

        return list_gspn

    @staticmethod
    def export_pnml(gspn):
        # TODO: EXPORT PNML FROM GSPN
        return True

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

        for place in edge_in.index:
            for transition in edge_in.columns:
                if edge_in.loc[place][transition] > 0:
                    gspn_draw.edge(place, transition)

        for place in edge_out.columns:
            for transition in edge_out.index:
                if edge_out.loc[transition][place] > 0:
                    gspn_draw.edge(transition, place)

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

        print('rendering CT')
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

if __name__ == "__main__":
    import time
    pntools = GSPNtools()
    nets = pntools.import_pnml('debug/pipediag.xml')
    mypn = nets[0]

    drawing = pntools.draw_gspn(mypn, 'pipediag', show=True)
    time.sleep(2)
    pntools.draw_enabled_transitions(mypn, drawing, 'pipediag', show=True)
