#! /usr/bin/env python3
from gspn_lib import gspn

if __name__ == "__main__":
    my_pn = gspn.GSPN()

    # the initial marking is set when the places are added
    places = my_pn.add_places(['p1', 'p2', 'p3', 'p4'], [1, 0, 1, 0])
    trans = my_pn.add_transitions(['t1', 't2', 't3', 't4'], ['imm', 'imm', 'imm', 'exp'], [5, 4, 1, 1])
    trans = my_pn.add_transitions(['t5', 't6', 't7'], ['imm', 'imm', 'exp'], [0.6, 0.4, 1])
    arc_in = {}
    arc_in['p1'] = ['t1', 't2', 't3']
    arc_in['p2'] = ['t4']
    arc_in['p3'] = ['t5', 't6']
    arc_in['p4'] = ['t7']
    arc_out = {}
    arc_out['t1'] = ['p2']
    arc_out['t2'] = ['p2']
    arc_out['t3'] = ['p2']
    arc_out['t4'] = ['p1']

    arc_out['t5'] = ['p4']
    arc_out['t6'] = ['p4']
    arc_out['t7'] = ['p3']
    a, b = my_pn.add_arcs(arc_in, arc_out)

    print('Initial Marking')
    print(my_pn.get_current_marking())
    print()

    my_pn.fire_transition(transition='t2')
    print('Marking after firing transition t2')
    print(my_pn.get_current_marking())
    print()

    my_pn.fire_transition(transition='t5')
    print('Marking after firing transition t5')
    print(my_pn.get_current_marking())
    print()

    my_pn.reset()
    print('Marking after GSPN reset = Initial Marking')
    print(my_pn.get_current_marking())
    print()
