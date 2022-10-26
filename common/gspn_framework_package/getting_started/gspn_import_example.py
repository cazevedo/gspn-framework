#! /usr/bin/env python3
from gspn_framework_package import gspn_tools

if __name__ == "__main__":

    pn_tool = gspn_tools.GSPNtools()
    my_pn = pn_tool.import_greatspn('gspn_example.PNPRO')[0]

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
