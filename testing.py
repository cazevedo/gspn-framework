import gspn as pn

myPN = pn.GSPN()

places = myPN.add_places(['p1','p2','p3','p4','p5'])

transitions = myPN.add_transitions(['t1','t2','t3','t4'],['imm','imm','imm','imm','imm'],[1, 1, 0.5, 0.5])

arc_in = {}
arc_in['p1'] = ['t1']
arc_in['p4'] = ['t4','t3','t2']
arc_in['p2'] = ['t2']

arc_out = {}
arc_out['t1'] = ['p4','p2']
arc_out['t2'] = ['p3']
arc_out['t3'] = ['p5']
arc_out['t4'] = ['p5']

in_arc, out_arc = myPN.add_arcs(arc_in,arc_out)


print('Places: ' , myPN.get_current_marking(), '\n')
print('Trans: ' , myPN.get_transitions(), '\n')
arcs_in , arcs_out = myPN.get_arcs()
print('Arcs IN: ' , arcs_in, '\n')
print('Arcs OUT: ' , arcs_out, '\n')

myPN.remove_transition('t3')

print('Places: ' , myPN.get_current_marking(), '\n')
print('Trans: ' , myPN.get_transitions(), '\n')
arcs_in , arcs_out = myPN.get_arcs()
print('Arcs IN: ' , arcs_in, '\n')
print('Arcs OUT: ' , arcs_out, '\n')