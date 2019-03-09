import gspn as pn
import gspn_tools as tools

myPN = pn.GSPN()

places = myPN.add_places(['p1', 'p2', 'p3', 'p4', 'p5', 'p6'], [1, 0, 0, 1, 0, 0])

transitions = myPN.add_transitions(['t1', 't2', 't3', 't4'], ['imm', 'imm', 'imm', 'imm'], [1, 1, 1, 1])

arc_in = {}
arc_in['p1'] = ['t1']
arc_in['p2'] = ['t2']
arc_in['p3'] = ['t3']
arc_in['p4'] = ['t4']
arc_in['p5'] = ['t4']

arc_out = {}
arc_out['t1'] = ['p2']
arc_out['t2'] = ['p3', 'p4']
arc_out['t3'] = ['p5']
arc_out['t4'] = ['p6']
a, b = myPN.add_arcs(arc_in ,arc_out)


otherPN = pn.GSPN()
places = otherPN.add_places(['i.p_init', 'i.p2_sub', 'p3_sub', 'p4_sub', 'f.p_fin'],[0, 0, 0, 1, 0])
transitions = otherPN.add_transitions(['t1_sub', 't2_sub'], ['imm', 'imm'], [1, 1])
arc_in = {}
arc_in['i.p_init'] = ['t1_sub']
arc_in['i.p2_sub'] = ['t1_sub']
arc_in['p3_sub'] = ['t2_sub']
arc_in['p4_sub'] = ['t2_sub']
arc_out = {}
arc_out['t1_sub'] = ['p3_sub']
arc_out['t2_sub'] = ['f.p_fin']
a, b = otherPN.add_arcs(arc_in, arc_out)

'''
tools.GSPNtools.draw_gspn(myPN, file = 'pn1')
tools.GSPNtools.draw_gspn(otherPN, file = 'pn2')
'''
expanded_pn = tools.GSPNtools.expand_pn(myPN,otherPN,'p4')
tools.GSPNtools.draw_gspn(expanded_pn, file='expandedPN')
