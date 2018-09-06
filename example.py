import gspn_tools as gst
import time
import warnings
import gspn_analysis

warnings.filterwarnings("ignore")

pntools = gst.GSPNtools()
# nets = pntools.import_pnml('debug/pipediag.xml')
nets = pntools.import_pnml('debug/performance_example.xml')
# nets = pntools.import_pnml('debug/simple_test.xml')
mypn = nets[0]
tr = mypn.prob_of_n_tokens('P1', 0)
print(tr)

# print(mypn.transition_throughput_rate('T2'))



# drawing = pntools.draw_gspn(mypn, 'mypn', show=True)
# # pntools.draw_enabled_transitions(mypn, drawing, 'mypn_enabled', show=True)
#
# ct_tree = gspn_analysis.CoverabilityTree(mypn)
# ct_tree.generate()
# ctmc = gspn_analysis.CTMC(ct_tree)
# ctmc.generate()
#
# tr = ctmc.compute_transition_rate()
# arc_in = {}
# arc_in['M0'] = 0.5
# arc_in['M11'] = 0.1
# arc_in['M3'] = 0.1
# arc_in['M4'] = 0.1
# arc_in['M6'] = 0.1
# arc_in['M9'] = 0.1
# tr = ctmc.get_prob_reach_states(arc_in,10)

# ctmc.get_steady_state()
# print(tr)

# tr = ctmc.compute_transition_probability(10)
# print(ctmc.transition_probability)
# tr = ctmc.get_sojourn_times()
# print(tr)
# tr = ctmc.create_infinitesimal_generator()
#
#
# pntools.draw_coverability_tree(ct_tree)
# pntools.draw_ctmc(ctmc)

# print('inital : ', mypn.get_current_marking())
# print('inital : ', mypn.get_initial_marking())
step = 1
try:
    # while True:
    for i in range(1):
        # print('DRAWING')
        # drawing = pntools.draw_gspn(mypn, 'mypn', show=False)
        # pntools.draw_enabled_transitions(mypn, drawing, 'mypn_enabled', show=True)
        # time.sleep(2)
        # current_marking = mypn.simulate(nsteps=1, reporting_step=1, simulate_wait=False)
        # print('Step # : ', step)
        # print(current_marking)
        # raw_input("")
        step = step + 1
except KeyboardInterrupt:
    print('ended')