import gspn_tools as gst
import time
import warnings
import gspn_analysis as analyse

warnings.filterwarnings("ignore")

pntools = gst.GSPNtools()
# nets = pntools.import_pnml('debug/pipediag.xml')
nets = pntools.import_pnml('debug/performance_example.xml')
# nets = pntools.import_pnml('debug/simple_test.xml')
mypn = nets[0]

pn_al = analyse.AnalyseGSPN(mypn)
ct_nodes, ct_edges = pn_al.coverability_tree()
# print(tree)

# print('inital : ', mypn.get_current_marking())
# print('inital : ', mypn.get_initial_marking())
step = 1
try:
    # while True:
    for i in range(1):
        # print('DRAWING')
        # drawing = pntools.draw_gspn(mypn, 'mypn', show=False)
        # pntools.draw_enabled_transitions(mypn, drawing, 'mypn_enabled', show=True)
        # time.sleep(1)
        # current_marking = mypn.simulate(nsteps=1, reporting_step=1, simulate_wait=False)
        # print('Step # : ', step)
        # print(current_marking)
        # raw_input("")
        step = step + 1
except KeyboardInterrupt:
    print('ended')