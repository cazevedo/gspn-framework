import gspn_tools as gst
import time
import warnings

warnings.filterwarnings("ignore")

pntools = gst.GSPNtools()
nets = pntools.import_pnml('debug/pipediag.xml')
mypn = nets[0]

# print('inital : ', mypn.get_current_marking())
step = 1
try:
    # while True:
    for i in range(40):
        drawing = pntools.draw_gspn(mypn, 'mypn', show=False)
        # time.sleep(1)
        pntools.draw_enabled_transitions(mypn, drawing, 'mypn_enabled', show=True)
        # time.sleep(1)
        current_marking = mypn.simulate(nsteps=1, reporting_step=1, simulate_wait=True)
        # print('Step # : ', step)
        # print(current_marking)
        # raw_input("")
        step = step + 1
except KeyboardInterrupt:
    print('ended')