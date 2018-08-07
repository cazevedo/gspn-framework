import gspn_tools as gst
# import gspn
import time
import warnings

warnings.filterwarnings("ignore")

pntools = gst.gspn_tools()
nets = pntools.import_pnml('debug/pipediag.xml')
mypn = nets[0]

print('inital : ', mypn.get_current_marking())
step = 1
try:
    while True:
        drawing = pntools.show_gspn(mypn)
        time.sleep(0.5)
        pntools.show_enabled_transitions(mypn, drawing)
        current_marking = mypn.execute()
        print('Step # : ', step)
        print(current_marking)
        raw_input("")
        step = step + 1
except KeyboardInterrupt:
    print('ended')