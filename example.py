import gspn_tools as gst
import time
import warnings
import gspn_analysis
# import matplotlib.pyplot as plt
import pylab as plt
import numpy as np

warnings.filterwarnings("ignore")

pntools = gst.GSPNtools()
nets = pntools.import_pnml('debug/escort_visitor4.xml')
mypn = nets[0]

ct_tree = gspn_analysis.CoverabilityTree(mypn)
ct_tree.generate()
# ctmc = gspn_analysis.CTMC(ct_tree)
# ctmc.generate()
print(len(ct_tree.nodes))
print(len(ct_tree.edges))

# print(ct_tree.boundness())

# mid = '577'
# print(ct_tree.nodes[mid])
# for i in ct_tree.edges:
#     if i[1] == mid:
#         print('IN : ', i)
#
# for i in ct_tree.edges:
#     if i[0] == mid:
#         print('OUT : ', i)

# print(ct_tree.nodes)
# print(ct_tree.edges)

# drawing = pntools.draw_gspn(mypn, 'escort_visitor_multi_gspn', show=True)
# pntools.draw_coverability_tree(ct_tree, 'escort_visitor_multi_ct_tree',show=True)
# pntools.draw_ctmc(ctmc, 'escort_visitor_multi_ctmc',show=True)


'''
count = 1
while(count > 0):
    pntools = gst.GSPNtools()
    # nets = pntools.import_pnml('debug/pipediag.xml')
    # nets = pntools.import_pnml('debug/performance_example.xml')
    nets = pntools.import_pnml('debug/escort_visitor3.xml')
    # nets = pntools.import_pnml('debug/parallel.xml')
    # nets = pntools.import_pnml('debug/simple_test.xml')
    mypn = nets[0]
    ct_tree = gspn_analysis.CoverabilityTree(mypn)
    ct_tree.generate()
    ctmc = gspn_analysis.CTMC(ct_tree)
    ctmc.generate()

    # drawing = pntools.draw_gspn(mypn, 'mypn', show=True)
    # pntools.draw_coverability_tree(ct_tree)
    # pntools.draw_ctmc(ctmc)

    for id, marking in ct_tree.nodes.items():
        print(id, marking)

    uni_dist = {}
    for state in ctmc.state.keys():
        uni_dist[state] = 1.0/len(ctmc.state)

    ctmc.compute_transition_rate()
    print(ctmc.get_steady_state())

    mypn.init_analysis()

    mw = mypn.mean_wait_time('t.R1EscortVisitor')
    print('Mean wait time: ', mw)
    # print(mypn.transition_throughput_rate('T9'))
    # print(mypn.transition_throughput_rate('T3'))
    # if count == 2:
    #     prob_available_1 = mypn.transition_probability_evolution(50,0.1, uni_dist, 'M3')
    #     prob_available_2 = mypn.transition_probability_evolution(50,0.1, uni_dist, 'M5')
    #     fprob1 = np.array(prob_available_1[1:]) + np.array(prob_available_2[1:])
    # elif count == 1:
    #     prob_available_1 = mypn.transition_probability_evolution(50,0.1, uni_dist, 'M3')
    #     prob_available_2 = mypn.transition_probability_evolution(50,0.1, uni_dist, 'M5')
    #     fprob2 = np.array(prob_available_1[1:]) + np.array(prob_available_2[1:])
    #
    # name = raw_input("Press enter")
    count=count-1

pntools = gst.GSPNtools()
# nets = pntools.import_pnml('debug/pipediag.xml')
# nets = pntools.import_pnml('debug/performance_example.xml')
nets = pntools.import_pnml('debug/escort_visitor3.xml')
# nets = pntools.import_pnml('debug/parallel.xml')
# nets = pntools.import_pnml('debug/simple_test.xml')
mypn = nets[0]

mw = []
for i in np.linspace(0.001,1, 1000):
    mypn.add_transitions(['T3'], ['exp'], [i])

    mypn.init_analysis()

    mw.append(mypn.mean_wait_time('t.R1EscortVisitor'))

throughput = np.linspace(0.001,1, 1000)
fig = plt.figure()
plt_pt1 = plt.plot(throughput, mw)
plt.xlabel("Transition T3 throughput rate")
plt.ylabel("Mean wait time of t.R1EscortVisitor place")
# plt.legend(loc='lower right')

plt.show()

# print(len(fprob1))
# plt.plot(range(len(fprob1)),fprob1, fprob2)
# t = np.linspace(0,50,500)
# t = t[:-1]
# fig = plt.figure()
# plt_pt1 = plt.plot(t, fprob1, label='Quick battery charging/Slow discharge')
# plt_pt2 = plt.plot(t, fprob2, label='Slow battery charging/Quick discharge')
# plt.xlabel("Time [s]")
# plt.ylabel("Probability of robot 1 being available")
# plt.legend(loc='lower right')
#
# plt.show()



# pl = mypn.get_current_marking()
# pl2 = []
# for k,v in pl.items():
#     pl2.append(k)
# pl2.sort()
#
# for i in pl2:
#     print(i, mypn.expected_number_of_tokens(i))

# mypn.transition_throughput_rate('T3')
# a = mypn.mean_wait_time('P9')
# print(a)


# print(mypn.transition_throughput_rate('T2'))


#
# drawing = pntools.draw_gspn(mypn, 'mypn', show=True)
# pntools.draw_enabled_transitions(mypn, drawing, 'mypn_enabled', show=True)
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
'''