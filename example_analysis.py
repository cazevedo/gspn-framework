import gspn_tools as gst
import gspn_analysis
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
import math
import seaborn as sns
from tqdm import tqdm
import pickle
import pandas as pd

def throughput(draw=False):
    '''
    Throughput and expected number of tokens metrics obtained for some relevant transitions and places.
    '''
    print('\n\n---- Running throughput example ----')

    pntools = gst.GSPNtools()

    print('Importing GSPN Model')
    mypn = pntools.import_pnml('multi_escort_run.xml')[0]

    if draw:
        drawing = pntools.draw_gspn(mypn, file=save_path+'petri_net', show=False)

    print('Generating CT')
    ct_tree = gspn_analysis.CoverabilityTree(mypn)
    ct_tree.generate()
    print('Bounded? : ', ct_tree.boundedness())

    if draw:
        pntools.draw_coverability_tree(ct_tree, file=save_path+'cv_tree', show=False)

    print('Generating CTMC')
    ctmc = gspn_analysis.CTMC(ct_tree)
    ctmc.generate()

    print('CTMC States:\n')

    _, ss = ct_tree.convert_states_to_latex()
    # print(ss)

    if draw:
        pntools.draw_ctmc(ctmc, file=save_path+'ctmc', show=False)

    print('Computing Steady State and Transition Rate ...')
    mypn.init_analysis()

    print('Throughput of R1 appointment attend: ')
    print(mypn.transition_throughput_rate('T6').iloc[0])
    print('Throughput of R2 appointment attend: ')
    print(mypn.transition_throughput_rate('T12').iloc[0])

    print('Expected # tokens Appointment Request')
    print(mypn.expected_number_of_tokens('p.AppointmentRequest').iloc[0])
    print('Expected # tokens No Appointment')
    print(mypn.expected_number_of_tokens('p.NOT_AppointmentRequest').iloc[0])
    print('Expected # tokens R1 Escorting')
    print(mypn.expected_number_of_tokens('t.R1EscortVisitor').iloc[0])
    print('Expected # tokens R1 Charging')
    print(mypn.expected_number_of_tokens('t.R1Charging').iloc[0])
    print('Expected # tokens R2 Escorting')
    print(mypn.expected_number_of_tokens('t.R2EscortVisitor').iloc[0])
    print('Expected # tokens R2 Charging')
    print(mypn.expected_number_of_tokens('t.R2Charging').iloc[0])

def mean_wt_speeds(save_path, draw=False):
    '''
    Mean wait time of place p.AppointmentRequest as a
    function of the firing rates of transitions T7 and T9.
    '''
    print('\n\n---- Running mean wait time example ----')

    pntools = gst.GSPNtools()

    print('Importing GSPN Model')
    mypn = pntools.import_pnml('multi_escort_run.xml')[0]

    if draw:
        drawing = pntools.draw_gspn(mypn, file=save_path+'petri_net', show=False)

    min_escort_rate = 0.1
    max_escort_rate = 10
    escort_rates = np.arange(min_escort_rate, max_escort_rate, 0.1)
    mean_wait_time = []
    bat_trans_rates = []

    for escort_trans_rate in escort_rates:
        # function that defines how the battery discharge rate ('T9') varies as a function of the robot speed ('T7')
        bat_trans_rate = math.pow(escort_trans_rate-5,2)
        bat_trans_rates.append(bat_trans_rate)

        # set the transitions firing rate
        mypn.add_transitions(['T7','T9'], tclass=['exp', 'exp'], trate=[escort_trans_rate, bat_trans_rate])

        mypn.init_analysis()

        mwt = mypn.mean_wait_time('p.AppointmentRequest')
        mwt = mwt.iloc[0]
        mean_wait_time.append(mwt)
        print('Mean Wait Time: ', mwt)

        transitions = mypn.get_transitions()

    print(np.shape(escort_rates))
    print('Escort Rates: ', list(escort_rates))
    print(np.shape(bat_trans_rates))
    print('Bat discharge rates: ', bat_trans_rates)
    print(np.shape(mean_wait_time))
    print('Mean wait time: ', mean_wait_time)

    sns.set()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot3D(escort_rates, bat_trans_rates, mean_wait_time)
    # ax.scatter3D(escort_rates, bat_trans_rates, mean_wait_time)

    ax.set_xlabel('T7 Firing Rate')
    ax.set_ylabel('T9 Firing Rate')
    ax.set_zlabel('\nMean Wait Time Visitor \n (p.AppointmentRequest)')

    # plt.show()
    fig.savefig(save_path+'mean_wait_time.pdf')

def transient_evol(save_path, draw=False):
    '''
    Transient evolution of the probability of the robot being
    available as a function of time, obtained for two scenarios:
    1) a set of batteries cheaper with low performance (slow charge, quick discharge);
    2) a set of batteries more expensive with higher performance (quick charge, slow discharge);
    '''

    print('\n\n---- Running transient evolution example ----')

    pntools = gst.GSPNtools()

    print('Importing GSPN Model')
    mypn = pntools.import_pnml('multi_escort_run.xml')[0]

    if draw:
        drawing = pntools.draw_gspn(mypn, file=save_path+'petri_net', show=False)

    print('Generating CT')
    ct_tree = gspn_analysis.CoverabilityTree(mypn)
    ct_tree.generate()
    print('Bounded? : ', ct_tree.boundedness())

    if draw:
        pntools.draw_coverability_tree(ct_tree, file=save_path+'cv_tree', show=False)

    print('Generating CTMC')
    ctmc = gspn_analysis.CTMC(ct_tree)
    ctmc.generate()

    if draw:
        pntools.draw_ctmc(ctmc, file=save_path+'ctmc', show=False)

    uni_dist = pd.DataFrame(1.0 / len(ctmc.state) * np.ones(len(ctmc.state)))
    uni_dist.index = ctmc.state

    states = ctmc.state.keys()
    states = sorted(states)

    # get the states where the place 'p.R1Available' has 1 token
    # means that robot R1 is available
    enabled_states = []
    for state_name in states:
        marking = ctmc.state[state_name][0]
        if ['p.R1Available', 1] in marking:
            enabled_states.append(state_name)

    # Quick battery charging/Slow discharge
    discharge_rate = 0.1
    charging_rate = 1.0

    # set the transitions firing rate
    mypn.add_transitions(['T3', 'T2'], tclass=['exp', 'exp'], trate=[discharge_rate, charging_rate])

    mypn.init_analysis()

    step = 0.1
    time_window = 50
    prob_available = []
    prob_sum = np.zeros(int(time_window/step))
    for state in enabled_states:
        print('Computing transition evolution for state: ', state)
        prob_avlbl = mypn.transition_probability_evolution(time_window, step, uni_dist, state)
        prob_sum += prob_avlbl
        prob_available.append(prob_avlbl)

    sns.set()
    fig = plt.figure()

    t = np.arange(0, time_window, step)
    plt_pt1 = plt.plot(t, prob_sum, label='Quick battery charging/Slow discharge')

    # Slow battery charging/Quick discharge
    discharge_rate = 1.0
    charging_rate = 0.1

    # set the transitions firing rate
    mypn.add_transitions(['T3', 'T2'], tclass=['exp', 'exp'], trate=[discharge_rate, charging_rate])

    mypn.init_analysis()

    prob_available = []
    prob_sum = np.zeros(int(time_window/step))
    for state in enabled_states:
        print('Computing transition evolution for state: ', state)
        prob_avlbl = mypn.transition_probability_evolution(time_window, step, uni_dist, state)
        prob_sum += prob_avlbl
        prob_available.append(prob_avlbl)


    plt_pt2 = plt.plot(t, prob_sum, label='Slow battery charging/Quick discharge')
    plt.xlabel("Time [s]")
    plt.ylabel("Probability of robot 1 being available")
    plt.legend(loc='lower right')

    # plt.show()
    plt_pt1.savefig(save_path+'prob_available_new.pdf')

def random_switch_tune(save_path, draw=False):
    '''
    Visitor mean wait time as a function of the robot R1
    escort speed, the visitors rate and the probability of robot R1 being
    assigned, instead of R2, to the escort task.
    '''
    print('\n\n---- Running random switch tunning example ----')

    pntools = gst.GSPNtools()

    print('Running Random Switch Tune...')

    print('Importing GSPN Model')
    mypn = pntools.import_pnml('multi_escort_run.xml')[0]

    if draw:
        drawing = pntools.draw_gspn(mypn, file=save_path+'petri_net', show=False)

    list_probs = np.arange(0.1, 1.0, 0.4)
    visitors_rates = np.arange(0.1, 10, 1)
    robot_speeds = np.arange(0.1, 10, 1)
    mean_wait_time = []

    for it in tqdm(range(len(list_probs))):
        mean_wait_time.append( np.zeros((len(visitors_rates), len(robot_speeds))) )

        prob_R1 = list_probs[it]
        prob_R2 = 1 - prob_R1

        mypn.add_transitions(['T6', 'T12'], tclass=['exp', 'exp'], trate=[prob_R1, prob_R2])

        for row_i, visitor_r in enumerate(visitors_rates):
            mypn.add_transitions(['T13'], tclass=['exp'], trate=[visitor_r])

            for column_i, r1_speed in enumerate(robot_speeds):
                mypn.add_transitions(['T7'], tclass=['exp'], trate=[r1_speed])

                mypn.init_analysis()

                mwt = mypn.mean_wait_time('p.AppointmentRequest')
                mwt = mwt.iloc[0]
                mean_wait_time[it][row_i][column_i] = mwt

                print('--------------------------------------------')
                print('R1 Prob: ', prob_R1, ' R2 Prob: ', prob_R2)
                print('Mean Wait Time: ', mwt)

    pickle.dump((visitors_rates, robot_speeds, list_probs, mean_wait_time), open(save_path+"mean_wait_time.p", "wb" ))

def plot_random_switch_tune(save_path):
    '''
    Plot a surface graph with the data obtained in the random_switch_tune function above.
    '''

    print('\n\n---- Plotting random switch data ----')

    (visitors_rates, robot_speeds, list_probs, mean_wait_time) = pickle.load(open(save_path+"mean_wait_time.p", "rb"))

    nsubplots = len(list_probs)

    sns.set()
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for index in range(nsubplots):
        print('Getting mesh grid...')

        X, Y = np.meshgrid(robot_speeds, visitors_rates)

        Z = np.ones_like(X) * list_probs[index]

        F = mean_wait_time[index]
        color_dimension = np.array(F) # change to desired fourth dimension
        minn, maxx = color_dimension.min(), color_dimension.max()
        norm = colors.Normalize(minn, maxx)
        m = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)
        m.set_array([])
        fcolors = m.to_rgba(color_dimension)

        print('Plotting surface...', index)
        surf = ax.plot_surface(X, Y, Z,
                        facecolors=fcolors, shade=False,
                        rstride=1, cstride=1)

    ax.set_xlabel('R1 Escort Speed (T1)')
    ax.set_ylabel('Visitors Rate (T13)')
    ax.set_zlabel('Probability of R1 being assigned (T6)')

    cbar = plt.colorbar(m)
    cbar.set_label('Visitor Mean Wait Time \n (p.AppointmentRequest)', rotation=270, labelpad=25)

    # plt.show()
    fig.savefig(save_path+'prob_select_wait.pdf')

def check_wait_time(save_path):
    '''
    The mean wait time of the place p.AppointmentRequest
    as a function of the number of robots.
    '''

    print('\n\n---- Running mean wait time as a function of number of robots example ----')

    pntools = gst.GSPNtools()

    print('Importing GSPN Model')
    mypn = pntools.import_pnml('rvary_escort_run.xml')[0]

    n_robots = []
    mean_wt = []

    # Get base gspn
    base_tr = mypn.get_transitions()
    base_pl = mypn.get_current_marking()
    base_arc_in, base_arc_out  = mypn.get_arcs_dict()

    mypn.add_places(['p.AppointmentRequest', 'p.NOT_AppointmentRequest'], [0, 1], set_initial_marking=True)

    mypn.add_transitions(['T100'], ['exp'], [10])
    arc_in = {}
    arc_in['p.NOT_AppointmentRequest'] = ['T100']
    arc_out = {}
    arc_out['T100'] = ['p.AppointmentRequest']
    mypn.add_arcs(arc_in, arc_out)

    arc_in = {}
    arc_in['p.AppointmentRequest'] = ['T6']
    arc_out = {}
    arc_out['T6'] = ['p.NOT_AppointmentRequest']
    mypn.add_arcs(arc_in, arc_out)

    mypn.init_analysis()
    mwt = mypn.mean_wait_time('p.AppointmentRequest')
    mwt = mwt.iloc[0]
    print('--------------------------------------------')
    print('# Robots: ', 1)
    print('Mean Wait Time: ', mwt)
    n_robots.append(1)
    mean_wt.append(mwt)

    n_extra_robots = 4
    for robot_i in tqdm(range(n_extra_robots)):
        new_tr = {k + str(robot_i): v for k, v in base_tr.items()}
        new_pl = {k + str(robot_i): v for k, v in base_pl.items()}
        new_arc_in = {k + str(robot_i): [s + str(robot_i) for s in v] for k, v in base_arc_in.items()}
        new_arc_out = {k + str(robot_i): [s + str(robot_i) for s in v] for k, v in base_arc_out.items()}

        mypn.add_transitions_dict(new_tr)
        mypn.add_places_dict(new_pl, set_initial_marking=True)
        mypn.add_arcs(new_arc_in, new_arc_out)

        arc_in = {}
        arc_in['p.AppointmentRequest'] = ['T6'+str(robot_i)]
        arc_out = {}
        arc_out['T6'+str(robot_i)] = ['p.NOT_AppointmentRequest']
        mypn.add_arcs(arc_in, arc_out)

        mypn.init_analysis()
        mwt = mypn.mean_wait_time('p.AppointmentRequest')
        mwt = mwt.iloc[0]
        print('--------------------------------------------')
        print('# Robots: ', robot_i+2)
        print('Mean Wait Time: ', mwt)

        n_robots.append(robot_i+2)
        mean_wt.append(mwt)

    pickle.dump((n_robots, mean_wt), open(save_path+"wait_n_robots.p", "wb"))

def plot_check_wait_time(save_path):
    '''
    Plot the data obtained above, in the check_wait_time function.
    '''

    print('\n\n---- Plotting mean wait time as a function of number of robots data ----')

    (n_robots, mean_wt) = pickle.load(open(save_path+"wait_n_robots.p", "rb"))

    # n_robots = [1,2,3,4,5,6,7]
    # mean_wt = [1.4084022038567503,
    #            0.6577021777334127,
    #            0.40774902102982863,
    #            0.28301730812930426,
    #            0.20840737442629714,
    #            0.1588914057248218,
    #            0.12374869549075418]

    sns.set()
    fig = plt.figure()

    plt.xticks(n_robots)

    plt.plot(n_robots, mean_wt, '-o')

    plt.xlabel("Number of Robots")
    plt.ylabel("Mean Wait Time of the Visitors \n (p.AppointmentRequest)")

    # plt.show()
    fig.savefig(save_path+'n_robots_wait.pdf')



if __name__ == "__main__":
    save_path = '/home/cazevedo/gspn-framework/'

    '''
    Obtain the throughput and expected number of tokens metrics obtained for some relevant transitions and places.
    '''
    throughput(draw=True)

    '''
    Obtain the mean wait time of place p.AppointmentRequest as a
    function of the firing rates of transitions T7 and T9.
    '''
    mean_wt_speeds(save_path, draw=False)

    '''
    Obtain the transient evolution of the probability of the robot being
    available as a function of time, obtained for two scenarios:
    1) a set of batteries cheaper with low performance (slow charge, quick discharge);
    2) a set of batteries more expensive with higher performance (quick charge, slow discharge);
    '''
    transient_evol(save_path, draw=False)

    '''
    Obtain the visitor mean wait time as a function of the robot R1
    escort speed, the visitors rate and the probability of robot R1 being
    assigned, instead of R2, to the escort task.
    '''
    random_switch_tune(save_path, draw=False)
    plot_random_switch_tune(save_path)

    '''
    Obtain the mean wait time of the place p.AppointmentRequest
    as a function of the number of robots.
    '''
    check_wait_time(save_path)
    plot_check_wait_time(save_path)