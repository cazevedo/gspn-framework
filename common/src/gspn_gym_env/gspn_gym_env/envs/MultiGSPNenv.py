import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from gspn_framework_package import gspn_tools

class MultiGSPNenv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, gspn_model=None, gspn_path=None, n_locations=None, n_robots=None,
                 set_actions=None, use_expected_time=False, verbose=False, idd=None):
        self.id = idd
        self.verbose = verbose
        # print('Multi GSPN Gym Env')

        if gspn_path != None:
            pn_tool = gspn_tools.GSPNtools()
            self.mr_gspn = pn_tool.import_greatspn(gspn_path)[0]
            # pn_tool.draw_gspn(mr_gspn)
        elif gspn_model != None:
            self.mr_gspn = gspn_model
        else:
            raise Exception('Please provide a GSPN object or a GSPN path of the environment model.')
        self.timestamp = 0

        # [max_n_tokens_in_place0, max_n_tokens_in_place1, ... max_n_tokens_in_placen]
        # we approximate this to: [n_robots, n_robots, ... nrobots]
        self.observation_space = spaces.MultiDiscrete(nvec=[n_robots]*len(self.mr_gspn.get_current_marking()))

        # # [0.0...1.0]
        # self.action_space = spaces.Box(low=0.0, high=1.0,
        #                                shape=(1,), dtype=np.float32)

        if set_actions:
            # when the number of robots (tokens) is smaller than the number of locations (places/transitions)
            # the most efficient approach is define a set of actions that are common to every robot
            # and replicate it with different names for each robot
            self.robots_map = {}
            self.robots_map_inv = {}
            self.set_actions = set_actions
            n_actions = len(set_actions) * n_robots
            m_zero = self.mr_gspn.get_sparse_marking()

            r_number = 0
            for pl, token in m_zero.items():
                if 'Available' not in pl and 'Global' not in pl:
                    self.robots_map[pl] = r_number
                    self.robots_map_inv[r_number] = pl
                    r_number += 1
        else:
            self.robots_map = None

            # get number of transitions in order to get number of actions
            # when the number of robots (tokens) is considerably bigger than the number of locations (places/transitions)
            # the most efficient approach is to use every single transition as an individual action
            imm_transitions = self.mr_gspn.get_imm_transitions()
            actions = imm_transitions.copy()
            for tr_name, tr_rate in imm_transitions.items():
                if tr_rate != 0:
                    del actions[tr_name]

            n_actions = len(actions.keys())

        self.enabled_parallel_transitions = {}
        # # {0,1,...,n_actions}
        self.action_space = spaces.Discrete(n_actions)

        self.use_expected_time = use_expected_time

        self.n_locations = n_locations
        if n_locations != None:
            all_actions = set(range(3 * self.n_locations))
            self.optimal_actions = all_actions - set(3 * np.array(range(self.n_locations)))

    def step(self, action):
        # get disabled actions in current state
        disabled_actions_names, disabled_actions_indexes = self.get_disabled_actions()

        # get current state
        current_state = self.get_current_state()
        if self.verbose:
            print('S: ', current_state)
            print('Enabled Timed transitions : ', self.enabled_parallel_transitions)

        # map input action to associated transition
        transition = self.action_to_transition(action)
        if self.verbose:
            print('Action: ', action)

        if transition != None:
            # apply action
            self.mr_gspn.fire_transition(transition)

            # # get reward
            # reward = self.reward_function(current_state, transition)

            # get execution time (until the next decision state)
            # get also the sequence of the fired transitions ['t1', 't2', ...]
            elapsed_time, fired_transitions = self.execute_actions(use_expected_time=self.use_expected_time)

            # in a MRS the fired timed transition may not correspond to the selected action
            # this is the expected time that corresponds to the selected action
            # action_expected_time = self.get_action_time(action)
            action_expected_time = self.get_action_time_noiseless(action)
            # action_expected_time = 1.0 / transition_rate

            # get reward
            reward = self.reward_function(current_state, transition, fired_transitions)

            self.timestamp += elapsed_time
        else:
            raise Exception('Disabled transition selected! This is not possible.')

            if self.verbose:
                print('Transition not enabled')
            # stay in the same state, return reward -1, timestamp 0
            # reward -1 to discourage actions that do not change the system state

            reward = -1
            # actions_info = ('action-not-available_'+str(action), -1)
            action_expected_time = 0

        if self.verbose:
            print('Reward: ', reward)
            print('Timestamp: ', self.timestamp)
            print('Action expected time: ', action_expected_time)
            print("S actions disabled: ", disabled_actions_names)

        # get enabled actions in the next state
        next_state_enabled_actions_names, next_state_enabled_actions_indexes = self.get_enabled_actions()

        # get next state
        next_state = self.marking_to_state()
        # next_state_string = self.get_current_state()
        if self.verbose:
            print("S': ", self.get_current_state())
            print("S' available actions: ", next_state_enabled_actions_names)
            print()

        episode_done = False

        return next_state, reward, episode_done, \
               {'timestamp': self.timestamp,
                'disabled_actions': (disabled_actions_names, disabled_actions_indexes),
                'next_state_enabled_actions': (next_state_enabled_actions_names, next_state_enabled_actions_indexes),
                'action_time': action_expected_time}
                # 'next_state_string': next_state_string}

    def reset(self):
        self.timestamp = 0
        self.mr_gspn.reset_simulation()
        next_state = self.marking_to_state()
        self.enabled_parallel_transitions = {}

        # get enabled actions in the next state
        next_state_enabled_actions_names, next_state_enabled_actions_indexes = self.get_enabled_actions()

        return next_state, {'timestamp': self.timestamp, 'actions_info': [],
                               'disabled_actions': (None, None),
                               'next_state_enabled_actions': (
                               next_state_enabled_actions_names, next_state_enabled_actions_indexes),
                               'action_time': None}

    def render(self, mode='human'):
        print('rendering not implemented')
        return True

    def close(self):
        self.reset()
        # print('Au Revoir Shoshanna!')

    def get_current_state(self):
        sparse_state = self.mr_gspn.get_current_marking(sparse_marking=True)
        # current_state = list(sparse_state.keys())[0]

        return sparse_state

    def action_to_transition(self, action):
        if self.robots_map:
            # round_nearest(n_actions * (action_multiplier - robot_number))
            action_index = np.round(len(self.set_actions)*(action/len(self.set_actions)
                                                           - np.floor(action/len(self.set_actions))), 0)
            robot_number = np.floor(action/len(self.set_actions))

            robot_loc = self.robots_map_inv[robot_number]
            if 'Nav' in robot_loc or 'Inp' in robot_loc:
                return None
            else:
                robot_loc_index = int(self.robots_map_inv[robot_number].split('L')[-1])

                transition = '_'+str(int(len(self.set_actions)*robot_loc_index + action_index))

                return transition
        else:
            action = self.from_index_to_action(int(action))

            # check if action exists in the enabled transitions; if don't fire any transition
            _, enabled_actions = self.mr_gspn.get_enabled_transitions()

            if action in enabled_actions.keys():
                return action
            else:
                return None

    def marking_to_state(self):
        # map dict marking to list marking
        marking_dict = self.mr_gspn.get_current_marking(sparse_marking=True)
        state = [0]*len(self.mr_gspn.get_current_marking().keys())
        for place_name, number_robots in marking_dict.items():
            token_index = self.mr_gspn.places_to_index[place_name]
            state[token_index] = number_robots

        return state

    def reward_function(self, sparse_state=None, transition=None, fired_transitions=None):
        reward = 0.0

        # robot scalability
        tr_index = self.from_action_to_index(transition)
        location_index = int(tr_index/3.0)
        # when this condition is true it means the mrs decided to inspect
        if location_index != tr_index/3.0:
            reward = 10.0

        # # robot scalability
        # for action in fired_transitions:
        #     if action == 'AllAvailable':
        #         reward = 10.0

        # robot scalability 2
        # tr_index = self.from_action_to_index(transition)
        # location_index = int(tr_index/3.0)
        # # when this condition is true it means the mrs decided to inspect
        # if location_index != tr_index/3.0:
        #     # easier for comparison reasons
        #     enabled_tr_str = ''.join(self.enabled_parallel_transitions.keys())
        #     # when this condition is true it means the mrs decided to inspect the left (L) panel
        #     if tr_index > location_index*3+1:
        #         # this means the panel needed inspection
        #         if 'NeedsInspAgainL'+str(location_index)+'L' not in enabled_tr_str:
        #             reward = 10.0
        #     # otherwise it means the mrs decided to inspect the right (R) panel
        #     else:
        #         # this means the panel needed inspection
        #         if 'NeedsInspAgainL'+str(location_index)+'R' not in enabled_tr_str:
        #             reward = 10.0

        # inspection test
        # if 'L4' in sparse_state.keys() and transition == '_6':
        #     reward = 10

        # inspection 1
        # if 'L4' in sparse_state.keys() and transition == '_6':
        #     reward = 1
        # elif 'L3' in sparse_state.keys() and transition == '_7':
        #     reward = 1

        # if 'FullL4' in sparse_state.keys() and 'FullL3' in sparse_state.keys() and transition == '_9':
        #     reward = 1

        return reward

    def fire_timed_transitions(self, enabled_timed_transitions, use_expected_time):
        if use_expected_time:
            # convert the rate into expected time and store that transition if it was not already stored
            for tr_name, tr_rate in enabled_timed_transitions.copy().items():
                if tr_name not in self.enabled_parallel_transitions:
                    self.enabled_parallel_transitions[tr_name] = [1.0 / tr_rate]

                n_sampled_times = len(self.enabled_parallel_transitions[tr_name])
                tr_index = self.mr_gspn.transitions_to_index[tr_name]
                arcs_in = self.mr_gspn.get_arc_in_m()
                places_dict = self.mr_gspn.get_places()
                input_place_ratios = []
                sample_new_time = True
                for i, tr_coord in enumerate(arcs_in.coords[1]):
                    if tr_coord == tr_index:
                        place_index = arcs_in.coords[0][i]
                        place_name = self.mr_gspn.index_to_places[place_index]
                        n_tokens = places_dict[place_name]
                        arc_weight = arcs_in.data[i]
                        ratio = int(n_tokens/arc_weight)
                        # the ratio gives us the number of sampled times that must exist in the
                        # parallel dict, for this specific transition
                        input_place_ratios.append(ratio)
                        if ratio <= n_sampled_times:
                            sample_new_time = False
                            break
                # sample the amount necessary such that the number of
                # sampled times equals the smallest the place ratio
                if sample_new_time and len(input_place_ratios) > 0:
                    while len(self.enabled_parallel_transitions[tr_name]) < min(input_place_ratios):
                        self.enabled_parallel_transitions[tr_name].append(1.0 / tr_rate)

        else:
            # convert the rate into sampled elapsed time
            # sample from each exponential distribution prob_dist(x) = lambda * exp(-lambda * x)
            # in this case the beta rate parameter is used instead, where beta = 1/lambda
            # store enabled transition if it was not already stored
            for tr_name, tr_rate in enabled_timed_transitions.copy().items():
                if tr_name not in self.enabled_parallel_transitions:
                    self.enabled_parallel_transitions[tr_name] = [np.random.exponential(scale=(1.0 / tr_rate),
                                                                                        size=None)]

                n_sampled_times = len(self.enabled_parallel_transitions[tr_name])
                tr_index = self.mr_gspn.transitions_to_index[tr_name]
                arcs_in = self.mr_gspn.get_arc_in_m()
                places_dict = self.mr_gspn.get_places()
                input_place_ratios = []
                sample_new_time = True
                for i, tr_coord in enumerate(arcs_in.coords[1]):
                    if tr_coord == tr_index:
                        place_index = arcs_in.coords[0][i]
                        place_name = self.mr_gspn.index_to_places[place_index]
                        n_tokens = places_dict[place_name]
                        arc_weight = arcs_in.data[i]
                        ratio = int(n_tokens / arc_weight)
                        # the ratio gives us the number of sampled times that must exist in the
                        # parallel dict, for this specific transition
                        input_place_ratios.append(ratio)
                        if ratio <= n_sampled_times:
                            sample_new_time = False
                            break
                # sample the amount necessary such that the number of
                # sampled times equals the smallest the place ratio
                if sample_new_time and len(input_place_ratios) > 0:
                    while len(self.enabled_parallel_transitions[tr_name]) < min(input_place_ratios):
                        self.enabled_parallel_transitions[tr_name].append(np.random.exponential(scale=(1.0 / tr_rate),
                                                                                                size=None))
        # delete the transitions that were enabled, didn't fire and are not longer enabled
        disabled_transitions = set(self.enabled_parallel_transitions.keys())-set(enabled_timed_transitions.keys())
        for tr_name in disabled_transitions:
            del self.enabled_parallel_transitions[tr_name]

        # select the transition with the lowest execution time
        execution_time = np.inf
        for tr_name, tr_time in self.enabled_parallel_transitions.items():
            new_min_time = min(tr_time)
            if new_min_time < execution_time:
                timed_transition = tr_name
                execution_time = new_min_time

        transitions_to_fire = []
        transitions_to_fire.append(timed_transition)

        # delete transition to be fired
        if len(self.enabled_parallel_transitions[timed_transition]) > 1:
            self.enabled_parallel_transitions[timed_transition].remove(execution_time)
        else:
            del self.enabled_parallel_transitions[timed_transition]

        # decreased elapsed time for the remaining enabled transitions
        for tr_name, tr_exp_time in self.enabled_parallel_transitions.copy().items():
            new_tr_time = list(np.array(tr_exp_time) - execution_time)
            if any(i <= 0 for i in new_tr_time):
                # if some enabled transition has zero time remaining, fire it also
                # according to PN formalism this should not happen
                # instead we should sum a very small time (e.g. 1e-6)
                # to ensure that only 1 transition fires at each time
                # when using expected time this arises more often
                pruned_new_tr_time = []
                for remaining_time in new_tr_time:
                    if remaining_time <= 0:
                        transitions_to_fire.append(tr_name)
                    else:
                        pruned_new_tr_time.append(remaining_time)
                # pruned_new_tr_time = [i for i in new_tr_time if i > 0]
                if len(pruned_new_tr_time) > 0:
                    self.enabled_parallel_transitions[tr_name] = pruned_new_tr_time
                else:
                    del self.enabled_parallel_transitions[tr_name]
            else:
                self.enabled_parallel_transitions[tr_name] = new_tr_time

        for transition_name in transitions_to_fire:
            self.mr_gspn.fire_transition(transition_name)

        return execution_time, transitions_to_fire

    def execute_actions(self, use_expected_time=False):
        enabled_timed_transitions, enabled_imm_transitions = self.mr_gspn.get_enabled_transitions()

        elapsed_time = 0
        fired_transitions = []
        while(enabled_timed_transitions and not enabled_imm_transitions):
            action_elapsed_time, tr_fired = self.fire_timed_transitions(enabled_timed_transitions, use_expected_time)
            elapsed_time += action_elapsed_time
            fired_transitions += tr_fired
            enabled_timed_transitions, enabled_imm_transitions = self.mr_gspn.get_enabled_transitions()

        return elapsed_time, fired_transitions

    def get_action_info_attributes(self, action):
        action_name = action[0]
        action_number = int(action_name.split('_')[-1])
        action_time = action[1]

        return action_name, action_number, action_time

    def get_rates_ground_truth(self):
        timed_transitions = self.mr_gspn.get_timed_transitions()
        true_rates = {}
        for name, rate in timed_transitions.items():
            action = int(name.split('_')[-1])
            true_rates[action] = rate

        return true_rates

    def from_index_to_action(self, action_index):
        return '_'+str(action_index)

    def from_action_to_index(self, action_name):
        return int(action_name.split('_')[-1])

    def get_disabled_actions(self):
        enabled_exp_transitions, enabled_imm_transitions = self.mr_gspn.get_enabled_transitions()

        disabled_actions_indexes = list(range(self.action_space.n))
        disabled_actions_names = list(map(self.from_index_to_action, disabled_actions_indexes))

        for enabled_action in enabled_imm_transitions.keys():
            disabled_actions_names.remove(enabled_action)
            action_index = int(enabled_action.split('_')[-1])
            disabled_actions_indexes.remove(action_index)

        return disabled_actions_names, disabled_actions_indexes

    def get_enabled_actions(self):
        enabled_exp_transitions, enabled_imm_transitions = self.mr_gspn.get_enabled_transitions()
        enabled_actions_names = list(enabled_imm_transitions.keys())
        enabled_actions_indexes = list(map(self.from_action_to_index, enabled_actions_names))

        return enabled_actions_names, enabled_actions_indexes

    def get_action_time(self, action):
        transition = 'Finished_'+str(action)
        transition_rate = self.mr_gspn.get_transition_rate(transition)
        action_expected_time = 1.0/transition_rate
        return action_expected_time

    def get_action_time_noiseless(self, action):
        if self.n_locations == None:
            raise Exception('Please specify the number of locations when instantiating the environment.')
        else:
            if action in self.optimal_actions:
                # 1.0/0.5
                return 2.0
            else:
                # 1.0/1.0
                return 1.0

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]