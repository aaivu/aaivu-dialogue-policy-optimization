
import random
import copy
import json
import cPickle as pickle
from typing import Dict, List, Tuple
import random
import torch
import math
import numpy as np
from collections import deque, namedtuple
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from deep_dialog import dialog_config

from agent import Agent

from deep_dialog.qlearning import DQN
from replay_buffer import ReplayBuffer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.epsilon = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.reset_noise()
        self.reset_sigma(self)
        
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))


    def reset_sigma(self):
        self.sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

        
    def forward(self, input):
        y = input.shape
        output = input.matmul(self.weight.t())
        noise = self.sigma.matmul(self.epsilon.t())
        output = output + noise
        if self.bias is not None:
            output += self.bias
        ret = output
        return ret


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_unit=64, fc2_unit=64):
        """Initialization."""
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.noisy_layer = NoisyLinear(fc2_unit, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        #reset_noise()
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = self.noisy_layer.forward(x)
        return out

    def reset_noise(self):
        """Reset all noisy layers."""
        self.noisy_layer.reset_noise()


class NoisyNet(Agent):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None, seed=1234):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.feasible_actions = dialog_config.feasible_actions
        self.num_actions = len(self.feasible_actions)

        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']
        self.experience_replay_pool = []  # experience replay pool <s_t, a_t, r_t, s_t+1>

        self.memory = ReplayBuffer(self.num_actions, 10000, 16, seed)
        self.t_step = 0
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.experience_replay_pool_size = params.get(
            'experience_replay_pool_size', 1000)
        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)

        self.max_turn = params['max_turn'] + 4
        self.state_dimension = 2 * self.act_cardinality + \
            7 * self.slot_cardinality + 3 + self.max_turn

        self.qnetwork_local = QNetwork(
            self.state_dimension, self.num_actions, seed).to(self.device)
        self.qnetwork_target = QNetwork(
            self.state_dimension, self.num_actions, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=5e-4)

        self.cur_bellman_err = 0

        # Prediction Mode: load trained DQN model
        if params['trained_model_path'] != None:
            self.dqn.model = copy.deepcopy(
                self.load_trained_DQN(params['trained_model_path']))
            self.clone_dqn = copy.deepcopy(self.dqn)
            self.predict_mode = True
            self.warm_start = 2

    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """

        self.current_slot_id = 0
        self.phase = 0
        self.request_set = ['moviename', 'starttime',
                            'city', 'date', 'theater', 'numberofpeople']

    def state_to_action(self, state):
        """ DQN: Input state, output action """

        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation)
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}

    def prepare_state_representation(self, state):
        """ Create the representation for each state """

        user_action = state['user_action']
        current_slots = state['current_slots']
        kb_results_dict = state['kb_results_dict']
        agent_last = state['agent_action']

        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep = np.zeros((1, self.act_cardinality))
        user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0

        ########################################################################
        #     Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent request slots
        ########################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

        turn_rep = np.zeros((1, 1)) + state['turn'] / 10.

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + \
            kb_results_dict['matching_all_constraints'] / 100.
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_count_rep[0, self.slot_set[slot]
                             ] = kb_results_dict[slot] / 100.

        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + \
            np.sum(kb_results_dict['matching_all_constraints'] > 0.)
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_binary_rep[0, self.slot_set[slot]] = np.sum(
                    kb_results_dict[slot] > 0.)

        self.final_representation = np.hstack([user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep,
                                               agent_inform_slots_rep, agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
        return self.final_representation

    def run_policy(self, state):
            #self.qnetwork_local.reset_noise()
            if self.warm_start == 1:
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                return self.rule_policy()
            else:
                # return self.dqn.predict(representation, {}, predict_model=True)

                state = torch.from_numpy(
                    state).float().unsqueeze(0).to(self.device)
                #print('state', state)
                self.qnetwork_local.eval()
                with torch.no_grad():
                    action_values = self.qnetwork_local(state)
                    self.qnetwork_local.train()
                #print('action', np.argmax(action_values.cpu().data.numpy()))
                return np.argmax(action_values.cpu().data.numpy())

    def rule_policy(self):
        """ Rule Policy """

        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {
                'taskcomplete': "PLACEHOLDER"}, 'request_slots': {}}
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks",
                                 'inform_slots': {}, 'request_slots': {}}

        return self.action_index(act_slot_response)

    def action_index(self, act_slot_response):
        """ Return the index of action """

        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        print act_slot_response
        raise Exception("action index not found")
        return None

    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over):
        """ Register feedback from the environment, to be stored as future training data """

        state = self.prepare_state_representation(s_t)
        action = self.action
        reward = reward
        next_state = self.prepare_state_representation(s_tplus1)
        done = episode_over
        # training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over)

        if self.predict_mode == False:  # Training Mode
            if self.warm_start == 1:
                self.memory.add(state, action, reward, next_state, done)
        else:  # Prediction Mode
            self.memory.add(state, action, reward, next_state, done)

    def train(self, batch_size=1, num_batches=100):
        """ Train DQN with experience replay """

        for iter_batch in range(num_batches):
            self.cur_bellman_err = 0
            c = 0
            for iter in range(len(self.memory)/(batch_size)):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

            # print ("cur bellman err %.4f, experience replay pool %s" % (float(self.cur_bellman_err)/len(self.experience_replay_pool), len(self.experience_replay_pool)))

    def learn(self, experiences, gamma):
        states, actions, reward, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(
            next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = reward + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1e-2)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)
    ################################################################################
    #    Debug Functions
    ################################################################################

    def save_experience_replay_to_file(self, path):
        """ Save the experience replay pool to a file """

        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
            print 'saved model in %s' % (path, )
        except Exception, e:
            print 'Error: Writing model fails: %s' % (path, )
            print e

    def load_experience_replay_from_file(self, path):
        """ Load the experience replay pool from a file"""

        self.experience_replay_pool = pickle.load(open(path, 'rb'))

    def load_trained_DQN(self, path):
        """ Load the trained DQN from a file """

        trained_file = pickle.load(open(path, 'rb'))
        model = trained_file['model']

        print "trained DQN Parameters:", json.dumps(
            trained_file['params'], indent=2)
        return model
