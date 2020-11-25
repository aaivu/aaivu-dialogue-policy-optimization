
import random
import copy
import json
import cPickle as pickle
from typing import Dict, List, Tuple
import random
import torch
import numpy as np
from collections import deque, namedtuple
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from deep_dialog import dialog_config

from agent import Agent

from deep_dialog.qlearning import DQN
from replay_buffer import ReplayBuffer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


def compute_returns(next_values, rewards, masks, gamma=0.99):
    #R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * next_values[step] * masks[step]
        returns.insert(0, R)
    return returns


class A2CAgent(Agent):
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
        # self.experience_replay_pool = []  # experience replay pool <s_t, a_t, r_t, s_t+1>

        #self.memory = ReplayBuffer(self.num_actions, 10000, 1, seed)
        self.t_step = 0
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)

        self.max_turn = params['max_turn'] + 4
        self.state_dimension = 2 * self.act_cardinality + \
            7 * self.slot_cardinality + 3 + self.max_turn

        self.actor = Actor(self.state_dimension, self.num_actions).to(device)
        self.critic = Critic(self.state_dimension, self.num_actions).to(device)
        self.t_step = 0

        self.memory = []
        self.episode_memory_list = []

        self.optimizer_actor = optim.Adam(self.actor.parameters())
        self.optimizer_critic = optim.Adam(self.critic.parameters())

        self.cur_bellman_err = 0
        self.entropy_weight = 1e-2
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
        """ epsilon-greedy policy """

        # return self.dqn.predict(representation, {}, predict_model=True)
        if self.warm_start == 1:
            # if len(self.experience_replay_pool) > self.experience_replay_pool_size:
            #    self.warm_start = 2
            return self.rule_policy()
        else:
            state = torch.from_numpy(
                state).float().unsqueeze(0).to(self.device)
            self.actor.eval()
            with torch.no_grad():
                dist = self.actor(state)
                #print('action', action)
            self.actor.train()
            self.critic.train()

            return dist.sample()

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
        dict_step = {'state': state, 'action': action,
                     'reward': reward, 'next_state': next_state, 'done': done}
        self.episode_memory_list.append(dict_step)
        #training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over)

    def add_to_memory(self):
        self.memory.append(self.episode_memory_list)

    def empty_memory(self):
        self.episode_memory_list = []

    def train(self, batch_size=1, num_batches=100):
        """ Train DQN with experience replay """

        for iter_batch in range(num_batches):
            self.cur_bellman_err = 0
            c = 0
            for iter in range(len(self.memory)/(batch_size)):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

            #print ("cur bellman err %.4f, experience replay pool %s" % (float(self.cur_bellman_err)/len(self.experience_replay_pool), len(self.experience_replay_pool)))

    def learn(self):
        print(len(self.memory))
        print('learn ....  made changes')
        for i in range(len(self.memory)):
            gamma = self.gamma
            memory_list = random.choice(self.memory)
            log_probs = []
            values = []
            rewards = []
            masks = []
            entropy = 0
            next_values = []
            for samples in memory_list:

                state = samples['state']
                action = torch.from_numpy(np.array(samples['action']))
                # print(action)
                reward = samples['reward']
                next_state = samples['next_state']
                done = samples['done']
                state = torch.from_numpy(state).float().to(device)
                #print(state, action, reward, next_state, done)

                dist, value = self.actor(state), self.critic(state)
                log_prob = dist.log_prob(action).unsqueeze(0)
                entropy += dist.entropy().mean()

                next_state = torch.from_numpy(next_state).float().to(device)
                next_value = self.critic(next_state)

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.tensor(
                    [reward], dtype=torch.float, device=device))
                masks.append(torch.tensor(
                    [1-done], dtype=torch.float, device=device))
                next_values.append(torch.tensor(
                    [next_value], dtype=torch.float, device=device))

                #state = next_state

                if done:
                    # print()
                    #print('Iteration: {}, Score: {}'.format(iter, i))
                    #episode_durations.append(i + 1)
                    # plot_durations()
                    break

            #next_state = torch.FloatTensor(next_state).to(device)
            #next_value = self.critic(next_state)
            returns = compute_returns(next_values, rewards, masks, gamma=0.9)
            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)

            advantage = returns - values

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
            # print(critic_loss)
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

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
