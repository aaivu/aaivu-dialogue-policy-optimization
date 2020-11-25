from __future__ import division
import argparse
import json
import random
import copy
import cPickle as pickle
import copy


import numpy as np
from random import randint
import string

from deep_dialog import dialog_config


def softmax(x, beta=1):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(np.asarray(x) * beta) / np.sum(np.exp(np.asarray(x) * beta), axis=0)


def probs_normalize(x, beta=1):
    return np.power(x, beta) / np.sum(np.power(x, beta), axis=0)


class SelfPlay():

    def __init__(self, sample_goals, movie_kb, epsilon_1, epsilon_2, epsilon_3, constant_reward=True):
        self.slots = ['actor', 'actress', 'city', 'closing', 'critic_rating', 'date', 'description', 'distanceconstraints', 'genre',
                      'greeting', 'implicit_value', 'movie_series', 'moviename', 'mpaa_rating', 'numberofpeople', 'numberofkids', 'taskcomplete',
                      'other', 'price', 'seating', 'starttime', 'state', 'theater', 'theater_chain', 'video_format', 'zip', 'result', 'ticket', 'mc_list']

        self.dict_slots = {}
        self.dict_values = {}
        self.slot_value = {}
        self.slot_value_list = {}
        self.dict_slots_prob = {}
        self.inform_slots_prob = {}
        self.request_slots_prob = {}

        self.count = 0
        self.slot_count = 0
        self.req_count = 0
        self.inform_count = 0
        self.count_error = 0
        self.count_e_n_c = 0
        self.max_reward = 100
        self.constant_reward = constant_reward
        self.mean_reward = 1

        self.sum_prob = 0
        self.list_slots = []
        self.list_probs = []
        self.slot_domain_list = {}

        self.sample = sample_goals
        self.movie_kb = movie_kb
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        self.epsilon_1_start = epsilon_1
        self.epsilon_2_start = epsilon_2
        self.slot_epsilon_start = epsilon_3
        self.slot_epsilon = self.slot_epsilon_start

    def reset(self, sample_goals, movie_kb):
        self.dict_slots = {}
        self.dict_values = {}
        self.slot_value = {}
        self.slot_value_list = {}
        self.dict_slots_prob = {}
        self.inform_slots_prob = {}
        self.request_slots_prob = {}
        self.slot_domain_list = {}

        self.count = 0
        self.slot_count = 0
        self.req_count = 0
        self.inform_count = 0
        self.count_error = 0
        self.count_e_n_c = 0
        self.max_reward = 100

        self.sum_prob = 0
        self.list_slots = []
        self.list_probs = []

        self.sample = sample_goals
        self.movie_kb = movie_kb
        self.epsilon_1 = 0.001
        self.epsilon_2 = 0.0001
        self.epsilon_1_start = 0.001
        self.epsilon_2_start = 0.0001
        self.slot_epsilon_start = 0.0001
        self.slot_epsilon = self.slot_epsilon_start

        self.calculate_probs()

    def sample_random_goal(self):
        sample_goal = random.choice(self.sample)
        print("random goal")
        return sample_goal

    def set_sample(self, goal_set):
        # print(self.sample)
        self.sample = goal_set
        # print(self.sample)

    def calculate_probs(self):

        print("####### recalculating probability #######")

        for i in self.slots:
            self.dict_slots[i] = False
            self.slot_domain_list[i] = []

        for i in self.movie_kb:
            for j in self.movie_kb[i]:
                if j in self.dict_slots:
                    self.dict_slots[j] = True
                    self.slot_domain_list[j].append(i)

        for i in self.slots:
            self.dict_values[i] = []

        for i in self.movie_kb:
            for j in self.movie_kb[i]:
                if j in self.dict_values:
                    if self.dict_slots[j] == True:
                        if self.movie_kb[i][j] not in self.dict_values[j]:
                            self.dict_values[j].append(self.movie_kb[i][j])

        for i in self.slots:
            self.slot_value[i] = False
        for i in self.sample:
            inform_slots = i['inform_slots']
            if len(inform_slots) != 0:
                for j in inform_slots:
                    self.slot_value[j] = True
                    self.slot_value_list[j] = []

        for i in self.slots:
            self.dict_slots_prob[i] = 0
            self.inform_slots_prob[i] = 0
            self.request_slots_prob[i] = 0

        for i in self.sample:
            inform_slots = i['inform_slots']
            request_slots = i['request_slots']

            if len(request_slots) != 0:
                for j in request_slots:
                    self.request_slots_prob[j] += 1
                    self.dict_slots_prob[j] += 1
                    self.req_count += 1
                    self.slot_count += 1

            if len(inform_slots) != 0:
                for j in inform_slots:
                    self.inform_slots_prob[j] += 1
                    self.dict_slots_prob[j] += 1
                    self.slot_count += 1
                    self.inform_count += 1
                    self.slot_value_list[j].append(inform_slots[j])

            self.count += 1

        self.num_slots_mean = self.slot_count / self.count

        for i in self.slots:
            if self.dict_slots_prob[i] != 0:
                self.inform_slots_prob[i] = self.inform_slots_prob[i] / \
                    self.dict_slots_prob[i]  # epsilon 0 ...........................
                self.request_slots_prob[i] = self.request_slots_prob[i] / \
                    self.dict_slots_prob[i]

            if self.inform_slots_prob[i] == 1.0:
                if self.dict_slots[i] == True:
                    self.inform_slots_prob[i] = 1 - self.epsilon_1
                    # epsilon 1 ...................
                    self.request_slots_prob[i] = self.epsilon_1
                else:
                    self.inform_slots_prob[i] = 1 - self.epsilon_2
                    # epsilon 2 .......................
                    self.request_slots_prob[i] = self.epsilon_2

            if self.request_slots_prob[i] == 1.0:
                # epsilon 3 .....................
                self.inform_slots_prob[i] = 0.05
                self.request_slots_prob[i] = 0.95

            if self.request_slots_prob[i] == 0 and self.inform_slots_prob[i] == 0:
                if self.dict_slots[i] == True:
                    self.request_slots_prob[i] = self.req_count / \
                        (self.req_count +
                         self.inform_count)  # epsilon 4 .........................
                    self.inform_slots_prob[i] = self.inform_count / \
                        (self.req_count +
                         self.inform_count)  # epsilon 4 .........................
                else:
                    # epsilon 5 ...........................
                    self.request_slots_prob[i] = 0.05
                    self.inform_slots_prob[i] = 0.95

            if self.dict_slots_prob[i] == 0:
                # epsilon 6 ..................
                self.dict_slots_prob[i] = self.slot_epsilon

        for i in self.slots:
            self.sum_prob += self.dict_slots_prob[i]

        for i in self.slots:
            self.dict_slots_prob[i] = self.dict_slots_prob[i] / self.sum_prob

        for i in self.dict_slots_prob:
            self.list_slots.append(i)
            self.list_probs.append(self.dict_slots_prob[i])

        print(self.dict_slots_prob)

        # return self.dict_slots_prob

    def sample_goal_prob(self):

        error_prob = 0.001  # epsilon 7 ........................

        num_slots = np.random.normal(
            self.num_slots_mean, self.num_slots_mean / 6, 1)

        if num_slots[0] < 3:
            num_slots = 3
        elif num_slots[0] > 10:
            num_slots = 10
        else:
            num_slots = int(round(num_slots[0]))

        user_goal = {}
        number = randint(0, len(self.movie_kb) - 1)
        user_goal['diaact'] = 'request'
        user_goal['request_slots'] = {}
        user_goal['inform_slots'] = {}

        list_slots_sample = np.random.choice(
            self.list_slots, num_slots, replace=False, p=self.list_probs)

        while True:
            list_slots_sample = np.random.choice(
                self.list_slots, num_slots, replace=False, p=self.list_probs)
            list_slots_informable = []
            for i in list_slots_sample:
                if self.dict_slots[i]:
                    list_slots_informable.append(i)
            # print(list_slots_informable)
            all_list = []
            for i in list_slots_informable:
                # print(self.slot_domain_list[i])
                all_list.extend(self.slot_domain_list[i])
            # print(all_list)
            if len(all_list) > 0:
                break

        #number = max(set(all_list), key=all_list.count)
        # print(number)

        for i in list_slots_sample:
            if random.random() < self.inform_slots_prob[i]:
                if self.dict_slots[i] == False:
                    if self.slot_value[i] == True:
                        user_goal['inform_slots'][i] = np.random.choice(
                            self.slot_value_list[i], 1)[0]
                    else:
                        if random.random() > 0.9:
                            user_goal['inform_slots'][i] = str(''.join(random.choice(
                                string.ascii_uppercase + string.digits) for _ in range(5)))

                else:
                    if random.random() > error_prob:
                        if i in self.movie_kb[number]:
                            user_goal['inform_slots'][i] = self.movie_kb[number][i]
                    else:
                        user_goal['inform_slots'][i] = np.random.choice(self.dict_values[i], 1)[
                            0]
            else:
                user_goal['request_slots'][i] = 'UNK'

        return user_goal

    def prob_update_epsilon_increment(self, theta_1=2, theta_2=2, theta_3=2, additive=True):
        self.dict_slots = {}
        self.dict_values = {}
        self.slot_value = {}
        self.slot_value_list = {}
        self.dict_slots_prob = {}
        self.inform_slots_prob = {}
        self.request_slots_prob = {}

        self.count = 0
        self.slot_count = 0
        self.req_count = 0
        self.inform_count = 0
        self.count_error = 0
        self.count_e_n_c = 0

        self.sum_prob = 0
        self.list_slots = []
        self.list_probs = []
        if additive:
            self.epsilon_1 = self.epsilon_1_start * theta_1
            self.epsilon_2 = self.epsilon_2_start * theta_2
            print(self.slot_epsilon)
            # print(epsilons_factor)
            self.slot_epsilon = self.slot_epsilon_start * theta_3
        else:
            self.epsilon_1 = self.epsilon_1 * theta_1
            self.epsilon_2 = self.epsilon_2 * theta_2
            print(self.slot_epsilon)
            # print(epsilons_factor)
            self.slot_epsilon = self.slot_epsilon * theta_3
        self.calculate_probs()

    def prioratize_failed_dialogues(self, reward_list, alpha=0.2, beta=1, normalize='naive'):
        dict_slots_rewards = {}
        for i in self.slots:
            dict_slots_rewards[i] = {}
            dict_slots_rewards[i]['reward'] = 0
            dict_slots_rewards[i]['count'] = 0
        for i in reward_list:
            agenda = i['agenda']
            reward = i['reward']
            inform_slots = agenda['inform_slots']
            request_slots = agenda['request_slots']
            for j in inform_slots:
                dict_slots_rewards[j]['reward'] += reward
                dict_slots_rewards[j]['count'] += 1
            for j in request_slots:
                dict_slots_rewards[j]['reward'] += reward
                dict_slots_rewards[j]['count'] += 1
        for i in self.slots:
            print(i)
            print("dict_slot_reawrd before", dict_slots_rewards[i]['reward'])
            print("count", dict_slots_rewards[i]['count'])
            if dict_slots_rewards[i]['count'] != 0:
                dict_slots_rewards[i]['reward'] = dict_slots_rewards[i]['reward'] * \
                    (-1) + (self.max_reward * dict_slots_rewards[i]['count'])
                print("reward after", dict_slots_rewards[i]['reward'])
                dict_slots_rewards[i]['mean_reward'] = dict_slots_rewards[i]['reward'] / \
                    dict_slots_rewards[i]['count']
                print("mean reward", dict_slots_rewards[i]['mean_reward'])
            else:
                dict_slots_rewards[i]['mean_reward'] = self.mean_reward
            print("probs before", self.dict_slots_prob[i])
            dict_slots_rewards[i]['probs'] = self.dict_slots_prob[i] * \
                ((dict_slots_rewards[i]['mean_reward'] /
                  self.max_reward) ** alpha)
            print("probs after", dict_slots_rewards[i]['probs'])

        self.list_slots = []
        self.list_probs = []
        for i in self.slots:
            self.list_slots.append(i)
            self.list_probs.append(dict_slots_rewards[i]['probs'])
        if normalize == 'softmax':
            self.list_probs = softmax(self.list_probs, beta=beta)
        else:
            self.list_probs = probs_normalize(self.list_probs, beta=beta)

        for i in range(len(self.slots)):
            self.dict_slots_prob[self.slots[i]] = self.list_probs[i]
            print(self.slots[i])
            print(self.dict_slots_prob[self.slots[i]])
