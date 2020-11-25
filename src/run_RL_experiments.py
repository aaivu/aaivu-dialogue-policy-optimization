from __future__ import division
import pickle
import random
import os
import glob
import argparse
import json

import copy
import numpy as np
from random import randint
import string

with open("config_rl.json") as json_config_file:
    config = json.load(json_config_file)

alpha = config["alpha"]
beta = config["beta"]
rlalgorithms = config["Rl algorithms"]
algorithm = config["algorithm"]

for i in rlalgorithms:
    if i == algorithm:
        agent_id = rlalgorithms[i]

pickle_in = open(
    "deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p", "rb")
initial_dict = pickle.load(pickle_in)
print(len(initial_dict))


test_dict = []
count = 0
while count <= (len(initial_dict)/5):
    elem = random.choice(initial_dict)
    #print (elem)
    if elem not in test_dict:
        test_dict.append(elem)
        count += 1
    else:
        print("############")

print(len(test_dict))

training_dict = [item for item in initial_dict if item not in test_dict]
print('size of training dict: ', len(training_dict))


def get_num_of_training_samples():
    return len(training_dict)


print("total length")
print(len(training_dict)+len(test_dict))


with open('deep_dialog/data/test_user_goals.pickle', 'wb') as test:
    pickle.dump(test_dict, test, protocol=pickle.HIGHEST_PROTOCOL)

with open('deep_dialog/data/training_user_goals.pickle', 'wb') as train:
    pickle.dump(training_dict, train, protocol=pickle.HIGHEST_PROTOCOL)


# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()

movie_kb_path = 'deep_dialog/data/movie_kb.1k.p'
movie_kb = pickle.load(open(movie_kb_path, 'rb'))

training_sample_count = 35
turn_count = 0

while training_sample_count <= 40:
    training_sample = training_dict[:training_sample_count]

    with open('deep_dialog/checkpoints/train.txt', 'a') as results:
        results.write("#######################################result for training sample {} turn count {} agent {} #######################################\n".format(
            training_sample_count, turn_count, algorithm))

    with open('deep_dialog/checkpoints/test.txt', 'a') as results:
        results.write("#######################################result for training sample {} turn count {} agent {} #######################################\n".format(
            training_sample_count, turn_count, algorithm))

    print(len(training_sample))

    with open('deep_dialog/data/training_sample.pickle', 'wb') as train:
        pickle.dump(training_sample, train,
                    protocol=pickle.HIGHEST_PROTOCOL)
        print("training sample added :", len(training_sample))

    os.system("python run_RL.py --agt {} --usr 1 --max_turn 40 --movie_kb_path deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --experience_replay_pool_size 1000 --episodes 200 --simulation_epoch_size 20 --write_model_dir deep_dialog/checkpoints/rl_agent/ --run_mode 3 --act_level 0 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 16 --goal_file_path deep_dialog/data/training_sample.pickle --warm_start 1 --warm_start_epochs 120 --alpha {} --beta {} --test_path deep_dialog/data/test_user_goals.pickle".format(9, alpha, beta))

    training_sample_count += 5
