import pickle
import random
import os
import glob

pickle_in = open("src/deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p","rb")
initial_dict = pickle.load(pickle_in)
print(len(initial_dict))
        
'''count of duplicationg elements'''
# wtf =[]     
# wtfcount = 0   
# for i in range (len(initial_dict)):
#     elem = initial_dict[i]
#     if elem not in wtf:
#         wtf.append(elem)
#     else:
#         wtfcount+=1 

# print(wtfcount)

test_dict = []
count =0
while count<=(len(initial_dict)/5):
    elem = random.choice(initial_dict)
    #print (elem)
    if elem not in test_dict:
        test_dict.append(elem)
        count +=1
    else:
        print("############")    

print(len(test_dict))

training_dict = [item for item in initial_dict if item not in test_dict]
print('size of training dict: ',len(training_dict))

def get_num_of_training_samples():
    return len(training_dict)

print(len(training_dict)+len(test_dict))

# for goal in initial_dict:
#     for test_goal in test_dict:
#         if not goal in test_goal: 
#             training_dict.append(goal)      


with open('src/deep_dialog/data/test_user_goals.pickle', 'wb') as test:
    pickle.dump(test_dict, test, protocol=pickle.HIGHEST_PROTOCOL)

with open('src/deep_dialog/data/training_user_goals.pickle', 'wb') as train:
    pickle.dump(training_dict, train, protocol=pickle.HIGHEST_PROTOCOL)    


# bashCommand = "python src/run.py --agt 9 --usr 1 --max_turn 40 --movie_kb_path src/deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --experience_replay_pool_size 1000 --episodes 500 --simulation_epoch_size 100 --write_model_dir src/deep_dialog/checkpoints/rl_agent/ --run_mode 3 --act_level 0 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 16 --goal_file_path src/deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --warm_start 1 --warm_start_epochs 120"
# import subprocess
# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()



training_sample_count = 25
turn_count = 0

while training_sample_count <= len(training_dict):

    while True:
        training_sample = training_dict[:training_sample_count]

        with open('src/deep_dialog/checkpoints/results.txt', 'a') as results:
            results.write("#######################################result for training sample {} turn count {}#######################################\n".format(training_sample_count,turn_count))
        
        with open('src/deep_dialog/data/training_sample.pickle', 'wb') as train:
            pickle.dump(training_sample, train, protocol=pickle.HIGHEST_PROTOCOL)
            print("training sample added :", len(training_sample))    

        os.system("python src/run.py --agt 9 --usr 1 --max_turn 40 --movie_kb_path src/deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --experience_replay_pool_size 1000 --episodes 200 --simulation_epoch_size 100 --write_model_dir src/deep_dialog/checkpoints/rl_agent/ --run_mode 3 --act_level 0 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 16 --goal_file_path src/deep_dialog/data/training_sample.pickle --warm_start 1 --warm_start_epochs 120")
        os.system("python src/draw_learning_curve.py --result_file src/deep_dialog/checkpoints/rl_agent/agt_9_performance_records.json --sample-rate {} --turn-count {}".format(training_sample_count,turn_count))
        
        list_of_files = glob.glob('src/deep_dialog/checkpoints/rl_agent/models/*.p') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        print ("latest  :",latest_file)
        os.system("python src/run.py --agt 9 --usr 1 --max_turn 40 --movie_kb_path src/deep_dialog/data/movie_kb.v2.p --dqn_hidden_size 80 --experience_replay_pool_size 1000 --episodes 100 --simulation_epoch_size 100 --write_model_dir src/deep_dialog/checkpoints/rl_agent/ --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 16 --goal_file_path src/deep_dialog/data/test_user_goals.pickle --trained_model_path {} --run_mode 3".format(latest_file)) 


        print(turn_count)
        
        if turn_count == 2:
            turn_count = 0
            training_sample_count += 5
            break 

        turn_count +=1  