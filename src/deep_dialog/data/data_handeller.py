import json
import random


with open('src/deep_dialog/data/dia_act_nl_pairs.v6.json') as f:
  data = json.load(f)

# file = open("test_data.json")  
dacts = data['dia_acts']

print(type(data),type(dacts.items()))

test_data = {}
test_data_wrapper = {}
item_list =[]

for i in dacts.items():
    if len(i[1])>4:
        sub_item_list =[] 
        for j in range (len(i[1])//5):
            sub_item_list.append(random.choice(i[1]))
        # print(len(new_list),new_list)
        test_data[i[0]] = sub_item_list
        for item in sub_item_list:
            item_list.append(item)

    else:
        temp_list = []
        temp_list.append(random.choice(i[1]))
        test_data[i[0]] =  temp_list
        # print(random.choice(i[1]))

test_data_wrapper['dia_acts'] = test_data

with open('src/deep_dialog/data/test_data.json', 'w') as outfile:
    json.dump(test_data_wrapper, outfile)

training_data = {}
training_data_wrapper = {}

for i in dacts.items():
    if len(i[1])>4:
        sub_item_list =[] 
        for j in i[1]:
            if not j in item_list:
                #print(j)
                sub_item_list.append(j)
        training_data[i[0]] = sub_item_list

    else:
        training_data[i[0]] = i[1]
        # print(i[1])

training_data_wrapper['dia_acts'] = training_data

with open('src/deep_dialog/data/training_data.json', 'w') as outfile:
    json.dump(training_data_wrapper, outfile)
