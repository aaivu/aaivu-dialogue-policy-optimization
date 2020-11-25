## Project / Research Source code

The important files and directories of the repository is shown below

    ├── deep_dialog
        ├── agents : contain RL agents
        ├── data : dialogue dataset with knowledge base and goals
        ├── dialog_system : contain dialogue manager
        ├── models : nlu and nlg models
        ├── nlg : Natural Language Generation Module
        ├── nlu : Natural Language Undersatanding Module 
        ├── qlearning : qlearning implementaiton in numpy
        ├── self_play : Self Play algorithm with reward based sampling
        ├── usersims : Agenda based user simualator
    ├── run_RL_experiment.py : To run the RL experiment on a given RL algorithm
    ├── run_RL.py : To run RL agent with the self-play and reward based sampling 
    ├── run.py : run agent without self-play and reward based sampling
    ├── user_goal_handler.py : run self-play and reward based sampling experiment varying number of samples
    ├── config.json : configuration file for user_goal_handler
    └── config_rl.json : configuration file for run_RL_experiment
