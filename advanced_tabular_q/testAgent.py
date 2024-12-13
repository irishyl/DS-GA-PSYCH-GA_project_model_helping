#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 23:57:27 2024

@author: lvxinyuan
"""
import random
import farmgame
from farmgame import *
import os
from scipy.spatial.distance import euclidean
from datetime import datetime
import csv
from ast import literal_eval
from trainAgent import *
import time
import logging

# Configure logging
logging.basicConfig(filename="debug_log.txt", 
                    level=logging.DEBUG, 
                    format="%(asctime)s - %(levelname)s - %(message)s")


def load_q_table_from_csv(filename):
    """
    Loads a Q-table from a CSV file and reconstructs it as a dictionary.

    Parameters:
    - filename: Path to the CSV file containing the Q-table.

    Returns:
    - A dictionary representing the Q-table.
    """
    Q = {}
    with open(filename, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                def parse_value(value, value_type=int, default=None):
                    return default if value == 'N/A' or value == '' else value_type(value)
                
                # Reconstruct the state tuple
                state = (
                    row["objectLayer"],                           # string
                    row["redfirst"] == "True",                    # boolean
                    row["resourceCond"],                         # string
                    row["costCond"],                             # string
                    row["visibilityCond"],                       # string
                    (parse_value(row["redRemainCapacity"]), parse_value(row["purRemainCapacity"])),  # integers
                    (row["red_energy"] or None, row["pur_energy"] or None),       # strings ('full' left as-is for clarity)
                    (parse_value(row["red_score"], int, 0), parse_value(row["pur_score"], int, 0)),  # integers, default 0
                    (row["red_helped"] == "True", row["pur_helped"] == "True"),  # booleans
                )
                action = row["Action"]  # string
                q_value = float(row["Q-Value"])  # float

                # Add state-action pair to Q-table
                Q[(state, action)] = q_value
            except Exception as e:
                print(f"Error processing row: {row}")
                print(f"Error details: {e}")
    return Q


def test_agents(farm: Farm, Q_red, Q_purple, session):
    """
    Run the game for one session and record relevant data.
    Returns a dictionary with the game result data.
    """
    
    state = get_state(farm)
    game_num = int(state[0][-2:])
    logging.info("Starting game session %d, game number %d", session, game_num)
    
    # Initialize tracking variables
    red_help_first = False
    purple_help_first = False
    red_also_help = False
    purple_also_help = False
    
    # Timeout setup
    start_time = time.time()
    timeout = 60  # seconds
    
    
    while not farm.is_done():
        # Timeout setup
        if time.time() - start_time > timeout:
            logging.warning("Timeout reached in game session %d, game number %d", session, game_num)
            break
        
        try:
            current_player = farm.whose_turn()["name"]
            Q = Q_red if current_player == "red" else Q_purple
            actions = get_legal_actions(farm)
            logging.debug("Legal actions for %s: %s", current_player, actions)
            
            # Choose action with max Q-value
            max_q_value = float('-inf')
            action = None
            for a in actions:
                action_id = a.id if a.type == ActionType.veggie else a.name
                q_value = Q.get((state, action_id), 0)
                if q_value > max_q_value:
                    max_q_value = q_value
                    action = a
            logging.debug("Chosen action: %s, Max Q-value: %.3f", action, max_q_value)
            
            # Take the chosen action and update the state
            farm.take_action(action)
            logging.debug("Action taken: %s", action)
            state = get_state(farm)
            logging.debug("Updated state: %s", state)
        
            # Update tracking variables
            red_help_first = farm.opponent_has_helped("red") and not farm.opponent_has_helped("purple")
            purple_help_first = farm.opponent_has_helped("purple") and not farm.opponent_has_helped("red")
            
            purple_also_help = red_help_first and farm.opponent_has_helped("purple")
            red_also_help = purple_help_first and farm.opponent_has_helped("red")
       
        except Exception as e:
            logging.error("Error occurred: %s", e)
            break  # Exit the loop on error
    
    logging.info("Game session %d, game number %d completed", session, game_num)
    
    # Collect game data
    game_data = {
        "session": session+1,
        "gameNum": game_num+1,
        "costCond": farm.costCond,
        "visibilityCond": farm.visibilityCond,
        "resourceCond": "even" if farm.resourceCond == "even" else "uneven",
        "purpleEnergy": farm.purpleplayer["energy"],
        "purpleBackpackSize": farm.purpleplayer["capacity"],
        "redEnergy": farm.redplayer["energy"],
        "redBackpackSize": farm.redplayer["capacity"],
        "purpHelpFirst": purple_help_first,
        "redHelpFirst": red_help_first,
        "purpAlsoHelp": purple_also_help,
        "redAlsoHelp": red_also_help,
        "nRedVeg": farm.redplayer['score'],
        "nPurpVeg": farm.purpleplayer['score'],
    }
    
    
    
    return game_data


def write_to_csv(filename, data):
    """
    Writes game data to a CSV file and organizes it in a time-stamped folder structure.
    
    Parameters:
    - filename: Name of the CSV file (without folder path).
    - data: List of dictionaries containing game data.
    """
    if not data:
        print("No data to write!")
        return
    
    # Create folder structure
    current_folder = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(current_folder, "output")
    time_folder = os.path.join(output_folder, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(time_folder, exist_ok=True)
    log_file_path = os.path.join(time_folder, filename)
    
    # Get the header from the keys of the first dictionary
    headers = data[0].keys()
    
    # Write to CSV
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Data successfully written to {log_file_path}")



def main():
    sessions = 1000
    layers = ["Items00","Items01","Items02","Items03","Items04",
              "Items05","Items06","Items07","Items08","Items09",
              "Items10","Items11"]
    # layers = ["Items01"] 
    resource_conditions = ["even", "unevenRed", "unevenPurple"]  # 3 values
    cost_conditions = ["low", "high"]                           # 2 values
    visibility_conditions = ["full", "self"]                   # 2 values
    red_first_choices = [True, False]                          # 2 values
    
    all_game_data = []  # To store data from all games
    Q_purple = load_q_table_from_csv("/Users/lvxinyuan/me/1-Projects/NYU/1-Courses/24_Fall_CCM/modeling-helping/modeling/output/q_table_purple.csv")
    Q_red = load_q_table_from_csv("/Users/lvxinyuan/me/1-Projects/NYU/1-Courses/24_Fall_CCM/modeling-helping/modeling/output/q_table_red.csv")
    
    
    for session in range(sessions):
        
        for layer in layers:
            print(f"Start session {session}, game {layer}")
            
            random_resource_cond = random.choice(resource_conditions)
            random_cost_cond = random.choice(cost_conditions)
            random_visibility_cond = random.choice(visibility_conditions)
            random_red_first = random.choice(red_first_choices)
            
            # print(f"next test farm: {random_resource_cond}, {random_cost_cond}, {random_visibility_cond}, {random_red_first}")
            # Configure the game for test
            test_farm = farmgame.configure_game(layer=layer,               
                                                resourceCond=random_resource_cond, 
                                                costCond=random_cost_cond,
                                                visibilityCond=random_visibility_cond, 
                                                redFirst=random_red_first
            )
            
            # Run the test agent and collect game data
            game_data = test_agents(test_farm, Q_red, Q_purple, session)
            # print("game data collected")
            all_game_data.append(game_data)
            
            # print(f"Finish session {session}, game {layer}")
        
        print(f"Finish session {session}")
    write_to_csv("game_results.csv", all_game_data)



if __name__ == "__main__":
    main()
    

    