#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 20:17:22 2024

@author: lvxinyuan
"""
import random
import farmgame
from farmgame import *
import csv
import re
import os
from scipy.spatial.distance import euclidean
from datetime import datetime
import itertools


''' Q learnining'''
def action_to_tuple(action: Action) -> tuple: # helper function
    try:
        result = (action.name, action.type, action.color, action.id, action.__str__())
        # print(f"Action to Tuple: {result}")
        return result
    except Exception as e:
        print(f"Error converting action to tuple: {action}. Exception: {e}")
        raise

def get_legal_actions(farm: Farm) -> list[Action]:
    try:
        actions = farm.legal_actions() # type(actios): <farmgame.Action object at 0x16141fa10>,...
        return actions
    except Exception as e:
        print(f"Error retrieving legal actions: {e}")
        raise

def get_state(farm: Farm) -> tuple:
    try:
        # Include energy, scores if visibility condition is "full"
        if farm.visibilityCond == "full":
            red_energy = (
                "full" if farm.redplayer["energy"] > 75 else
                "moderate" if farm.redplayer["energy"] > 50 else
                "low" if farm.redplayer["energy"] > 25 else
                "depleted"
            )
            pur_energy = (
                "full" if farm.purpleplayer["energy"] > 75 else
                "moderate" if farm.purpleplayer["energy"] > 50 else
                "low" if farm.purpleplayer["energy"] > 25 else
                "depleted"
            )
        else:
            red_energy = ""
            pur_energy = ""
        
        scores = (farm.redplayer["score"], farm.purpleplayer["score"]) if farm.visibilityCond == "full" else ()

        result = (
            # Env setting
            farm.objectLayer,
            farm.redfirst,
            farm.resourceCond,
            farm.costCond,  # farm.stepcost (the same thing)
            farm.visibilityCond,
            
            # Change after every action
            # (farm.redplayer["loc"]["x"], farm.redplayer["loc"]["y"]), # red location
            # (farm.purpleplayer["loc"]["x"], farm.purpleplayer["loc"]["y"]), # purple location
        
            # TODO need to debug
            (farm.redplayer['capacity'] - len(farm.redplayer['contents']) , 
             farm.purpleplayer['capacity'] - len(farm.purpleplayer['contents'])), # remain backpack capacity

            (red_energy, pur_energy), # visibilityCond=self -> empty tuple
            scores, # visibilityCond=self -> empty tuple
            (farm.opponent_has_helped("red"), farm.opponent_has_helped("purple")), # Has the player helped
            # tuple(action.name for action in farm.farmbox.contents), # farmbox content1 (incorporated in has the player helped)
            
        )
        return result
    except Exception as e:
        print(f"Error extracting state from farm: {e}")
        raise


def discount_distance(player_loc, veggie_full):
    match = re.search(r"\((\d+),(\d+)\)", veggie_full)
    if match:
        veggie_loc = int(match.group(1)), int(match.group(2))
    
    min = 0
    max = 19 
    normalized_euclidean = (euclidean(player_loc, veggie_loc) - min) / (max - min)

    return normalized_euclidean
    

def update_q_value(Q, state, action, reward, score, next_state, legal_actions, alpha, gamma):
    try:
        print("Update Q table...")
        current_q = Q.get((state, action), 0) # action=red, purple, box, pillow
        
        max_next_q = 0
        for a in legal_actions:
            # print(f"in update_q_value, action = {a.name}")
            action_id = None
            if a.type == ActionType.veggie:
                action_id = a.id
            else:
                action_id = a.name
            q_value = Q.get((next_state, action_id), 0) # default 0
            print(f"Action: {a.id}, action_id: {action_id}, Q-value: {q_value}")
            
            if q_value > max_next_q:
                max_next_q = q_value
        print(f"max_next_q = {max_next_q}")
        
        
        # updating Q table
        Q[(state, action)] = current_q + alpha * ((reward + score) + gamma * max_next_q - current_q)
        print(f"Updated Q-value for (state, action): {((state, action), Q[(state, action)])}")

    except Exception as e:
        print(f"Error updating Q-value: {e}")
        raise
    

def dual_q_learning(episodes, alpha, gamma, epsilon, num_games=12):
    # Initialize separate Q-tables
    Q_red, Q_purple = {}, {}

    for episode in range(episodes):
        # print(f"\nStarting Episode {episode + 1}")
        
        layers = ["Items00","Items01","Items02","Items03","Items04",
                  "Items05","Items06","Items07","Items08","Items09",
                  "Items10","Items11"]
        resource_conditions = ["even", "unevenRed", "unevenPurple"]  # 3 values
        cost_conditions = ["low", "high"]                           # 2 values
        visibility_conditions = ["full", "self"]                   # 2 values
        red_first_choices = [True,False]                                 # 1 value

        # Iterate over all combinations
        for layer, resource_cond, cost_cond, visibility_cond, red_first in itertools.product(
            layers, resource_conditions, cost_conditions, 
            visibility_conditions, red_first_choices
        ):
            # Configure the game with the specific combination
            farm = farmgame.configure_game(
                layer=layer,                  # Single value
                resourceCond=resource_cond,   # Single value
                costCond=cost_cond,           # Single value
                visibilityCond=visibility_cond,  # Single value
                redFirst=red_first            # Single value
            )
    
            # Initialize the game
            state = get_state(farm)
            red_score = 0
            purple_score = 0
            action_id = None
    
            while not farm.is_done():
                # Determine whose turn it is
                current_player = farm.whose_turn()["name"]
                Q = Q_red if current_player == "red" else Q_purple
                player_color = "Red" if current_player == "red" else "Purple"
    
                # Get legal actions
                actions = get_legal_actions(farm)
                # print(f"legal_actions = {[action.id for action in actions]}") # ['Tomato00', 'Turnip01'...]
                # print_state_details(state, player_color, actions)
    
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = random.choice(actions)
                    print("Epsilon-greedy-selection: Choose Random action.")
                else:
                    print("Epsilon-greedy-selection: Choose Max Q action.")
                    max_q_value = float('-inf')  # Initialize to negative infinity
                    action = None
                    
                    for a in actions:
                        action_id = None
                        if a.type == ActionType.veggie: 
                            action_id = a.id
                        else: 
                            action_id = a.name
                            
                            
                        q_value = Q.get((state, action_id), 0) #before modification, doesn't start with the closet veggie
                        print(f"in maxing Q val: action = {a.id}, acion_id = {action_id}, q_value = {q_value:.3f}")
                        
                        # Update the best action if the current Q-value is higher
                        if q_value > max_q_value:
                            max_q_value = q_value
                            action = a
                            print(f"current best action = {str(a)}")
                    print(f"max Q value action = {action.id}, max_q_value = {max_q_value:.3f}")
                print(f"{player_color} Player Chosen Action: {action}\n")
                
                # increment score
                if action.type == ActionType.veggie:
                    if action.name in ['tomato', 'strawberry']:
                        distance_discount = discount_distance((farm.redplayer["loc"]["x"], farm.redplayer["loc"]["y"]),
                                                            str(action))
                        red_score += 2 - distance_discount * farm.stepcost 
                        
                        print(f"step_cost = {farm.stepcost}, distance_discount = {distance_discount}, red_score = {red_score}")
                        action_id = action.id
                    elif action.name in ['turnip', 'eggplant']: 
                        distance_discount = discount_distance((farm.purpleplayer["loc"]["x"], farm.purpleplayer["loc"]["y"]),
                                                            str(action))
                        purple_score += 2 - distance_discount * farm.stepcost
                        print(f"step_cost = {farm.stepcost}, distance_discount = {distance_discount}, purple_score = {purple_score}")
                        action_id = action.id
                else:
                    action_id = action.name

                score = red_score  if current_player == "red" else purple_score
                
                
                # Next state and reward
                farm.take_action(action)
                next_state = get_state(farm)
                reward = farm.reward(current_player)
                # print(f"{player_color} Score: {score}")
                # print(f"Detailed: Red Score: {red_score}")
                # print(f"Detailed: Purple Score: {purple_score}")
                # print(f"{player_color} Reward: {reward}\n")
                
                # Update Q-table for the current player
                update_q_value(
                    Q, state, action_id, reward, score, next_state,
                    get_legal_actions(farm), alpha, gamma
                )
                

                # Transition to the next state
                state = next_state
            
            # print(f"Game {game_index + 1} in Episode {episode + 1} complete.\n")
            
        # print(f"Episode {episode + 1} complete.\n")

        # Reset the game for the next episode
        farm = configure_game()

    
    return Q_red, Q_purple


''' Debugging helper functions'''
def print_state_action_details(state, player_color, actions):

    if len(state) == 13:
        print(
                f"\n{player_color} Player's Turn...\n"
                f"**State**: object layer = {state[0]}, "
                f"redfirst = {state[1]}, "
                f"resource condition = {state[2]}, "
                f"cost condition = {state[3]}, "
                f"visibility condition = {state[4]}, "
                # f"red player location = {state[5]}, "
                # f"purple player location = {state[6]}, "
                f"player remaining capacities = {state[5]}, "
                f"player energy = {state[6]}, "
                f"player score = {state[7]}, " 
                f"has helped = {state[-1]}, "
            )
    else:
        print(
                f"\n{player_color} Player's Turn\n"
                f"State: object layer = {state[0]}, "
                f"redfirst = {state[1]}, "
                f"resource condition = {state[2]}, "
                f"cost condition = {state[3]}, "
                f"visibility condition = {state[4]}, "
                # f"red player location = {state[5]}, "
                # f"purple player location = {state[6]}, "
                f"player remaining capacities = {state[5]}, "
                f"has helped = {state[-1]}, "
            )
    print(f"**Legal Actions**: {[str(action) for action in actions]}\n")


def print_q_table(Q, player_name="Player"):
    """
    Prints the Q-table in a readable format.
    
    Parameters:
    - Q: The Q-table (dictionary).
    - player_name: Name of the player (e.g., "Red" or "Purple").
    """
    print(f"\nQ-Table for {player_name}:\n")
    if not Q:
        print("Q-Table is empty!")
        return

    for (state, action), q_value in Q.items():
        print(f"State: {state}, Action: {action}, Q-Value: {q_value:}")
    print("\nEnd of Q-Table\n")


def save_q_table_to_csv(Q, player_name="Player", filename="q_table.csv"):
    """
    Saves the Q-table to a CSV file.

    Parameters:
    - Q: The Q-table (dictionary).
    - player_name: Name of the player (e.g., "Red" or "Purple").
    - filename: The name of the CSV file to save the Q-table.
    """
    if not Q:
        print(f"Q-Table for {player_name} is empty. Nothing to save!")
        return
    
    current_folder = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(current_folder, "output")
    time_folder = os.path.join(output_folder, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(time_folder, exist_ok=True)
    full_file_path = os.path.join(time_folder, filename)
    
    with open(full_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["objectLayer", "redfirst", "resourceCond", 
                         "costCond", "visibilityCond", "redRemainCapacity", "purRemainCapacity",
                         "red_energy", "pur_energy", 
                         "red_score", "pur_score", 
                         "red_helped", "pur_helped",
                         "Action", "Q-Value"]) # write header
        
        for (state, action), q_value in Q.items(): # write state-action-Q-value
            writer.writerow([state[0], state[1], state[2],
                             state[3], state[4], state[5][0], state[5][1], 
                             state[6][0] if len(state[6]) > 0 else "N/A", state[6][1] if len(state[6]) > 0 else "N/A",
                             state[7][0] if len(state[7]) > 0 else "N/A", state[7][1] if len(state[7]) > 0 else "N/A",
                             state[8][0] if len(state[8]) > 0 else "N/A", state[8][1] if len(state[8]) > 0 else "N/A",
                             action, q_value])
    
    print(f"Q-Table for {player_name} saved to {full_file_path}.")


def test_agents(farm: Farm, Q_red, Q_purple, filename="test_agent.txt"):
    
    current_folder = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(current_folder, "output")
    time_folder = os.path.join(output_folder, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(time_folder, exist_ok=True)
    log_file_path = os.path.join(time_folder, filename)

    
    with open(log_file_path, mode='w') as file:
        state = get_state(farm)
        file.write(f"**Current State** = {state}\n")
        
        while not farm.is_done():
            current_player = farm.whose_turn()["name"]
            file.write(f"\n\nCurrent Player = {current_player.capitalize()}\n")
            Q = Q_red if current_player == "red" else Q_purple
            
            cur_location = (farm.redplayer["loc"]["x"], farm.redplayer["loc"]["y"]) if current_player == "red" else (farm.purpleplayer["loc"]["x"], farm.purpleplayer["loc"]["y"])
            oth_location = (farm.purpleplayer["loc"]["x"], farm.purpleplayer["loc"]["y"]) if current_player == "red" else (farm.redplayer["loc"]["x"], farm.redplayer["loc"]["y"])
            file.write(f"Current Player Location: {cur_location}, Other player location: {oth_location}\n")
            
    
            actions = get_legal_actions(farm)
            # print_state_action_details(state, current_player, actions)
            file.write(f"Legal Actions: {[str(action) for action in actions]}\n")
            
            
            # Choose the Max Q action
            max_q_value = float('-inf')  # Initialize to negative infinity
            action = None
            
            for a in actions:
                action_id = None
                if a.type == ActionType.veggie:
                    action_id = a.id
                else:
                    action_id = a.name
                q_value = Q.get((state, action_id), 0)
                # print(f"in maxing Q val: action = {a.id}, action_id = {action_id}, q_value = {q_value}")
                
                # Update the best action if the current Q-value is higher
                if q_value > max_q_value:
                    max_q_value = q_value
                    action = a
                    # print(f"-current best action = {str(a)}, q_value = {max_q_value}")
                    file.write(f"-current best action = {str(a)}, q_value = {max_q_value:.3f}\n")
            
    
            # print(f"\n{current_player.capitalize()} chooses action: {action.id}, max_q_value = {max_q_value}")
            file.write(f"\n{current_player.capitalize()} chooses action: {action.id}, max_q_value = {max_q_value}\n")
    
            # Update to next state
            farm.take_action(action)
            state = get_state(farm)
        
        file.write("Game Over!")
        file.write(f"Final Scores - Red: {farm.redplayer['score']}, Purple: {farm.purpleplayer['score']}\n")
        file.write(f"Final energy - Red: {farm.redplayer['energy']}, Purple: {farm.purpleplayer['energy']}\n")
        file.write(f"Red Bonus Points: {farm.redplayer['bonuspoints']}, Purple Bonus Points: {farm.purpleplayer['bonuspoints']}\n")
    print(f"Test results saved to {log_file_path}")


'''main'''
def main():
    # 1. train Q-learning agents (uses an exploration strategy)
    episodes = 1000   # Number of training episodes
    alpha = 0.1       # Learning rate
    gamma = 0.9       # Discount factor
    epsilon = 1    # Exploration probability
    
    print("-------Training Q-learning agents...-------")
    Q_red, Q_purple = dual_q_learning(episodes, alpha, gamma, epsilon)
    
    # Print final Q-tables
    # print_q_table(Q_red, player_name="Red")
    # print_q_table(Q_purple, player_name="Purple")
    print(f"Q_red length {len(Q_red)}")
    print(f"Q_purple length {len(Q_purple)}")
    
    
    # Save Q tables
    save_q_table_to_csv(Q_red, player_name="Red", filename="q_table_red.csv")
    save_q_table_to_csv(Q_purple, player_name="Purple", filename="q_table_purple.csv")
    
    print("Training complete!\n")
    

    # 3. test agent (does not update Q-values)
    print("-------Test agent----------")
    test_farm = configure_game(layer="Items00", 
                               resourceCond="even", 
                               costCond="low", 
                               visibilityCond="full", 
                               redFirst=True)
    test_agents(test_farm, Q_red, Q_purple, "test_agent_Item.txt")



if __name__ == "__main__":
    main()