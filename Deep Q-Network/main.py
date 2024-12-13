import torch
import torch.nn as nn
import torch.nn.functional as F
#import shortuuid
import random
import farmgame
import copy
import matplotlib.pyplot as plt
import numpy as np

# Ensure device is set up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# (Kept the QNetwork and OpponentNetwork classes from the original code, but added .to(device) in __init__)
class TrainingMetrics:
    def __init__(self):
        self.q_losses = []
        self.opponent_losses = []
        self.rewards = {'red': [], 'purple': []}
        self.epsilon_history = []
        self.q_values_stats = []

    def plot_training_metrics(self):
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(15, 10))
        
        # Q-Network Loss with Smoothing
        plt.subplot(2, 2, 1)
        if len(self.q_losses) > 0:
            window = min(100, len(self.q_losses))
            smoothed_losses = np.convolve(self.q_losses, np.ones(window)/window, mode='valid')
            plt.plot(smoothed_losses, label='Smoothed Loss', color='blue')
            plt.plot(self.q_losses, alpha=0.3, label='Raw Loss', color='lightblue')
            plt.title('Q-Network Training Loss')
            plt.xlabel('Training Iterations')
            plt.ylabel('Loss')
            plt.legend()

        # Opponent Network Loss
        plt.subplot(2, 2, 2)
        if len(self.opponent_losses) > 0:
            plt.plot(self.opponent_losses)
            plt.title('Opponent Network Training Loss')
            plt.xlabel('Training Iterations')
            plt.ylabel('Loss')

        # Agent Rewards
        plt.subplot(2, 2, 3)
        if len(self.rewards['red']) > 0 and len(self.rewards['purple']) > 0:
            plt.plot(self.rewards['red'], label='Red Agent', color='red')
            plt.plot(self.rewards['purple'], label='Purple Agent', color='purple')
            plt.title('Agent Rewards Over Time')
            plt.xlabel('Games')
            plt.ylabel('Reward')
            plt.legend()

        # Epsilon Decay
        plt.subplot(2, 2, 4)
        if len(self.epsilon_history) > 0:
            plt.plot(self.epsilon_history)
            plt.title('Epsilon Decay')
            plt.xlabel('Training Steps')
            plt.ylabel('Epsilon Value')

        plt.tight_layout()
        plt.show()

    
class QNetwork(nn.Module):
    def __init__(self, input_channels=4, feature_vector_size=13, input_size=20, output_size=400):
        super(QNetwork, self).__init__()
        # (Previous implementation remains the same)
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Fully connected layers
        conv_output_size = input_size // (2**4)  # Adjust for the additional convolutional layer
        flattened_size = 64 * (conv_output_size ** 2)
        self.sub_fc1 = nn.Linear(feature_vector_size, 32)  # Feature vector layer from 13 to 32
        self.bn_subfc1 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(flattened_size + 32, 96)
        self.bn_fc1 = nn.BatchNorm1d(96)
        self.fc2 = nn.Linear(96, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 128)  # Added extra FC layer
        self.bn_fc3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, output_size)  # Final output layer
        # Move network to GPU if available
        self.to(device)
    
    def forward(self, x, feature_vector):
        # Ensure inputs are on the same device
        x = x.to(device)
        feature_vector = feature_vector.to(device)
        
        # (Rest of the forward method remains the same)
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.elu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.elu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)  # Flatten CNN output
        #weights = self.bn1.weight.data
        # Fully connected layers for feature vector
        y = F.elu(self.bn_subfc1(self.sub_fc1(feature_vector)))
        
        # Concatenate CNN and feature vector outputs
        x = torch.cat((x, y), dim=1)
        
        # Fully connected layers with BatchNorm
        x = F.elu(self.bn_fc1(self.fc1(x)))
        x = F.elu(self.bn_fc2(self.fc2(x)))
        x = F.elu(self.bn_fc3(self.fc3(x)))
        q_values = self.fc4(x)  # Final layer still no activation
        return q_values

class OpponentNetwork(nn.Module):
    def __init__(self, input_channels=4, feature_vector_size=13, input_size=20, output_size=400):
        super(OpponentNetwork, self).__init__()
        # (Previous implementation remains the same)
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Fully connected layers
        conv_output_size = input_size // (2**4)  # Adjust for the additional convolutional layer
        flattened_size = 64 * (conv_output_size ** 2)
        self.sub_fc1 = nn.Linear(feature_vector_size, 32)  # Feature vector layer from 13 to 32
        self.bn_subfc1 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(flattened_size + 32, 96)
        self.bn_fc1 = nn.BatchNorm1d(96)
        self.fc2 = nn.Linear(96, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 128)  # Added extra FC layer
        self.bn_fc3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, output_size)  # Final output layer
        # Move network to GPU if available
        self.to(device)
    
    def forward(self, x, feature_vector):
        # Ensure inputs are on the same device
        x = x.to(device)
        feature_vector = feature_vector.to(device)
        
        # (Rest of the forward method remains the same)
        
        # Convolutional layers with BatchNorm and MaxPooling
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.elu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.elu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)  # Flatten CNN output
        
        # Fully connected layers for feature vector
        y = F.elu(self.bn_subfc1(self.sub_fc1(feature_vector)))
        
        # Concatenate CNN and feature vector outputs
        x = torch.cat((x, y), dim=1)
        
        # Fully connected layers with BatchNorm
        x = F.elu(self.bn_fc1(self.fc1(x)))
        x = F.elu(self.bn_fc2(self.fc2(x)))
        x = F.elu(self.bn_fc3(self.fc3(x)))
        
        policy_logits = F.softmax(self.fc4(x), dim=1)
        return policy_logits

class DQWON(object):
    def __init__(self, **kwargs):
        # Previous initialization remains the same
        #self.id = shortuuid.uuid()
        self.metrics = TrainingMetrics()
        self.states = []
        self.current_state = None
        self.state_after_action = None
        self.action_count = 0
        self.epsilon=1
        self.policy_update_after=1000
        self.target_update_frequency = 100
        self.gamma = 0.9
        self.identity = kwargs.get('color','red')
        self.agent_type="deep-q-w/opponent network"
        self.verbose = kwargs.get('verbose',False) 
        self.agent_information = {"color":self.identity, "agent_type":self.agent_type}
        self.learnt_enough = True
        
        #4 channels 1:red player location, 2, purple player location, 3: red veggies. 4: purple veggies
        # Explicitly move tensors to device
        self.farm_representation = torch.zeros(4, 20, 20, device=device)
        self.features = torch.zeros(10, device=device)
        self.valid_action_mask = torch.zeros((1, 400), device=device)
        
        # Move networks to device
        self.q_network = QNetwork(input_channels=4, input_size=20, output_size=400).to(device)
        self.target_network = copy.deepcopy(self.q_network)
        self.target_network.eval()
        
        self.opponent_network = OpponentNetwork(input_channels=4, input_size=20, output_size=400).to(device)
        
        self.batch_size = kwargs.get('batch_size', 10)
        self.epochs = kwargs.get('epochs', 10)
        
        self.q_experience_inputs = []
        self.q_experience_feature_inputs = []
        self.q_experience_targets = []
        
        self.op_experience_inputs = []
        self.op_experience_feature_inputs = []
        self.op_experience_targets = []
        
        
    def new_game(self):
        self.state_after_action = None
    
    def update_target_network(self, step):
        if step % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
    
    def state_to_tensor(self, state):
        # Create tensors on the correct device
        tensor_representation = torch.zeros(4, 20, 20, device=device)
        tensor_features = torch.zeros(13, device=device)
        action_mask_for_state = torch.zeros(400, device=device)
        
        # (Rest of the method remains the same, just ensure tensors are created on device)
        for i in state.players:
            if (i['name']=='red'):
                tensor_representation[0, i['loc']['x'], i['loc']['y']] = 1
                tensor_features[0] = i['energy']
                hand = i['backpack']['contents']
                for item in hand:
                    if (item.color=='red'):
                        tensor_features[1]+=1
                    else:
                        tensor_features[2]+=1
                tensor_features[3] = i['backpack']['capacity']-len(hand)
            else:
                tensor_representation[1, i['loc']['x'], i['loc']['y']] = 1
                tensor_features[4] = i['energy']
                hand = i['backpack']['contents']
                for item in hand:
                    if (item.color=='purple'):
                        tensor_features[5]+=1
                    else:
                        tensor_features[6]+=1
                tensor_features[7] = i['backpack']['capacity']-len(hand)
        for deposit in state.farmbox.contents:
            if(deposit.color=='red'):
                tensor_features[8]+=1
            else:
                tensor_features[9]+=1
                
        for action in state.legal_actions():
            if(action.color=='red'):
                channel = 2
            elif(action.color=='purple'):
                channel = 3
            else:
                if(state.whose_turn()['name']=='red'):
                    channel = 2
                else:
                    channel = 3
            tensor_representation[channel, action.loc['x'], action.loc['y']] = 1
            action_mask_for_state[20*action.loc['x']+action.loc['y']] = 1
        
        action_mask_for_state = action_mask_for_state.unsqueeze(0).float()
        
        tensor_features[10] = tensor_features[0]*(tensor_features[1]+tensor_features[6]+tensor_features[8])
        tensor_features[11] = tensor_features[4]*(tensor_features[2]+tensor_features[5]+tensor_features[9])
        
        if(state.costCond=='high'):
            tensor_features[-1] = 1
        
        return tensor_representation, action_mask_for_state.unsqueeze(0).float(), tensor_features
    
    
    def state_to_features(some_state):
        return 0
    # Take a game state and append it to the history
    def update(self, state, repeat):
        
        #self.states.append(state)
        
        self.current_state = copy.deepcopy(state)
        
        self.farm_representation, self.valid_action_mask, self.features = self.state_to_tensor(self.current_state)
        
        if(not repeat and self.learnt_enough and self.state_after_action):
            self.remember_op_actions()
            
    def remember_op_actions(self):
        curr_tensor, _, _ = self.state_to_tensor(self.current_state)
        action_found = False  # Track if a valid action is found

        for action in self.state_after_action.legal_actions():
            possible_returned_state = self.state_after_action.take_action(action, inplace=False)
            possible_returned_tensor, _, possible_features = self.state_to_tensor(possible_returned_state)
            if torch.equal(possible_returned_tensor, curr_tensor):
                action_found = True
                break

        if not action_found:
            print("No valid actions found in remember_op_actions.")
            return  # Exit the method if no action is valid

        op_policy = torch.zeros((1, 400), device=device)
        action_index = 20 * action.loc['x'] + action.loc['y']
        op_policy[0, action_index] = 1

        self.op_experience_inputs.append(possible_returned_tensor.unsqueeze(0).float())
        self.op_experience_feature_inputs.append(possible_features.unsqueeze(0).float())
        self.op_experience_targets.append(op_policy)  # Add target Q-values
    
    def choose_action(self):
        
        """
        Choose an action using epsilon-greedy policy based on Q-network predictions.
        
        Args:
            epsilon (float): Exploration rate for epsilon-greedy policy
            
        Returns:
            Action
        """
        # Convert farm representation to correct format for network
        # Add batch dimension and ensure it's a float tensor
        
        if self.action_count > self.policy_update_after:
            self.learnt_enough = True
            self.action_count = 0
            self.epsilon = max(0.1, self.epsilon * 0.9)
            self.metrics.epsilon_history.append(self.epsilon)
            print(f"Changing policy to {self.epsilon} greedy")

        self.action_count += 1
        state_input = self.farm_representation.unsqueeze(0).float()
        feature_input = self.features.unsqueeze(0).float()

        self.q_network.eval()

        with torch.no_grad():
            q_values = self.q_network(state_input, feature_input)
            # Fill illegal actions with -inf
            masked_q_values = q_values.masked_fill(self.valid_action_mask == 0, float('-inf'))
            action_idx = torch.argmax(masked_q_values).item()

        legal_actions = self.current_state.legal_actions()
        if not legal_actions:
            print(f"No legal actions available for {self.identity}.")
            return None  # Or return a default action, if applicable.

        action = random.choice(legal_actions)

        if (torch.rand(1).item() >= self.epsilon and self.current_state.playersDict[self.identity]['energy'] > 0):
            x = action_idx // 20
            y = action_idx % 20
            for a in legal_actions:
                if a.loc['x'] == x and a.loc['y'] == y:
                    action = a
                    break

        target_q_values = self.calculate_target_values()
        self.store_experience(state_input, feature_input, target_q_values)

        self.state_after_action = self.current_state.take_action(action, inplace=False)

        if self.action_count % self.batch_size == 0:
            self.train(batch_size=self.batch_size)
            if self.learnt_enough:
                self.train_ON(batch_size=self.batch_size)

        self.update_target_network(self.action_count)


        return action
    
    def calculate_target_values(self):
        # Use GPU-accelerated computation
        target_q_values = torch.zeros((1, 400), device=device)
        state = copy.deepcopy(self.current_state)
        
        # Use a list to store future states to minimize repeated computations
        future_states = []
        indices = []
        
        # Precompute future states and their indices
        for action in state.legal_actions():
            index = 20*action.loc['x']+action.loc['y']
            future_state = state.take_action(action, inplace=False)
            future_states.append(future_state)
            indices.append(index)
        
        # Batch process future states
        future_tensors = []
        future_masks = []
        future_features = []
        
        for fut_state in future_states:
            farm_tensor, action_mask, feature_vector = self.state_to_tensor(fut_state)
            future_tensors.append(farm_tensor.unsqueeze(0))
            future_masks.append(action_mask)
            future_features.append(feature_vector.unsqueeze(0))
        
        # Convert to batch tensors
        future_tensors = torch.cat(future_tensors)
        future_masks = torch.cat(future_masks)
        future_features = torch.cat(future_features)
        
        # Compute opponent predictions in batch
        with torch.no_grad():
            self.opponent_network.eval()
            op_policies = self.opponent_network(future_tensors, future_features)
            op_policies = torch.mul(op_policies, future_masks)
            op_policies = op_policies / op_policies.sum(dim=1, keepdim=True)
        
        # Compute expected values
        for i, (fut_state, index) in enumerate(zip(future_states, indices)):
            reward_result = fut_state.reward(self.identity)

            # Check if `reward_result` is an int or a tuple
            if isinstance(reward_result, tuple):
                rwd, done = reward_result
            else:
                rwd = reward_result
                done = False  # Assume `done` is False if not provided

            if done:
                target_q_values[0, index] += rwd
            elif fut_state.playersDict[self.identity]['energy'] == 0:
                target_q_values[0, index] = -100
            else:
                # Use batched Q-value computation
                with torch.no_grad():
                    next_farm_tensor, next_action_mask, next_features = self.state_to_tensor(fut_state)
                    next_state_input = next_farm_tensor.unsqueeze(0)
                    next_feature_input = next_features.unsqueeze(0)

                    next_q_values = torch.mul(
                        self.target_network(next_state_input, next_feature_input),
                        next_action_mask
                    )
                    max_q_value = next_q_values.max().item()

                # Compute expected quality
                expected_quality = max_q_value
                target_q_values[0, index] += self.gamma * expected_quality
        
        return target_q_values

    def store_experience(self, state_input, feature_input, target_q_values):
        
        """Store a new experience in the experience buffer"""
        # Append the new experience to the deque (buffer)
        self.q_experience_inputs.append(state_input)  
        self.q_experience_feature_inputs.append(feature_input)  
        self.q_experience_targets.append(target_q_values)  # Add target Q-values
        
    def train(self, batch_size=10, learning_rate=0.0001):
        # Similar to previous implementation, but ensure tensors are on GPU
        if len(self.q_experience_inputs) < batch_size:
            return
        
        optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.q_network.train()
        
        # Move batches to GPU
        state_batch = torch.cat(self.q_experience_inputs).to(device)
        feature_batch = torch.cat(self.q_experience_feature_inputs).to(device)
        target_batch = torch.cat(self.q_experience_targets).to(device)
        
        
        # Rest of the training method remains similar
        # Just ensure all computations happen on the same device
        
        print(f"Target Q-values stats: min={target_batch.min():.4f}, mean={target_batch.mean():.4f}, max={target_batch.max():.4f}")

        # Train for multiple epochs
        epochs = self.epochs
        
        loss_fn = nn.MSELoss()
        
        for epoch in range(epochs):
            # Forward pass
            predicted_q_values = self.q_network(state_batch, feature_batch)
            
            # Calculate loss
            
            loss = loss_fn(predicted_q_values, target_batch)
            
           
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            # Add this right after loss.backward() to monitor gradient values
            # for name, param in self.q_network.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name} grad stats: mean={param.grad.mean()}, max={param.grad.max()}, min={param.grad.min()}")
            nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1)
            optimizer.step()
            self.metrics.q_losses.append(loss.item())
            epoch += 1
            if self.verbose and (epoch + 1) % 10 == 0:  # Print every 10 epochs
                print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
        if self.verbose:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
        #for i in range(10): print(target_batch[i].max())
        #for i in range(10): print(predicted_q_values[i].max())
        # Clear the experience buffers after training
        self.q_experience_inputs = []
        self.q_experience_feature_inputs = []
        self.q_experience_targets = []
    
    def train_ON(self, batch_size=10, learning_rate=0.01):
        if len(self.op_experience_inputs) < batch_size:
            return

        optimizer = torch.optim.Adam(self.opponent_network.parameters(), lr=learning_rate)          
        # Set network to training mode
        self.opponent_network.train()

        # Concatenate experiences into batches and move to device
        state_batch = torch.cat(self.op_experience_inputs).to(device)
        feature_batch = torch.cat(self.op_experience_feature_inputs).to(device)
        target_batch = torch.cat(self.op_experience_targets).to(device)

        epochs = self.epochs
        for epoch in range(epochs):
            # Forward pass
            predicted_op_policy = self.opponent_network(state_batch, feature_batch)
            predicted_op_policy_log = torch.log(predicted_op_policy)  # Apply log directly

            # Calculate loss
            loss_fn = nn.KLDivLoss(reduction='batchmean')  # Explicitly set reduction to avoid warning
            loss = loss_fn(predicted_op_policy_log, target_batch)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.opponent_network.parameters(), max_norm=1)
            optimizer.step()
            self.metrics.opponent_losses.append(loss.item())

            if self.verbose and (epoch + 1) % 10 == 0:  # Print every 10 epochs
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        if self.verbose:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        # Clear the experience buffers after training
        self.op_experience_inputs = []
        self.op_experience_feature_inputs = []
        self.op_experience_targets = []
        

def plot_q_values_distribution(q_network, state_input, feature_input):
    """
    Plot Q-values distribution for a given state
    
    Args:
        q_network (QNetwork): The Q-network to evaluate
        state_input (torch.Tensor): Input state tensor
        feature_input (torch.Tensor): Input feature tensor
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np

    # Ensure inputs are on the correct device and have batch dimension
    if state_input.dim() == 3:
        state_input = state_input.unsqueeze(0)
    if feature_input.dim() == 1:
        feature_input = feature_input.unsqueeze(0)

    try:
        with torch.no_grad():
            q_network.eval()
            q_values = q_network(state_input.to(device), feature_input.to(device))
            q_values = q_values.cpu().numpy().flatten()

        plt.figure(figsize=(10, 6))
        plt.hist(q_values, bins=50, edgecolor='black')
        plt.title('Q-Values Distribution')
        plt.xlabel('Q-Value')
        plt.ylabel('Frequency')
        plt.show()
    except Exception as e:
        print(f"Error plotting Q-values distribution: {e}")

def plot_action_heatmap(q_values, valid_action_mask):
    """
    Plot heatmap of Q-values on the 20x20 grid
    
    Args:
        q_values (torch.Tensor or np.ndarray): Q-values tensor
        valid_action_mask (torch.Tensor or np.ndarray): Mask for valid actions
    """
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        # Convert to numpy if not already
        if torch.is_tensor(q_values):
            q_values = q_values.cpu().numpy()
        if torch.is_tensor(valid_action_mask):
            valid_action_mask = valid_action_mask.cpu().numpy()

        # Reshape Q-values to 20x20 grid
        q_map = q_values.reshape(20, 20)
        
        # Mask invalid actions
        masked_q_map = np.ma.masked_where(valid_action_mask.reshape(20, 20) == 0, q_map)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(masked_q_map, cmap='hot')
        plt.colorbar(label='Q-Value')
        plt.title('Q-Values Action Heatmap')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.show()
    except Exception as e:
        print(f"Error plotting action heatmap: {e}")
# Modify the main game loop to work with GPU
if __name__ == "__main__":
    # Existing game initialization code
    red_agent = DQWON(color="red", verbose=True, batch_size=30, epochs=20)
    purple_agent = DQWON(color="purple", verbose=True, batch_size=30, epochs=20)
    

    for game in range(10000):
        print()
        print("STARTING GAME: ", game+1)
        print()
        layer_rand = f"Items{random.randint(0,11):02}"
        TheFarm = farmgame.configure_game(layer=layer_rand,resourceCond=random.choice(['even', 'uneven']),costCond=random.choice(['low', 'high']),visibilityCond="full",redFirst=True)
        state = TheFarm

        done = False
        r, p = False, False

        red_agent.new_game() 
        purple_agent.new_game()

        while not done:
            red_agent.update(state, r) 
            purple_agent.update(state, p)

            currentplayer = state.players[state.turn]
            print()
            print(currentplayer['name'] + "'s turn!")

            if currentplayer['name'] == "red":
                action = red_agent.choose_action()
                print("action: ", red_agent.action_count, "regime: ",red_agent.epsilon)
                r = True
                p = False
            else:
                action = purple_agent.choose_action()
                r = False
                p = True

            if action is None:
                print("THIS SHOULDN'T happen")
                print(currentplayer['name'] + " has no more moves.")
                done = True
            else:
                print(currentplayer['name'] + " player picks " + action.name)
                state = state.take_action(action, inplace=False)

                # Check if the game is done based on rewards
                reward = state.reward(currentplayer['name'])

                # Get rewards for both agents
                red_reward = state.reward('red')
                purple_reward = state.reward('purple')

                # Log rewards
                red_agent.metrics.rewards['red'].append(red_reward if isinstance(red_reward, int) else red_reward[0])
                purple_agent.metrics.rewards['purple'].append(purple_reward if isinstance(purple_reward, int) else purple_reward[0])

                # Check game completion
                if isinstance(reward, tuple):
                    _, done = reward
                else:
                    # If reward is not returning done status, we need an alternative way to determine if game is done
                    done = False
        if game % 100 == 0:
            red_agent.metrics.plot_training_metrics()
            purple_agent.metrics.plot_training_metrics()
                    

        # Print final rewards
        red_reward = state.reward('red')
        purple_reward = state.reward('purple')
        print("red reward", red_reward if isinstance(red_reward, int) else red_reward[0])
        print("purple reward", purple_reward if isinstance(purple_reward, int) else purple_reward[0])
