import gymnasium as gym
import time
class Engine:
    """Defines the environment function from the generator engine.
       Expects the following:
        - reset() to reset the env a start position(s)
        - step() to make an action and update the game state
        - legal_moves_generator() to generate the list of legal moves
    """
    def __init__(self, local_setup_info:dict={}) -> None:
        """Initialize Engine"""
        self.Environment = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='rgb_array' )
        self.setup_info = local_setup_info
        # Specify required positions that must be reached before the episode is terminated
        self.sub_goal_obs = [3] 
        self.terminal_goal = 15 # Specify terminal goal so we can introduce negative reward for not reaching it
    
        # Ledger of the environment with meta information for the problem
        ledger_required = {
            'id': 'Unique Problem ID',
            'type': 'Language/Numeric',
            'description': 'Problem Description',
            'goal': 'Goal Description'
            }
        
        ledger_optional = {
            'reward': 'Reward Description',
            'punishment': 'Punishment Description (if any)',
            'state': 'State Description',
            'constraints': 'Constraints Description',
            'action': 'Action Description',
            'author': 'Author',
            'year': 'Year',
            'render_data':{'render_mode':'rgb_array', 
                           'render_fps':4}
        }
        ledger_gym_compatibility = {
            # Limited to discrete actions for now, set to arbitrary large number if uncertain
            'action_space_size':4, 
        }
        self.ledger = ledger_required | ledger_optional | ledger_gym_compatibility
        # Initialize history
        self.action_history = []
        self.obs_history = []
        self.action_limit = local_setup_info['action_limit']

    def reset(self, start_obs:any=None):
        """Fully reset the environment."""
        obs, _ = self.Environment.reset()
        self.action_history = []
        self.obs_history = []
        return obs
    
    def step(self, state:any=None, action:any=None):
        """Enact an action."""
        # Record action history
        self.action_history.append(action)
       
        # Return outcome of action
        obs, reward, terminated, truncated, info = self.Environment.step(action)
        self.obs_history.append(obs)
        
        if len(self.action_history)>=self.action_limit:
            reward = 0
            terminated = True
            
        return obs, reward, terminated, info

    def legal_move_generator(self, obs:any=None):
        """Define legal moves at each position"""
        legal_moves = [0,1,2,3] #self.Environment.legal_moves(obs)
        return legal_moves
    
    def render(self, state:any=None):
        """Render the environment."""
        render = self.Environment.render()
        return render
    
    def close(self):
        """Close/Exit the environment."""
        self.Environment.close()
