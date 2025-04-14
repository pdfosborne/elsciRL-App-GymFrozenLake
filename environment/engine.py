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

    def reset(self, start_obs:any=None):
        """Fully reset the environment."""
        obs, _ = self.Environment.reset()
        self.action_history = []
        self.obs_history = []
        return (obs,(0,0))
    
    def step(self, state:any=None, action:any=None):
        """Enact an action."""
        # Record action history
        self.action_history.append(action)
        # Action space will always be numeric, step function may need to #
        # convert to form to match underlying engine, example:
        # self._action_to_outcome = {
        #     0: np.array([1, 0]),
        #     1: np.array([0, 1]),
        #     2: np.array([-1, 0]),
        #     3: np.array([0, -1])
        # }
        # Return outcome of action
        obs, reward, terminated, truncated, info = self.Environment.step(action)
        self.obs_history.append(obs)
        if terminated:
            if len(self.sub_goal_obs)>0:
                all_found = True
                for sub_goal in self.sub_goal_obs:
                    if sub_goal not in self.obs_history:
                        all_found = False
                if all_found:
                    reward = 1
                else:
                    reward = 0

            if obs!=self.terminal_goal:
                reward = -1
            else:
                reward = reward
        else:
            reward = -0.0025

        if len(self.action_history)>=100:
            reward = 0
            terminated = True

        # Add inventory to observation
        if 3 in self.obs_history:
            if 15 in self.obs_history:
                obs_inv = (obs, (1,1))
            else:
                obs_inv = (obs, (1,0))
        elif 15 in self.obs_history:
            obs_inv = (obs, (0,1))
        else:
            obs_inv = (obs, (0,0))
            
        return obs_inv, reward, terminated, info

    def legal_move_generator(self, obs:any=None):
        """Define legal moves at each position"""
        legal_moves = [0,1,2,3]#self.Environment.legal_moves(obs)
        return legal_moves
    
    def render(self, state:any=None):
        """Render the environment."""
        render = self.Environment.render()
        return render
    
    def close(self):
        """Close/Exit the environment."""
        self.Environment.close()
