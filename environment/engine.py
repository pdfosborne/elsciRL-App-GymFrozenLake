from scienceworld import ScienceWorldEnv
class Engine:
    """Defines the environment function from the generator engine.
       Expects the following:
        - reset() to reset the env a start position(s)
        - step() to make an action and update the game state
        - legal_moves_generator() to generate the list of legal moves
    """
    def __init__(self, task:str='1-1') -> None:
        """Initialize Engine"""
        self.Environment = ScienceWorldEnv(task)
        
    def reset(self):
        """Fully reset the environment."""
        obs, _ = self.Environment.reset()
        inventory = self.Environment.inventory()
        look = self.Environment.look()
        
        obs = obs.replace('\n\t',' ').replace('\n', '')
        inventory = inventory.replace('\n\t',' ').replace('\n', '')
        look = look.replace(': \n\t',': ').replace(':\n\t',': ').replace('\n\t',', ').replace('\n', '. ')
        obs_output = obs + '. ' + inventory + '. ' + look
        return obs_output

    
    def step(self, state:any, action:any):
        """Enact an action."""
        # In problems where the agent can choose to reset the env
        if (state=="ENV_RESET")|(action=="reset task"):
            self.reset()
            
        obs, reward, terminated, _ = self.Environment.step(action)
        inventory = self.Environment.inventory()
        look = self.Environment.look()

        obs = obs.replace('\n\t',' ').replace('\n', '')
        inventory = inventory.replace('\n\t',' ').replace('\n', '')
        look = look.replace(': \n\t',': ').replace(':\n\t',': ').replace('\n\t',', ').replace('\n', '. ')
        obs_output = obs + '. ' + inventory + '. ' + look
        return obs_output, reward, terminated

    def legal_move_generator(self, obs:any=None):
        """Define legal moves at each position"""
        legal_moves = self.Environment.getValidActionObjectCombinations()
        return legal_moves
