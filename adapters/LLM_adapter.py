from torch import Tensor

import numpy as np
from gymnasium.spaces import Box

# Link to relevant ENCODER
from elsciRL.adapters.LLM_state_generators.text_ollama import OllamaAdapter


class Adapter:
    def __init__(self, setup_info:dict={}) -> None:   

        env_size = setup_info['environment_size']
        if env_size == "4x4":
            self.obs_mapping = {
                0:'You are at the start position.', 
                1:'You are on ice, to your south is an ice hole.',
                2:'You are on ice.', 
                3:'You are on ice, to your south is an ice hole.',
                4:'You are on ice, to your east is an ice hole.', 
                5:'You fell through a hole in the ice!', 
                6:'You are on ice, to your west and east are ice holes.', 
                7:'You fell through a hole in the ice!',
                8:'You are on ice, to your south is an ice hole.',
                9:'You are on ice, to your north is an ice hole.', 
                10:'You are on ice, to your east is an ice hole.', 
                11:'You fell through a hole in the ice!',
                12:'You fell through a hole in the ice!', 
                13:'You are on ice, to your west is an ice hole.', 
                14:'You are on ice.', 
                15:'You found the chest!'
            }
        else:
            raise ValueError("LanguageAdapter only supports 4x4 grid size.")
             
        # Define observation space
        self.observation_space = Box(low=-1, high=1, shape=(1,384), dtype=np.float32)


        self.LLM_adapter = OllamaAdapter(
            model_name=setup_info.get('model_name', 'llama3.2'),
            base_prompt=setup_info.get('system_prompt', 'You are playing a navigating a grid based Gym environment.'),
            context_length=2000,
            action_history_length=setup_info.get('action_history_length', 5),
            encoder=setup_info.get('encoder', 'MiniLM_L6v2')
        )

        
    def adapter(self, state: str, legal_moves:list = None, episode_action_history:list = None, encode:bool=True, indexed: bool = False) -> Tensor:     
        """ Use Language description for every student for current grid position """

        state = 'Original environment state '+ str(state) + ', transformed language description: ' + self.obs_mapping[state]


        # Use the elsciRL LLM adapter to transform and encode
        state_encoded = self.LLM_adapter.adapter(
            state=state, 
            legal_moves=legal_moves, 
            episode_action_history=episode_action_history, 
            encode=encode, 
            indexed=indexed
        )

        return state_encoded
    
    
            
            