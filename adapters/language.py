from typing import Dict, List
import pandas as pd
import torch
from torch import Tensor
# StateAdapter includes static methods for adapters
from elsciRL.encoders.sentence_transformer_MiniLM_L6v2 import LanguageEncoder

class LanguageAdapter:
    _cached_state_idx: Dict[str, int] = dict()

    def __init__(self, setup_info:dict={}) -> None:
        # Language encoder doesn't require any preset knowledge of env to use
        self.encoder = LanguageEncoder()

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
                9:'You are on ice.', 
                10:'You are on ice, to your east is an ice hole.', 
                11:'You fell through a hole in the ice!',
                12:'You fell through a hole in the ice!', 
                13:'You are on ice, to your west is an ice hole.', 
                14:'You are on ice.', 
                15:'You found the chest!'
            }
        else:
            raise ValueError("LanguageAdapter only supports 4x4 grid size.")
    
    def adapter(self, state:any, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """ Use Language name for every id for current character position """
        # ---
        # Convert to language
        state = self.obs_mapping[state]
        # ---
        
        # Encode to Tensor for agents
        if encode:
            state_encoded = self.encoder.encode(state=state)
        else:
            state_encoded = state

        if (indexed):
            state_indexed = list()
            for sent in state:
                if (sent not in LanguageAdapter._cached_state_idx):
                    LanguageAdapter._cached_state_idx[sent] = len(LanguageAdapter._cached_state_idx)
                state_indexed.append(LanguageAdapter._cached_state_idx[sent])

            state_encoded = torch.tensor(state_indexed)

        return state_encoded