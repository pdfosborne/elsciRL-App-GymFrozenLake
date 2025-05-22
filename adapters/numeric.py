from typing import Dict, List
import pandas as pd
import torch
from torch import Tensor
# StateAdapter includes static methods for adapters
from elsciRL.encoders.poss_state_encoded import StateEncoder
from gymnasium.spaces import Discrete

class Adapter:
    _cached_state_idx: Dict[str, int] = dict()

    def __init__(self, setup_info:dict={}) -> None:
        # TODO: Update this based on the current problem, each requires preset knowledge of all possible states/actions/objects
        # - Possible States
        # - Possible Actions
        # - Prior Actions
        # - Possible Objects
    
        # Initialise encoder based on all possible env states
        env_size = setup_info['environment_size']
        num_positions = int(env_size.split("x")[0]) * int(env_size.split("x")[-1])
        possible_positions = [i for i in range(num_positions)]
        self.encoder = StateEncoder(possible_positions)
        # --------------------------------------------------------------------
        # ONLY IF USING GYMNASIUM BASED AGENTS
        # - Observation space is required for Gym based agent, prebuilt HELIOS encoders provide this (TODO)
        # self.observation_space = self.encoder.observation_space
        # - Otherwise defined here:
        #   - See https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
        #   - Observations are dictionaries with the agent's and the target's location.
        #   - Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        
        # State encoder converts 2-d space to a discrete array of size 4*4
        # -> Now we have actions in observation need to extend observation space
        self.observation_space = Discrete(num_positions)
           
    def adapter(self, state:any, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """ Use State Encoder to convert id position to tensor """    
        # Encode to Tensor for agents
        if encode:
            state_encoded = self.encoder.encode(state=(state), indexed=True)   
        else:
            state_encoded = state

        if (indexed):
            state_indexed = list()
            for sent in state:
                if (sent not in Adapter._cached_state_idx):
                    Adapter._cached_state_idx[sent] = len(Adapter._cached_state_idx)
                state_indexed.append(Adapter._cached_state_idx[sent])

            state_encoded = torch.tensor(state_indexed)

        return state_encoded