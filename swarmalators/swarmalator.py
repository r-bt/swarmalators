import numpy as np
import time

class Swarmalator:

    def __init__(self, agents: int, K: int, J: int, A:int = 1, B:int = 1):
        """
        Intialize swarmalator model

        Parameters
        ----------
        agents : int
                Number of agents
        K : int
            Phase coupling coefficient
        J: int
            Spatial-phase interaction coeffi- cient 
        positions: list
            Initial positions of all the agents
        """
        np.random.seed(0) # Debug have the same random numbers

        self._agents = agents
        self._K = K
        self._J = J
        self._A = A
        self._B = B

        # Init positon state
        self.position_state = np.random.rand(self._agents, 2) 
        self.position_state[:, 0] = 0 # Inherent velocity in x-dir
        self.position_state[:, 1] = 0 # Inherent velocity in y-dir

        self.velocity = None
        
        # Init phase state
        self.phase_state = np.random.rand(self._agents, 2)

        half_len = len(self.phase_state) // 2
        self.phase_state[:half_len, 0] = 1
        self.phase_state[half_len:, 0] = -1
        # self.phase_state[:, 0] = 0

        self.phase_state[:, 1] *= 2 * np.pi # Inital phase is random value between [0, 2pi]

        # Keep track of time between updates
        self._updated = time.time()

    def update(self, positions):
        """
        Perform one tick update of swarmalator model
        """

        phases = self.phase_state[:, 1:]

        phase_sin_difference = np.sin(phases.T - phases) 
        phase_cos_difference = np.cos(phases.T - phases)

        # Now we get get vectors between each agent and calculate magnitudes
        vectors = positions[:, :2] - positions[:, :2][:, np.newaxis]
        pairwise_distances = np.linalg.norm(vectors, axis=2)

        with np.errstate(divide='ignore', invalid='ignore'):
            phase_sum = np.where(pairwise_distances != 0, phase_sin_difference / pairwise_distances, 0)
            position_sum = np.where(
                pairwise_distances[:, :, np.newaxis] != 0,
                (vectors / pairwise_distances[:, :, np.newaxis]) * (self._A + self._J * phase_cos_difference[:, :, np.newaxis]) - self._B * (vectors / (np.square(pairwise_distances)[:, :, np.newaxis])),
                0
            )

        phase_sum = np.sum(phase_sum, axis=1)
        position_sum = np.sum(position_sum, axis=1)

        delta_phases = self.phase_state[:, 0] + (self._K/self._agents) * phase_sum

        delta_position = self.position_state + (1/self._agents) * position_sum

        self.phase_state[:, 1] += delta_phases * (time.time() - self._updated)

        self.velocity = delta_position

        self._updated = time.time()

        # Bound the values between 0 and 2 pi
        self.phase_state[:, 1] %= 2 * np.pi
    
    def get_phase_state(self):
        return self.phase_state
    
    def get_velocity(self):
        return self.velocity



       


