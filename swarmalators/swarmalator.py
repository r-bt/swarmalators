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
        self.inherent_velocity = np.random.rand(self._agents, 2) 
        self.inherent_velocity[:, 0] = 0 # Inherent velocity in x-dir
        self.inherent_velocity[:, 1] = 0 # Inherent velocity in y-dir

        self.velocity = np.zeros((self._agents, 2))

        half_len = self._agents // 2

        self.c = np.random.rand(self._agents, 1)

        self._chiral = False # Whether to do chiral behaviours

        self.c[:half_len, 0] = 0.5
        self.c[half_len:, 0] = -0.5
        
        # Init phase state (0 is natural freuqnecy, 1 is phase)
        self.phase_state = np.random.rand(self._agents, 2)

        # half_len = len(self.phase_state) // 2
        self.phase_state[:half_len, 0] = 0
        self.phase_state[half_len:, 0] = 0
        # self.phase_state[:, 0] = 1/
        # self.phase_state[:, 1] *= 2*np.pi
        self.phase_state[:, 1] = np.linspace(0, 2 * np.pi, self._agents, endpoint=False)

        # Keep track of time between updates
        self._updated = time.time()

    def update(self, positions):
        """
        Perform one tick update of swarmalator model

        Note: We perform using matrix multiplication since numpy supports vectorization and is faster than for loops
        """

        # Calculate x_j - x_i and |x_j = x_i|
        vectors = positions[:, :2] - positions[:, :2][:, np.newaxis]
        distances = np.linalg.norm(vectors, axis=2)

        np.fill_diagonal(distances, 1e-6) # Avoid division by zero

        # Calculate the phase difference
        # Note: Multiply by -1 since we are doing x_i = x_j but we want x_j - x_i
        phase_difference = -1 * np.subtract.outer(self.phase_state[:, 1], self.phase_state[:, 1])

        # Calculate Q terms
        natural_frequencies = self.phase_state[:, 0]
        phase_normalized = natural_frequencies / np.absolute(natural_frequencies)
        phase_normalized = np.nan_to_num(phase_normalized)

        Q_x = (np.pi / 2) * np.absolute(np.subtract.outer(phase_normalized, phase_normalized))
        Q_theta = (np.pi / 4) * np.absolute(np.subtract.outer(phase_normalized, phase_normalized))

        if not self._chiral:
            Q_x = 0
            Q_theta = 0

        # Calculate cos and sin terms

        phase_cos_difference = np.cos(phase_difference - Q_x)
        phase_sin_difference = np.sin(phase_difference - Q_theta)

        # Calculate velocity contributions
        velocity_contributions = (self._A + self._J * phase_cos_difference[:, :, np.newaxis]) * vectors / distances[:, :, np.newaxis] - self._B * vectors / np.square(distances[:, :, np.newaxis])

        # Calculate chiral contribution
        chiral_contribtuion = self.c * np.stack([np.cos(self.phase_state[:, 1] + np.pi/2), np.sin(self.phase_state[:, 1] + np.pi / 2)], axis=1)

        if not self._chiral:
            chiral_contribtuion = 0

        # Calculate velocity and delta_phase
        velocity = chiral_contribtuion + 1/self._agents * np.sum(velocity_contributions, axis=1)
        delta_phase = self.phase_state[:, 0] + (self._K / self._agents) * np.sum(phase_sin_difference / distances, axis=1)

        print(phase_sin_difference)

        # Update phase and velocity
        self.phase_state[:, 1] += delta_phase * (time.time() - self._updated)
        self.velocity = velocity

        self._updated = time.time()
        self.phase_state[:, 1] %= 2 * np.pi
    
    def get_phase_state(self):
        return self.phase_state
    
    def get_velocity(self):
        return self.velocity





       


