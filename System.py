import numpy as np

# Numerical model of a medium inside a laser cavity.
# Used to calculate the density matrix of the medium
# See https://doi.org/10.1080/09500340.2013.865806 for specifics of the algorithm


class System:
    # -------------
    # Define Fields
    # -------------
    # Default Fields are that of a simple Two Level System

    N = 2  # Number of levels
    omega = [5]  # (Hz) -- Rabi Frequency(ies) of the System
    fixed_deltas, delta = [0], 0  # Fixed Detuning(s), Signal Detuning
    QSource = np.array([[0, 1], [0, 0]])  # Source Matrix
    dephasing, QDephase = False, np.array([[0, 0], [0, 0]])  # Set to True if the system has dephasing, Dephase matrix
    ham = None  # Method for the Hamiltonian of the System

    # -----------
    # Constructor
    # -----------

    # Create the object with the given key word parameters
    def __init__(self, **kwargs):
        # Loops through the key word arguments and sets the corresponding parameters
        for key, value in kwargs.items():
            match key:
                case 'omega':
                    self.omega = value
                case 'N':
                    assert type(value) == int, "The system must have an integer number of energy levels!"
                    self.N = value
                case 'QSource':
                    assert np.shape(value) == (self.N, self.N), "The source matrix must be an N x N matrix!"
                    self.QSource = np.array(value)
                case 'dephase':
                    assert type(value) == bool, "The dephasing parameter must be a boolean!"
                    self.dephase = value
                case 'QDephase':
                    assert np.shape(value) == (self.N, self.N), "The dephase matrix must be an N x N matrix!"
                    self.QDephase = np.array(value)
                case 'fixed_deltas':
                    self.fixed_deltas = value
                case 'delta':
                    assert type(value) == int, "The detuning must be an integer!"
                    self.delta = value
                case 'ham':
                    assert callable(value), "The Hamiltonian must be entered as a function!"
                    assert np.shape(value(0, 0)) == (self.N, self.N), \
                           "The Hamiltonian function must return a N x N matrix!"
                    self.ham = value

        # Sets the Hamiltonian to that of a simple two level system if no Hamiltonian is provided
        if self.ham is None:
            self.ham = self.ham_2level

    #  Get the N x N density matrix
    def get_rho(self):
        S, W = self.get_MSW_opt()[1:]
        B = -np.matmul(np.linalg.inv(W), S)

        rhoNN = 1
        for i in range(self.N - 1):
            rhoNN -= B[i * (self.N + 1)]
        B = np.append(B, [rhoNN])
        B = np.reshape(B, (self.N, self.N))
        return B

    #  Get the M, S, & W matrices
    #  Less optimized, more intuitive algorithm
    def get_MSW(self):
        # Calculate the Hamiltonian for the current detuning of the system
        H = np.array(self.ham(self.delta, self.omega[-1]))

        # Initialize the N x N density matrix
        rho = np.zeros((self.N, self.N))

        # Initialize the N^2 x N^2 M matrix
        M = np.zeros((self.N**2, self.N**2), dtype=complex)

        # Reshape the dephase matrix
        if self.dephasing:
            assert np.shape(self.QDephase) == (self.N, self.N), "The dephase matrix must be an N x N matrix!"
            dephase = np.reshape(self.QDephase, (1, self.N ** 2))[0]

        # Calculate the elements of M
        for i in range(self.N**2):
            for k in range(self.N**2):
                # Calculate Indices
                beta = self.__nzrem(i+1, self.N) - 1
                alpha = self.__long_inc(i, beta) - 1
                sigma = self.__nzrem(k+1, self.N) - 1
                epsilon = self.__long_inc(k, sigma) - 1

                # Set an element of the density matrix to unity
                rho[epsilon, sigma] = 1

                # Calculate the Q matrix
                Q = (-1j) * (np.matmul(H, rho) - np.matmul(rho, np.conj(H)))

                # Reset the density matrix
                rho[epsilon, sigma] = 0

                # Set corresponding element of M
                M[i][k] = Q[alpha][beta]

                # Add the source terms
                if i == self.N**2-1:
                    if epsilon == sigma:
                        M[np.arange(0, self.N**2, self.N+1), k] += self.QSource[:, epsilon]

                # Add the dephasing terms
                if self.dephasing and i == k:
                    M[i][k] += dephase[i]

        # Define the S matrix
        S = np.copy(M[:-1, -1])

        # Define the reduced M matrix M'
        W = np.copy(M[0:-1, 0:-1])

        # Calculate the W matrix
        for i in range(self.N-1):
            W[:, i*(self.N+1)] -= S

        return M, S, W

    # Optimized algorithm to calculate M, S, & W matrices
    def get_MSW_opt(self):
        # Initialize the Hamiltonian
        H = np.array(self.ham(self.delta, self.omega[-1]))
        # print(H[2,0])

        # Initialize the N^2 x N^2 M matrix
        M = np.zeros((self.N ** 2, self.N ** 2), dtype=complex)

        col = -1
        index1 = np.arange(0, self.N)
        index2 = np.arange(0, self.N * (self.N - 1) + 1, self.N)
        index3 = np.arange(0, self.N**2, self.N + 1)

        for i in range(self.N):
            for j in range(self.N):
                col += 1
                M[(index1 + (i * self.N)), col] = 1j * np.conj(H[:, j])
                M[(index2 + j), col] -= 1j * np.array(H[:, i], dtype=complex)
                if i == j:
                    M[index3, col] += self.QSource[:, i]

        if self.dephasing:
            assert np.shape(self.QDephase) == (self.N, self.N), "The dephase matrix must be an N x N matrix!"
            dephase = np.reshape(self.QDephase, (1, self.N ** 2))[0]
            for i in range(len(dephase)):
                M[i][i] += dephase[i]

        # Define the matrix S
        S = np.copy(M[:-1, -1])
        # Define the reduced M matrix M'
        W = np.copy(M[0:-1, 0:-1])

        # Calculate W matrix
        for i in range(self.N - 1):
            W[:, i * (self.N + 1)] -= S

        return M, S, W

    # --------------
    # Helper Methods
    # --------------
    @staticmethod
    def __nzrem(a, b):
        rem = a % b
        if rem == 0:
            return b
        else:
            return rem

    def __long_inc(self, a, b):
        return int(1 + (a - b)/self.N)

    # ----------------------------------------------
    # Hamiltonians of various simple N level Systems
    # ----------------------------------------------

    # Hamiltonian of a simple 2 Level System
    def ham_2level(self, delta, omega):
        # There should be one Rabi frequency for a 2 Level System
        assert len(self.omega) == 1, "There should only be one Rabi frequency for this system!"
        return [[0, omega / 2],
                [omega / 2, -1 * (delta + 0.5j)]]

    # Hamiltonian of a 3 Level System
    def ham_3level(self, delta, omega):
        # There should be two Rabi frequencies for a 3 Level System
        assert len(self.omega) == 2, "There should only be two Rabi frequencies for this system!"
        return [[delta/2, 0, self.omega[0]/2],
                [0, -delta/2, omega/2],
                [self.omega[0]/2, omega/2, -(self.fixed_deltas[0] + 0.5j)]]
