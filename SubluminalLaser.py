import numpy as np
import scipy.constants as scons
import System
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# Steady State Condition 1 (SS1): -χ'' = 1/Q
# Steady State Condition 2 (SS2): ν(1 + χ'/2) = Ωc, ν = δrp - ν0
# ----------------------------------------------------------------------------------------------------------------------

class SubluminalLaser:
    # ------------------------------------------------------------------------------------------------------------------
    # Define Fields
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Shared Fields
    # ------------------------------------------------------------------------------------------------------------------
    delta_s = [0]  # (Hz) Detuning of the signal field
    omega_s = [0]  # (Hz) -- Rabi frequency of the signal field
    cavity_len = 2  #0.72+4e-7  # (m) -- Cavity length
    FSR = scons.c / cavity_len  # (Hz) -- Cavity Free Spectral Range
    I_sat = 83  # (W/m^2 = kg/s^3) -- Saturated Intensity
    trans_wl = 795e-9  # (m) -- Lasing Wavelength
    trans_freq = scons.c / trans_wl  # (Hz) -- Lasing Frequency
    angular_freq = 2 * scons.pi * trans_freq  # (Rad/s) -- Angular Frequency
    laser_temp = 420  # (K) -- Laser Temperature
    reflectivity = 0.99  # Reflectivity of the Cavity Mirror
    Q_factor = (2 * scons.pi * trans_freq * cavity_len) / (scons.c * (1 - reflectivity ** 2))  # Cavity Q-factor
    Asus = 0

    # ------------------------------------------------------------------------------------------------------------------
    # Fields for the DPAL Pump
    # ------------------------------------------------------------------------------------------------------------------
    DPAL = False
    buffer_pressure = 380  # Torr
    P = 10 ** (4.312 - 4040 / laser_temp)
    n = P / (scons.k * laser_temp)
    gamma_d = 2 * np.pi * 20e6 * buffer_pressure  # Decay factor of the DPAL pump
    alpha, beta = 1, 1
    omega_85, omega_87 = 2 * np.pi * 3.034e9, 2 * np.pi * 6.835e9  # Frequency difference between |1> and |2> of Rb 85 & Rb 87
    frac_85, frac_87 = 0.72, 0.28  # Natural fractions of Rubidium 85 and Rubidium 87 respectively
    dpal_ndensity = 10 ** 6  # (atoms/m^2) --  Number density of atoms in the DPAL cell
    dpal_cavity_len = 0.1  # (m) -- Length of the DPAL cell

    # ------------------------------------------------------------------------------------------------------------------
    # Fields for the Raman Pump
    # ------------------------------------------------------------------------------------------------------------------
    raman_ndensity = 1e18  # (atoms/m^2) -- Number density of atoms in the Raman cell
    raman_cavity_len = 0.1  # (m) -- Length of the Raman cell

    # Initialize the DPAL and Raman systems
    system_DPAL85, system_DPAL87, system_raman = None, None, None

    def __init__(self, **kwargs):
        # Parameters of a Raman Gain System using 85Rb
        N = 3  # Modeled as a 3 Level System

        # Decay Rates of the System
        gamma_11 = 0  # (Hz) Decay rate from |1> to |1>
        gamma_12 = 2 * np.pi * 1e6 + 2 * np.pi * 1.5e7  # (Hz) Decay rate from |1> to |2>
        gamma_13 = 0  # (Hz) Decay rate from |1> to |3>

        gamma_21 = 2 * np.pi * 1e6  # (Hz) Decay rate from |2> to |1>
        gamma_22 = 0  # (Hz) Decay rate from |2> to |2>
        gamma_23 = 0  # (Hz) Decay rate from |2> to |3>

        tau_31 = 27.679e-9  # (s) Lifetime of the |3> to |1> transition
        tau_32 = 26.235e-9  # (s) Lifetime of the |3> to |2> transition
        gamma_31 = 1 / (2 * tau_31)  # (Hz) Decay rate from |3> to |1>
        gamma_32 = 1 / (2 * tau_32)  # (Hz) Decay rate from |3> to |2>
        gamma_33 = 0  # (Hz) Decay rate from |3> to |3>

        omega_rp = 2 * np.pi * 2.873e8  # (Hz) Raman pump Rabi frequency
        delta_rp = 2 * np.pi * 1.6e9  # (Hz) Raman pump detuning

        for key, value in kwargs.items():
            match key:
                case 'delta_s':
                    self.delta_s[-1] = value
                case 'omega_s':
                    self.omega_s[-1] = value
                case 'cavity_len':
                    self.cavity_len = value
                case 'laser_temp':
                    self.laser_temp = value
                case 'omega_rp':
                    omega_rp = value
                case 'delta_rp':
                    delta_rp = value

        # Create the Raman system with the relevant parameters
        self.system_raman = System.System(N=N,
                                          omega=[omega_rp, self.omega_s[-1]],
                                          delta=self.delta_s[0],
                                          fixed_deltas=delta_rp,
                                          QSource=[[gamma_11, gamma_21, gamma_31],
                                                   [gamma_12, gamma_22, gamma_32],
                                                   [gamma_13, gamma_23, gamma_33]])

        # Set the Hamiltonian of the Raman system
        self.system_raman.ham = self.ham_3L
        self.Asus = (scons.hbar * scons.c * self.raman_ndensity) * (
            self.system_raman.QSource[0][2]) ** 2 / self.I_sat

        # DPAL System, Not Currently Used (But Here for Future Use if Necessary)
        if self.DPAL:
            self.system_DPAL85 = System.System(N=4,
                                               omega=[(2 * np.pi * 1.516 * (10 ** 9)), self.omega_s[-1]],
                                               fixed_deltas=0,
                                               delta=self.delta_s[0],
                                               QSource=[
                                                   [0, (2 * np.pi * 1e6), 36.1e6 / 2, 38.1e6 / 2],
                                                   [(2 * np.pi * 1.6e6), 0, 36.1e6 / 2, 8.1e6 / 2],
                                                   [0, 0, 0, 5.36e9],
                                                   [0, 0, 4.27e9, 0]],
                                               dephasing=True,
                                               QDephase=[[0, -self.alpha * self.gamma_d, -self.gamma_d, -self.gamma_d],
                                                         [-self.alpha * self.gamma_d, 0, -self.gamma_d, -self.gamma_d],
                                                         [-self.gamma_d, -self.gamma_d, 0, -self.beta * self.gamma_d],
                                                         [-self.gamma_d, -self.gamma_d, -self.beta * self.gamma_d, 0]])
            self.system_DPAL85.ham = self.Ham4L85
            self.system_DPAL87 = System.System(N=4,
                                               omega=[2 * np.pi * 1.516e9, self.omega_s[-1]],
                                               fixed_deltas=0,
                                               delta=self.delta_s[0],
                                               QSource=[[0, 2 * np.pi * 1e6, 36.1e6 / 2, 38.1e6 / 2],
                                                        [(2 * np.pi * 1.6e6), 0, 36.1e6 / 2, 38.1e6 / 2],
                                                        [0, 0, 0, 5.36e9],
                                                        [0, 0, 4.27e9, 0]],
                                               dephasing=True,
                                               QDephase=[[0, -self.alpha * self.gamma_d, -self.gamma_d, -self.gamma_d],
                                                         [-self.alpha * self.gamma_d, 0, -self.gamma_d, -self.gamma_d],
                                                         [-self.gamma_d, -self.gamma_d, 0, -self.beta * self.gamma_d],
                                                         [-self.gamma_d, -self.gamma_d, -self.beta * self.gamma_d, 0]])
            self.system_DPAL87.ham = self.Ham4L87

    # ------------------------------------------------------------------------------------------------------------------
    # Key Algorithm Methods
    # ------------------------------------------------------------------------------------------------------------------

    # Calculate the spectral sensitivity of the laser
    def find_SSF(self, delta_len, **kwargs):
        orginal_len = self.cavity_len

        #self.findOptcavity_len(**kwargs)
        solution_0 = self.find_steady_state(**kwargs)
        nu_0 = solution_0[0]
        empty_cav_omega_0 = self.get_omega_c(solution_0[3])

        self.set_cavity_len(self.cavity_len + delta_len)
        solution_1 = self.find_steady_state(**kwargs, fixed_m=True, mode_guess=solution_0[3])
        nu_1 = solution_1[0]
        empty_cav_omega_1 = self.get_omega_c(solution_1[3])
        self.cavity_len = orginal_len

        S_EC = empty_cav_omega_1 - empty_cav_omega_0
        S_L = nu_1 - nu_0

        return S_EC / S_L

    # Calculate the steady state solution of the laser
    def find_steady_state(self, **kwargs):
        # Will track progress if True
        track_prog = True

        # Code will recurse to find best steady state solution
        max_recursions = 1  # Maximum number of recursions
        recursion = 1  # Current recursion

        fixed_m = False  # If true, code will search for best solution for the given mode
        mode_guess = 905659  # Current mode

        # Upper bound for successful convergence to SS2
        convergence_condition = 4000  # (rad/s)

        num_points = 10000  # Number of points

        # Sets the relevant parameters
        for key, value in kwargs.items():
            match key:
                case 'track_prog':
                    track_prog = value
                case 'recursion':
                    recursion = value
                case 'max_recursions':
                    max_recursions = value
                case 'mode_guess':
                    mode_guess = value
                case 'convergence_condition':
                    convergence_condition = value
                case 'fixed_m':
                    fixed_m = value
                case 'num_points':
                    num_points = num_points

        pairs = self.find_pairs(**kwargs)  # Find the δ_s, Ω_s pairs that satisfy SS1
        pairs_clean = [pair for pair in pairs if not (pair[1] <= 730)]  # Remove the pairs that are not valid solutions
        len_pairs = len(pairs_clean)  # Calculate the number of 'clean' pairs
        possible_sols = []  # Best solution to SS2 for each δ_s, Ω_s pair
        ms = []  # Mode corresponding to each of the solutions

        # Determine δ_s, Ω_s pair that best satisfies SS2
        for i in range(len_pairs):
            # Update the user with relevant information about the progress of the code
            if track_prog:
                if i % round(len_pairs / 4) == 0:
                    print(f"Process of finding steady state solution is {i / len_pairs:.1%} complete!")
                    print(f"Current δ_s, Ω_s pair being checked: {pairs_clean[i]}")

            m = mode_guess  # Set current mode 'm' to mode guess
            pprev = self.checkSS(m, pairs_clean[i])
            m += 1  # Increment mode

            # If the mode is not fixed, use Newton's method to find best mode
            if not fixed_m:
                prev = self.checkSS(m, pairs_clean[i])  # Check SS2 for the mode 'mode_guess + 1' and the δ_s, Ω_s pair

                # If the mode better agrees with SS2, continue increasing m. Otherwise, decrease m
                if pprev - prev >= 0:
                    increment = True
                else:
                    increment = False
                    prev = self.checkSS(m-2, pairs_clean[i])

                # Continue searching until a mode is found that agrees worse with SS2
                while pprev - prev >= 0:
                    if increment:
                        m += 1
                    else:
                        m -= 1
                    pprev = prev
                    prev = self.checkSS(m, pairs_clean[i])

            # Add the solution to the array of possible solutions
            possible_sols += [pprev]
            ms += [m - 1]

        solutionIndx = np.argmin(possible_sols)
        # print(np.where(np.array(possible_sols) < 4000))
        print(
            f"Recursion {recursion} solution: {pairs_clean[solutionIndx] + [ms[solutionIndx]] + [possible_sols[solutionIndx]]}")
        if possible_sols[solutionIndx] > convergence_condition and recursion < max_recursions:
            optDelta = pairs_clean[solutionIndx][0]
            solution = self.find_steady_state(delta_min=optDelta * ((recursion - 0.5) / recursion),
                                              deltaMax=optDelta * (recursion + 0.5) / recursion,
                                              recursion=(recursion + 1),
                                              num_points=num_points)
        else:
            solution = pairs_clean[solutionIndx] + [ms[solutionIndx]] + [possible_sols[solutionIndx]]
        return solution

    # Locate the Rabi Frequency vs. Detuning curve
    def find_curve(self, deltas):
        length = len(deltas)
        if length < 3:
            print("Please use a larger array!")
            return None
        curIndx = int((length - 1) / 2)
        omega = self.get_omega_s(deltas[curIndx])[0]
        indices = [curIndx]
        while omega < 730:
            indxSpace = [-np.ceil(indices[0] / 2), np.ceil(indices[0] / 2)]
            newIndices = []
            for index in indices:
                for space in indxSpace:
                    curIndx = int(index + space)
                    omega = self.get_omega_s(deltas[curIndx])[0]
                    if omega > 730:
                        return curIndx
                    newIndices += [curIndx]

            if newIndices[0] == 0:
                if length % 2 == 0:
                    omega = self.get_omega_s(deltas[-1])[0]
                    if omega > 730:
                        return length - 1
                return None
            indices = newIndices
        return curIndx

    # Calculate the Rabi Frequency & Detuning pairs that satisfy steady state condition 1
    def find_pairs(self, **kwargs):
        # Code will update progress if True
        track_prog = True

        # Range of detunings that will be checked
        delta_min, deltaMax, num_points = -0.15e9, 0.15e9, 2001

        # Maximum number of iterations to find omega_s
        max_its = 10000

        # Change the corresponding parameters
        for key, value in kwargs.items():
            match key:
                case 'delta_min':
                    delta_min = value
                case 'deltaMax':
                    deltaMax = value
                case 'num_points':
                    num_points = value
                case 'track_prog':
                    assert type(value) == bool, "Track progress must be either True or False"
                    track_prog = value
        # Ensures that num_points is an odd number
        if num_points % 2 == 0:
            num_points += 1
            print("Number of points was changed to an odd number!")

        deltas = np.linspace(delta_min, deltaMax, num_points)

        # # Initialize the array that will hold the delta_s and omega_s pairs
        pairs = []
        curve_index = self.find_curve(deltas)

        max_its = max_its/1000
        while curve_index is None:
            inc = (deltaMax - delta_min)/2
            delta_min -= inc
            deltaMax += inc
            deltas = np.linspace(delta_min, deltaMax, num_points)
            curve_index = self.find_curve(deltas)
        max_its = max_its*1000

        print("Progress of Finding Pairs:")
        self.progBar(0.1)

        boundReached = False
        for i in range(num_points - curve_index):
            if boundReached:
                omega, chi = 730, np.nan
            else:
                omega, chi = self.get_omega_s(deltas[curve_index + i], max_its=max_its)
                if omega < 730:
                    boundReached = True
            pairs += [[deltas[curve_index + i], omega, chi]]

        while not boundReached:
            increaseFactor = (deltaMax - delta_min)/2
            curDeltaMax = deltaMax
            deltaMax += increaseFactor
            newDeltas = np.linspace(curDeltaMax + 1, deltaMax, 200)
            for i in range(len(newDeltas)):
                if boundReached:
                    omega, chi = 730, np.nan
                else:
                    omega, chi = self.get_omega_s(newDeltas[i], max_its=max_its)
                    if omega < 730:
                        boundReached = True
                pairs += [[newDeltas[i], omega, chi]]

        # Reverse the order of pairs so the data is sorted
        length = len(pairs)
        tempPairs = []
        for i in range(length):
            tempPairs += [pairs[length - (i + 1)]]
        pairs = tempPairs

        self.progBar(0.5)

        boundReached = False

        for i in range(curve_index):
            if boundReached:
                omega, chi = 730, np.nan
            else:
                omega, chi = self.get_omega_s(deltas[curve_index - (i + 1)], max_its=max_its)
                if omega < 730:
                    boundReached = True
            pairs += [[deltas[curve_index - (i + 1)], omega, chi]]

        while not boundReached:
            increaseFactor = (deltaMax - delta_min)/2
            current_delta_min = delta_min
            delta_min -= increaseFactor
            newDeltas = np.linspace(delta_min, current_delta_min, 200)
            for i in range(len(newDeltas)):
                if boundReached:
                    omega, chi = 730, np.nan
                else:
                    omega, chi = self.get_omega_s(newDeltas[-(i+1)], max_its=max_its)
                    if omega < 730:
                        boundReached = True
                pairs += [[newDeltas[-(i+1)], omega, chi]]

        self.progBar(1)

        # ------------------------------------------------------------------
        # Slower algorithm that manually calculates omega_s for each delta_s
        # ------------------------------------------------------------------
        # for i in range(num_points):
        #     # Informs the user of the progress of the code
        #     if track_prog:
        #         if i % round(num_points / 4) == 0:
        #             print(f"Process of finding pairs is {i / num_points:.1%} complete! ")
        #     omega, chi = self.get_omega_s(deltas[i], max_its=max_its)
        #     pairs += [[deltas[i], omega, chi]]
        return pairs

    # ------------------------------------------------------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Part of Algorithm
    # ------------------------------------------------------------------------------------------------------------------

    # Calculate the Rabi Frequency that satisfies steady state condition 1 for the given detuning
    def get_omega_s(self, delta_s, **kwargs):
        # Save the starting value of omega_s
        omega_s0 = self.omega_s[-1]

        max_its = 100000
        for key, value in kwargs.items():
            match key:
                case 'max_its':
                    max_its = value

        self.set_delta(delta_s)
        chi = self.get_chi()
        err = abs(-chi.imag - 1 / self.Q_factor)
        curIt = 0

        stepSign = 1
        damping = 1

        while err > 1e-5 * (1 / self.Q_factor) and not (curIt > max_its):
            if -chi.imag < 1 / self.Q_factor:
                stepSign = -1
                damping = 0.5 * damping

            if -chi.imag > 1 / self.Q_factor:
                stepSign = 1

            stepSize = 100 * abs(abs(self.Q_factor * chi.imag * self.Asus) - self.omega_s[-1])
            if stepSize > abs(self.omega_s[-1]):
                stepSize = 0.3 * self.omega_s[-1]

            self.set_omega(self.omega_s[-1] + stepSign * damping * stepSize)
            chi = self.get_chi()
            err = abs(-chi.imag - 1 / self.Q_factor)
            curIt += 1

        optOmega = self.omega_s[-1]
        self.omega_s[-1] = omega_s0

        return optOmega, chi

    #  Get effective susceptibility
    def get_chi(self):
        chiRam = self.get_chi_raman()
        if self.DPAL:
            chiDPAL = self.get_chi_DPAL()
        else:
            chiDPAL = 0

        chiEff = (self.dpal_cavity_len / self.cavity_len) * chiDPAL + (
                self.raman_cavity_len / self.cavity_len) * chiRam
        return chiEff

    # Get susceptibility of the Raman cell
    def get_chi_raman(self):
        rhoRam = self.system_raman.get_rho()
        chiRam = -(rhoRam[2][0] * self.Asus) / self.omega_s[-1]
        # print(chiRam)
        return chiRam

    # Get susceptibility of the DPAL cell
    def get_chi_DPAL(self):
        # Get the density matrices for the three systems
        rho85 = self.system_DPAL85.get_rho()
        rho87 = self.system_DPAL87.get_rho()

        # Calculate the susceptibilities
        chi85 = (rho85[2][0] + rho85[2][1]) * (scons.hbar * scons.c * self.dpal_ndensity) / (
                self.I_sat * self.omega_s[-1]) * (self.system_DPAL85.QSource[0][2]) ** 2
        chi87 = (rho87[2][0] + rho87[2][1]) * (scons.hbar * scons.c * self.dpal_ndensity) / (
                self.I_sat * self.omega_s[-1]) * (self.system_DPAL87.QSource[0][2]) ** 2

        # Calculate the effective susceptibility
        assert self.frac_85 + self.frac_87 == 1, "The fraction of Rb85 plus the fraction of Rb87 should equal unity!"

        chiDPAL = self.frac_85 * chi85 + self.frac_87 * chi87
        return chiDPAL

    # Check if a Rabi Frequency, Detuning pair satisfies steady state condition 2
    def checkSS(self, m, pair):
        deltaRP = self.system_raman.fixed_deltas
        return abs((pair[2].real / 2 + 1) * (2 * scons.pi * self.trans_freq + pair[0]) - self.get_omega_c(m))
        #Took out - deltaRP
    # Calculate the empty cavity resonance frequency
    def get_omega_c(self, m):
        return (2 * scons.pi * m * scons.c) / self.cavity_len

    # Change the laser's signal field detuning
    def set_delta(self, delta_s):
        self.delta_s[-1] = delta_s
        self.system_raman.delta = delta_s
        if self.DPAL:
            self.system_DPAL85.delta = delta_s
            self.system_DPAL87.delta = delta_s

    # Change the laser's signal field Rabi Frequency
    def set_omega(self, omega_s):
        self.omega_s[-1] = omega_s
        self.system_raman.omega[1] = omega_s
        if self.DPAL:
            self.system_DPAL87.omega[-1] = omega_s
            self.system_DPAL85.omega[-1] = omega_s

    # Change the laser's cavity length
    def set_cavity_len(self, cavity_len):
        self.cavity_len = cavity_len
        self.Q_factor = (2 * scons.pi * self.trans_freq * cavity_len) / (scons.c * (1 - self.reflectivity ** 2))

    # ------------------------------------------------------------------------------------------------------------------
    # Other
    # ------------------------------------------------------------------------------------------------------------------

    # Implements a progress bar
    def progBar(self, percent):
        numChars = 50
        numDone = int(percent * numChars)
        print("[", end="")
        for i in range(numDone):
            print("/", end="")
        for i in range(numChars - numDone):
            print("-", end="")
        print("]")

    def findOptcavity_len(self, **kwargs):
        sol = self.find_steady_state(**kwargs)
        deltaRP = self.system_raman.fixed_deltas
        cavity_len = (2 * scons.pi * sol[3] * scons.c) / (
                (sol[2].real / 2 + 1) * (2 * scons.pi * self.trans_freq - deltaRP + sol[0]))
        self.set_cavity_len(cavity_len)
        return sol

    # ------------
    # Hamiltonians
    # ------------

    # Hamiltonian for the Raman Pump
    def ham_3L(self, delta_s, omega_s):
        gammas = self.system_raman.QSource

        # Relevant Decay Factors
        gamma_12 = gammas[1][0]
        gamma_21 = gammas[0][1]
        gamma_31 = gammas[0][2]

        omega_rp = self.system_raman.omega[0]
        delta_rp = self.system_raman.fixed_deltas
        ham = 0.5 * np.array([[-1j * gamma_12, 0, omega_s],
                              [0, -2 * delta_s + 2 * delta_rp - 1j * gamma_21, omega_rp],
                              [omega_s, omega_rp, -2 * (delta_s + 1j * gamma_31)]])
        return ham

    # Hamiltonian for the DPAL Pump
    def Ham4L85(self, delta_s):
        gammas = self.system_DPAL85.QSource
        omega_p = self.system_DPAL85.omega[0]
        return 0.5 * np.array([[-1j * gammas[1][0], 0, self.omega_s[-1], omega_p],
                               [0, 2 * self.omega_85 - 1j * gammas[0][1], self.omega_s[-1], omega_p],
                               [self.omega_s[-1], self.omega_s[-1],
                                -2 * delta_s - 1j * (2 * gammas[0][2] + gammas[3][2]), 0],
                               [omega_p, omega_p, 0,
                                -2 * self.system_DPAL85.fixed_deltas - 1j * (2 * gammas[0][3] + gammas[2][3])]])

    # Hamiltonian for the DPAL Pump
    def Ham4L87(self, delta_s):
        gammas = self.system_DPAL87.QSource
        omega_p = self.system_DPAL87.omega[0]
        return 0.5 * np.array([[-1j * gammas[1][0], 0, self.omega_s[-1], omega_p],
                               [0, 2 * self.omega_87 - 1j * gammas[0][1], self.omega_s[-1], omega_p],
                               [self.omega_s[-1], self.omega_s[-1],
                                -2 * delta_s - 1j * (2 * gammas[0][2] + gammas[3][2]), 0],
                               [omega_p, omega_p, 0,
                                -2 * self.system_DPAL87.fixed_deltas - 1j * (2 * gammas[0][3] + gammas[2][3])]])
