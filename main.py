import datetime
import csv

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as scons

import SubluminalLaser
import System
import NLevelTest


def popPlot(system, **kwargs):
    '''Plot the density matrix elements of any N level system
    :param System.System system: The System object
    :param callable() ham: The Hamiltonian of the system
    :keyword int numPoints: Number of points to plot
    '''

    # Ensure that a valid System object is passed as a parameter and that the Hamiltonian is a function
    assert isinstance(system, System.System), "Must enter a valid System object!"

    # --------------------------------------
    # Define the default plotting parameters
    # --------------------------------------

    # Which densities to plot and colors
    levels = np.arange(1, system.N+1)
    colors = system.N*["green"]

    # Range of deltas and number of points
    deltaMin = -100  # Lower Bound of x
    deltaMax = 100  # Upper Bound of x
    numPoints = 1000  # Number of points to plot

    # Labels
    xlabel = "δ/Γ"  # X Axis Label
    title = f"Density Matrix Elements for a {system.N} Level System"  # Plot Title

    # Check kwargs and change parameters accordingly
    for key, value in kwargs.items():
        match key:
            case 'numPoints':
                numPoints = value
            case 'deltaMin':
                deltaMin = value
            case 'deltaMax':
                deltaMax = value
            case 'title':
                title = value
            case 'xlabel':
                xlabel = value
            case 'levels':
                if type(value) == int:
                    assert 0 < value <= system.N, "Level must be between 1 and N"
                    levels = np.array([value])
                else:
                    assert max(value) <= system.N and min(value) > 0, "Levels must be between 1 and N"
                    levels = np.array(value)
            case 'color':
                colors = system.N * [value]
            case 'colors':
                colors = value

    # Ensure that all the entered parameters are consistent with requirements
    assert len(colors) >= len(levels), "Not enough colors given!"

    # Define the range of deltas to be considered
    deltas = np.linspace(deltaMin, deltaMax, numPoints)

    # Save the original detuning of the system
    orgDelta = system.delta

    # Initialize the matrix that will hold the relevant density matrix elements
    rhos = np.zeros((len(levels), len(deltas)))

    # Calculate the density matrix elements for each delta
    for i in range(len(deltas)):
        system.delta = deltas[i]
        output = system.get_rho()
        for j in range(len(levels)):
            rhos[j][i] = output[levels[j]-1][levels[j]-1].real

    # Reset the detuning
    system.delta = orgDelta

    # Create subplots
    fig, axs = plt.subplots(len(levels), 1, sharex='all')

    # Change axs to an array if it is not one
    if not isinstance(axs, np.ndarray):
        axs = [axs]

    # Plot the desired density matrix curves
    for i in range(len(levels)):
        axs[i].plot(deltas, rhos[i], c=colors[i])
        axs[i].set_ylabel(f"$ρ_{{{levels[i]*11}}}$")

    # Add labels to the plot
    fig.supxlabel(xlabel)
    fig.suptitle(title)

    # Save the figure
    curDateTime = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    plt.savefig(f"DMElems-{system.N}_Levels_{curDateTime}.jpeg")
    plt.show()

def susDeltaPlot(system, **kwargs):
    # Ensure that a valid System object is passed as a parameter and that the Hamiltonian is a function
    assert isinstance(system, System.System), "Must enter a valid System object!"

    # Define default plotting parameters
    deltaMin = -10**10
    deltaMax = 10**10
    numPoints = 1000

    for key, value in kwargs.items():
        match key:
            case 'deltaMin':
                deltaMin = value
            case 'deltaMax':
                deltaMax = value
            case 'numPoints':
                numPoints = value

# Density Matrix Plotting
if __name__ != '__main__':
    # Two Level System Plot
    # system2L = System.System()
    # popPlot(system2L, levels=[1, 2], deltaMin=-50, deltaMax=50, numPoints=200, colors=["cyan", "green"])
    #
    # #Three Level System Plot
    # system3L = System.System(N=3, omega=[1, 1], comDelt=[0], QSource=[[0, 0, 0.5], [0, 0, 0.5], [0, 0, 0]])
    # system3L.ham = system3L.Ham3L
    # popPlot(system3L, levels=[1, 2, 3], colors=['cyan', 'blue', 'green'])

    myLaser = SubluminalLaser.SubluminalLaser(omega_s=3e6)
    sysRam = myLaser.system_raman
    # sysDpal5 = laser.systemDpal85
    # sysDpal7 = laser.systemDpal87
    plt.rcParams["font.family"] = "Cambria"
    plt.rcParams["font.size"] = 11
    detuningP = 2 * np.pi * 1.6 * 10 ** 9
    myLaser.system_raman.comDelt = detuningP
    popPlot(sysRam, deltaMin=detuningP-100*myLaser.system_raman.QSource[1][2], deltaMax=100*myLaser.system_raman.QSource[1][2] + detuningP, numPoints=1000, levels=[1, 2, 3], colors=[(242/255, 116/255, 5/255), (242/255, 116/255, 5/255), (115/255, 23/255, 2/255)], xlabel="Laser Detuning $(\delta_{s})\ [Rad\ s^{-1}]$", title="Density Matrix Elements of the Raman Cell (Ωs = 3e6 Rad/s)")
    # popPlot(sysDpal5, deltaMin=-10**9, deltaMax=10**9, numPoints=1000, xlabel="$\delta_{s}$", color='red', levels=3)
    # popPlot(sysDpal7, deltaMin=-10**9, deltaMax=10**9, numPoints=1000, xlabel="$\delta_{s}$", color='blue', levels=3)

# Susceptibility Plotting
if __name__ != '__main__':
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=[8, 5], sharex='all')
    fig.suptitle("Real and Imaginary Susceptibility of Laser Cavity (Ωs = 3e6 Rad/s)")
    laser = SubluminalLaser.SubluminalLaser(omega_s=3*10**6)
    #laser.system_raman.omega[0] = 3*10**8
    detuningP = 2 * np.pi * 1.6 * 10 ** 9
    laser.system_raman.comDelt = detuningP
    deltas = np.arange(detuningP-100*laser.system_raman.QSource[1][2], 100*laser.system_raman.QSource[1][2] + detuningP, 10**7)
    invQ = [1/laser.Q_factor] * len(deltas)
    chis = np.zeros(len(deltas))
    chis2 = np.zeros(len(deltas))
    plt.rcParams["font.family"] = "Cambria"
    plt.rcParams["font.size"] = 12


    for i in range(len(deltas)):
        laser.set_delta(deltas[i])
        chi = laser.get_chi()
        chis[i] = chi.real
        chis2[i] = -chi.imag

    ax1.plot(deltas, chis, c=(242/255, 116/255, 5/255))
    ax2.plot(deltas, chis2, c=(115/255, 23/255, 2/255))
    #ax2.set_ylim()
    #ax2.set_yscale('log')
    ax2.set_xlabel("Laser Detuning $(\delta_{s})\ [Rad\ s^{-1}]$")
    ax1.set_ylabel("Real Susceptibility $(\chi')$")
    ax2.set_ylabel("Imaginary Susceptibility $(-\chi'')$")
    # Save the figure
    curDateTime = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    plt.savefig(f"Susceptibility_{curDateTime}.jpeg")
    plt.show()

# δ_s, ω_s Pair Plotting
if __name__ != '__main__':
    detuningP = 2 * np.pi * 1.6 * 10 ** 9
    myLaser = SubluminalLaser.SubluminalLaser(omega_s=1000)
    myLaser.system_raman.comDelt = detuningP
    output = myLaser.find_pairs(deltaMin=detuningP-100*myLaser.system_raman.QSource[1][2], deltaMax=100*myLaser.system_raman.QSource[1][2] + detuningP, numPoints=100)
    x = []
    yIm = []
    yRe = []
    z = []
    for pair in output:
        x += [pair[0]]
        yIm += [-pair[2].imag]
        yRe += [pair[2].real]
        z += [pair[1]]
        #print(pair)
    plt.rcParams["font.family"] = "Cambria"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=[8, 5])
    ax.plot(x, yIm, c=(242/255, 116/255, 5/255))
    ax.set_xlabel("Laser Detuning $(\delta_{s})\ [Rad\ s^{-1}]$")
    ax.set_ylabel("Rabi Frequency of Laser $(\omega_s)\ [Rad\ s^{-1}]$")
    curDateTime = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    plt.savefig(f"PairsSus_{curDateTime}.jpeg")
    plt.show()

    fig, ax = plt.subplots(figsize=[8, 5])
    fig.suptitle("Solutions to the Amplitude Single-mode Laser Equation")
    ax.plot(x, z, c=(242 / 255, 116 / 255, 5 / 255))
    ax.set_xlabel("Laser Detuning $(\delta_{s})\ [Rad\ s^{-1}]$")
    ax.set_ylabel("Rabi Frequency of Laser $(\Omega_s)\ [Rad\ s^{-1}]$")
    curDateTime = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    plt.savefig(f"Pairs_{curDateTime}.jpeg")
    plt.show()

# SSF Calculation
if __name__ == '__main__':
    #detuningP = 2 * np.pi * 1.6e9
    myLaser = SubluminalLaser.SubluminalLaser(omega_s=1000)
    #myLaser.system_raman.fixed_deltas = detuningP

    # #solution = myLaser.find_steady_state(deltaMin=detuningP-100*myLaser.system_raman.QSource[1][2], deltaMax=100*myLaser.system_raman.QSource[1][2] + detuningP, numPoints=10000)
    # deltaRPs = np.linspace(2*np.pi*0.1e9, 2*np.pi*1.6e9, 11)
    # solutions = []
    # for delta in deltaRPs:
    #     myLaser.system_raman.fixed_deltas = delta
    #     solution = myLaser.find_SSF(1e-9, deltaMin=delta - 100 * myLaser.system_raman.QSource[1][2],
    #                                 deltaMax=100 * myLaser.system_raman.QSource[1][2] + delta, numPoints=5000)
    #
    #     print(solution)
    #     solutions += [solution]
    # fig, ax = plt.subplots(figsize=[8, 5])
    # ax.plot(deltaRPs, solutions)
    # curDateTime = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    # plt.savefig(f"SSFs_{curDateTime}.jpeg")
    # plt.show()
    detuningP = 2 * np.pi * 1.6e9
    solution = myLaser.find_SSF(5e-7, deltaMin=detuningP - 100 * myLaser.system_raman.QSource[1][2],
                                deltaMax=100 * myLaser.system_raman.QSource[1][2] + detuningP, numPoints=5000)
    print(solution)

# Plotting 1st Laser Equation Condition vs. δ_s
if __name__ != '__main__':
    myLaser = SubluminalLaser.SubluminalLaser(omega_s=10**8)
    delta_s = np.linspace(-0.15*10**9, 0, 1000)
    dif = []
    for delta in delta_s:
        dif += [2 * scons.pi * myLaser.trans_freq - (2 * np.pi * 1.6 * 10 ** 9) + delta - myLaser.get_omega_c(905657)]
    fig, ax = plt.subplots(figsize=[8, 5])
    ax.plot(delta_s, dif, c='blue')
    ax.set_xlabel("$\delta_s$")
    ax.set_ylabel("$Dif$")
    plt.show()

# Testing Random Stuffs
if __name__ != '__main__':
    myLaser = SubluminalLaser.SubluminalLaser(omega_s=10**10)
    print(myLaser.find_steady_state(deltaMin=-0.5*10**(10), deltaMax=0.5*10**(10)))

# Testing if Hamiltonian is modified
if __name__ != '__main__':
    myLaser = SubluminalLaser.SubluminalLaser(omega_s=10 ** 10)
    myLaser.system_raman.comDelt = -2 * np.pi * 1.6 * 10 ** 9

    # Parameters for which detunings will be checked
    deltaMin, deltaMax, numPoints = -0.5 * 10 ** 9, 0.5 * 10 ** 9, 2001
    deltas = np.linspace(deltaMin, deltaMax, numPoints)

    f = open('DensityMatrixElements.csv', 'w')
    writer = csv.writer(f)
    rhos = np.zeros(len(deltas))
    for i in range(numPoints):
        myLaser.system_raman.delta = deltas[i]
        rho13 = myLaser.system_raman.get_rho()[2][0]
        rhos[i] = [rho13.real, rho13.imag]
        # writer.writerow([rho13.real, rho13.imag])

if __name__ != '__main__':
    NLevelTest.test()







