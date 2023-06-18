import numpy as np
import matplotlib.pyplot as plt
import json

from pVQD import *
from exactsimulation import *

#physic sistem configuration

n_spins = 3
depth = 3
magnetic_fields = [1/4, 1]


#algorithm configuration
s = np.pi/2
learning_rate = 1
time_step = 0.05
n_time_steps = 40
sim_config = ['global', 'shots']
shots = 8000


num_parameters = ((depth+1)*n_spins+depth*(n_spins-1))
init_parameters = np.zeros(num_parameters)
init_shift = np.ones(num_parameters)*0.01



#running the algorithm
algo = pVQD(n_spins,init_parameters, init_shift, depth, magnetic_fields)
data = algo.run(time_step, n_time_steps, sim_config, shots, learning_rate, s)

'''

#simulating exact evolution
inter = [i*time_step for i in range(n_time_steps+1)]
exact_log = {"exact_x" : [], "exact_z": [],'n_Time_Steps': n_time_steps, 'time_step': time_step}
exact_log["exact_x"] = exact_meas_of_X (n_spins, magnetic_fields[0], magnetic_fields[1], inter)
exact_log["exact_z"] = exact_meas_of_Z (n_spins, magnetic_fields[0], magnetic_fields[1], inter)


with open("data/exact_data_"+str(n_time_steps)+"_steps.dat", "w") as f:
    json.dump(exact_log, f)


'''


'''
#---------------------------------------------------
#plotting result
plt.figure(1)
plt.subplot(211)
plt.plot(inter, exact_x,linestyle="dashed",color="black",label="Exact")
plt.errorbar(inter, par_results['X'], yerr = errors['X'], color="C1",label="pVQD",linestyle="",marker=".")

plt.ylabel(r'$\langle\sigma_x\rangle$')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(inter, exact_z,linestyle="dashed",color="black",label="Exact")
plt.errorbar(inter, par_results['Z'], yerr = errors['Z'],color="C1",label="pVQD",linestyle="",marker=".")

plt.xlabel('time')
plt.ylabel(r'$\langle\sigma_z\rangle$')

plt.legend()
plt.grid()
plt.show()
'''
