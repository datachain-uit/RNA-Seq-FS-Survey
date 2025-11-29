def compute_energy(cpu_util, wall_time, cpu_tdp=26.36):
    return (cpu_util / 100.0) * cpu_tdp * wall_time

def compute_edp(energy, wall_time):
    return energy * wall_time

def compute_carbon(energy_joule, ef=1.32e-4):
    return energy_joule * ef
