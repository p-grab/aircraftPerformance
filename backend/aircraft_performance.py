import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

power_nom = 6000
Kp = 0.5


def calc_cz_cx(a, b, c):
    czcx = pd.DataFrame({})
    czcx['cz'] = np.arange(0.1, 1.6, 0.02)
    czcx['cx'] = a * czcx['cz'] * czcx['cz'] + b * czcx['cz'] + c
    czcx['E'] = czcx['cz']**3 / czcx['cx'] / czcx['cx']
    return czcx


def calc_w_and_gamma(czcx, h, m, S, g, engine_type):
    czcx['rho'] = 1.225*(1 - h / 44.3)**5.256
    czcx['V'] = (2 * m * g / czcx['rho'] / S / czcx['cz'])**0.5

    if engine_type == 'prop':
        czcx['Nr'] = 1000 * (7E-07*h * 1000 * h * 1000 - 0.0257 * h * 1000 + 231.38)

    if engine_type == 'turbo':
        power_0 = power_nom * (czcx['rho']/1.225)**0.85
        power_min = power_0 * (1 - Kp * 1)
        a = 340.3 * math.sqrt((288.15-6.5*h)/288.15)
        czcx['a'] = czcx['V'] / a
        czcx['Nr'] = (power_0 - czcx['a'] * (power_0 - power_min)) * czcx['V']

    czcx['Nn'] = (m * g * (2 / czcx['rho'] / S * m * g / czcx['E'])**0.5)
    czcx['delta_N'] = czcx['Nr'] - czcx['Nn']
    czcx['w'] = czcx['delta_N'] / m / g
    czcx['gamma'] = np.arcsin(czcx['w'] / czcx['V'])
    czcx['gamma_st'] = czcx['gamma'] * 180 / np.pi
    return czcx


def remove_inf_values(czcx, schedule):
    with pd.option_context('mode.use_inf_as_na', True):
        schedule.dropna(inplace=True)
        czcx.dropna(inplace=True)


def linear_approx_zero(x1, x2, y1, y2):
    a = (y1 - y2) / (x1 - x2)
    b = y1 - x1 * a
    return - b / a


def calc_max_velocity(schedule, h):
    max_v = 0
    index = -1
    max_i = 0
    for (data, w) in zip(schedule['V_'+str(h)], schedule['w_'+str(h)]):
        index += 1
        if max_v <= data and w >= 0:
            max_v = data
            max_i = index
    if max_v:
        return linear_approx_zero(schedule['V_'+str(h)][max_i], schedule['V_'+str(h)][max_i + 1], schedule['w_'+str(h)][max_i], schedule['w_'+str(h)][max_i + 1])
    return 0


def calc_min_V(schedule, h):
    min_v = 100000
    for (data, w) in zip(schedule['V_'+str(h)], schedule['w_'+str(h)]):
        if min_v >= data and w >= 0:
            min_v = data
    if min_v < 100000:
        return min_v
    return 0


def calc_max_w(schedule, h):
    return schedule['w_'+str(h)].max()


def calc_max_gamma(schedule, h):
    return schedule['gamma_st'+str(h)].max()


def calc_V_w(schedule, h):
    return schedule['V_'+str(h)][schedule['w_'+str(h)].idxmax()]


def calc_V_gamma(schedule, h):
    return schedule['V_'+str(h)][schedule['gamma_st'+str(h)].idxmax()]


def calc_theoretical(w_max_list):
    theo = []
    for w in w_max_list:
        if w > 0:
            theo.append(w)
    return len(theo)


def calc_practical(w_max_list):
    prac = []
    for w in w_max_list:
        if w > 0.5:
            prac.append(w)
    return len(prac)


def calc_time_min(theoretical, w_max_list, h, step):
    t_min_list = []
    h_max = int(h / step)
    for h in range(0, h_max):
        if h * step != theoretical:
            t_min_list.append(theoretical * 1000 / w_max_list[0] * math.log(1 / (1 - h * step / theoretical))/60)
    return t_min_list


# inicial schedules:
schedule = pd.DataFrame({})
heights = pd.DataFrame({})
powers = pd.DataFrame({})
V_min_list = []
V_max_list = []
V_w_list = []
V_gamma_list = []
gamma_max_list = []
w_max_list = []
t_min_list = []


# INPUT DATA1:
m = 2200
S = 17.02
g = 9.81
# Cz(Cx)
a = 0.0365
b = -0.0011
c = 0.0265
engine_type = "prop"
step = 0.02

# # INPUT DATA2:
# m = 1800
# S = 9.6
# g = 9.81
# # Cz(Cx)
# a = 0.0478
# b = -0.0006
# c = 0.0235
# engine_type = "turbo"
# step = 0.02


h = 0.0
while (h < 20):
    czcx = calc_cz_cx(a, b, c)
    czcx = calc_w_and_gamma(czcx, h, m, S, g, engine_type)

    schedule['V_'+str(h)] = czcx['V']
    schedule['w_'+str(h)] = czcx['w']
    schedule['gamma_st'+str(h)] = czcx['gamma_st']

    remove_inf_values(czcx, schedule)
    V_max_list.append(calc_max_velocity(schedule, h) if calc_max_velocity(schedule, h) else None)
    V_min_list.append(calc_min_V(schedule, h) if calc_min_V(schedule, h) else None)

    w_max_list.append(calc_max_w(schedule, h))
    gamma_max_list.append(calc_max_gamma(schedule, h))
    V_w_list.append(calc_V_w(schedule, h))
    V_gamma_list.append(calc_V_gamma(schedule, h))
    if calc_max_w(schedule, h) < 0:
        break
    h += step

theoretical = calc_theoretical(w_max_list) * step
practical = calc_practical(w_max_list) * step
t_min_list = calc_time_min(theoretical, w_max_list, h, step)


heights['h'] = np.arange(0, theoretical + step, step)
fig1, ax1 = plt.subplots(nrows=3, ncols=1, figsize=(6, 7))
fig1.suptitle("Wykres Ofertowy", fontsize=22)

V_col = schedule.columns[:: int(3 / step)]
w_col = schedule.columns[1::int(3 / step)]
gamma_col = schedule.columns[2::int(3 / step)]
for v, w, gamma in zip(V_col, w_col, gamma_col):
    ax1[0].plot(schedule[v], schedule[w], label=f"h = {round(float(str(v)[2:]), 2)}")
    ax1[1].plot(schedule[v], schedule[gamma], label=f"h = {round(float(str(v)[2:]), 2)}")

ax1[2].plot(V_max_list, heights['h'], label='V_max')
ax1[2].plot(V_min_list, heights['h'], label='V_min')
ax1[2].plot(w_max_list, heights['h'], label='w_max')
ax1[2].plot(gamma_max_list, heights['h'], label='gamma_max')
ax1[2].plot(V_w_list, heights['h'], label='V_w')
ax1[2].plot(V_gamma_list, heights['h'], label='V_gamma')
try:
    ax1[2].plot(t_min_list, heights['h'][0:-2], label='t_h (min)')
except:
    ax1[2].plot(t_min_list, heights['h'][0:-1], label='t_h (min)')
ax1[2].axhline(y=theoretical, color='r', linestyle='--', label='theoretical')
ax1[2].axhline(y=practical, color='b', linestyle=':', label='practical')

ax1[0].set_xlabel("Velocity [m/s]")
ax1[1].set_xlabel("Velocity [m/s]")
ax1[2].set_xlabel("V [m/s], w [m/s], γ [deg], t [min], ")
ax1[0].set_ylabel("w [m/s]")
ax1[1].set_ylabel("gamma [deg]")
ax1[2].set_ylabel("h [km]")

ax1[0].legend(loc='upper center', bbox_to_anchor=(1.04, 1), fancybox=True, shadow=True)
ax1[1].legend(loc='upper center', bbox_to_anchor=(1.04, 1), fancybox=True, shadow=True)
ax1[2].legend(loc='upper center', bbox_to_anchor=(1.05, 1), fancybox=True, shadow=True)

plt.show()
