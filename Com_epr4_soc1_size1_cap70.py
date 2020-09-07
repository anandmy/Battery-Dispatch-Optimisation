

import numpy as np
import pandas as pd
from math import *
import ipdb
from pyomo.environ import *
import time

st = time.time()
#Change file path accordingly
load_df = pd.read_csv('Downloads\99520\Commercial.csv', header=None)
#Dataset taken from OpenEI, "https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/"
load_df.drop(load_df.columns[[0,2,3,4,5,6,7,8,9,10]], axis = 1, inplace = True)
load_df.drop(labels=0, axis=0, inplace=True)
load_df.head()

print((load_df).shape)

load = np.array(load_df)
load = np.resize(load, (365, 24))
load = load.astype('float64')
#Prices for ConEd NYC Commercial
on_peak = 0.3155 # 0.2197 for Residential
off_peak = 0.0115

"""
#Texas Commercial Rates (use for the load and PV dataset)
Peak hours 1 PM to 7 PM only from June to September (day 152 to day 273)
on_peak = 0.102253 # 0.183415 for Residential
off_peak = 0.005975 # 0.056101 for Residential
"""

#Create price matrix
prices = np.zeros((365, 24))

for i in range(0, 365):
    for j in range(0, 24):
        if j in range(7, 23):
            prices[i][j] = float(on_peak)
        else:
            prices[i][j] = float(off_peak)

print(prices[0][1], prices[0][7], prices[0][22], prices[0][23])

#pyomo modelling starts
model = ConcreteModel()

model.discount = 0.07 #Discount Factor


#Change battery specifications here
osize = 1 #Oversizing or number of battery packs
rated_energy = 11.6 * osize
rem_energy = rated_energy * 0.8 #Energy remaining at the beginning of second-life use
model.max_energy = rem_energy #Initialize
model.degradation_limit = 0.2 * rated_energy
epr = 4 #Energy to Power Ratio. Determines peak power wrt to SoH or energy capacity on any given day

model.eff = 0.95 #One way charging/dischargin efficiency
calendar_deg = 0.01

DB = [] #stores daily benefits/revenue
daily_degrad = [] #stores daily degradation
daily_degrad.append(0)
daily_throughput = [] #stores daily throughput

YB = [] #stores yearly benefits/revenue
Y_degrad = [] #stores yearly degradation
Y_tp = [] #stores yearly throughput
Y_OM_cost = []
Y_Deg_cost = []
om_cost = 27 #O&M Cost associated with per unit power usage per day # divided by 365 to change from per year to per day. Value taken from EPRI Handbook for residential scale storage.


LB = [] #stores long-term or total benefits/revenue

n_years = 15

for n in range(0, n_years): #or range(1, n_years+1)
    
    #if (sum(daily_degrad) <= model.degradation_limit): #Use if simulation needs to be stopped when degradation criteria is met
    print("Entering year:", n)
    for model.day in range(1, 366):

        model.max_energy = model.max_energy - daily_degrad[-1]
        model.soc10 = 0.1 * model.max_energy
        model.soc90 = 0.9 * model.max_energy
        model.max_power = model.max_energy / epr #Maintaining a energy-to-power ratio of 4

        model.horizon = 24
        model.H = RangeSet(0, model.horizon - 1)
                        
        model.price = prices[model.day - 1, :]
        model.load = load[model.day - 1, :]

        model.pd = Var(model.H, domain = NonNegativeReals, bounds = (0, model.max_power))
        model.pc = Var(model.H, domain = NonNegativeReals, bounds = (0, model.max_power))
        model.e = Var(model.H, domain = NonNegativeReals, bounds = (model.soc10, model.soc90))
        model.p = Var(model.H, domain = Reals)
        model.tp = Var(domain = Reals)

        model.grid = Var(model.H, domain = Reals)

        def power(model, i):
            return model.p[i] == model.pc[i] - model.pd[i]
        model.power_cons = Constraint(model.H, rule = power)

        def grid_power(model, i):
            return model.p[i] == model.grid[i] - model.load[i]
        model.grid_cons = Constraint(model.H, rule = grid_power)
        
        """
        def energy_balance(model, i):
            return sum(model.grid[i] - model.p[i] for i in model.H) == sum(model.load[i] for i in model.H)
        model.energy_cons = Constraint(model.H, rule = energy_balance)
        """

        def throughput(model, i):
            return model.tp == sum(abs(model.p[i]) for i in model.H)
        model.energy_throughput = Constraint(model.H, rule = throughput)
            
        def soc_update(model, i):
            if i == 0:
                return model.e[i] == model.e[model.horizon - 1] + model.pc[model.horizon - 1] * model.eff - model.pd[model.horizon - 1] / model.eff
            else:
                return model.e[i] == model.e[i - 1] + model.pc[i - 1] * model.eff - model.pd[i - 1] / model.eff
        model.soc = Constraint(model.H, rule = soc_update)

        def obj_fn(model):
            return (sum(model.price[i] * model.load[i] for i in model.H) - sum(model.price[i] * model.grid[i] for i in model.H))
        model.obj = Objective(rule = obj_fn, sense = maximize)
            
        io_options = dict()
        opt = SolverFactory('gams')
        io_options['solver'] = 'cplex'

        results = opt.solve(model)
        problem = results.Problem()
        sol = problem['Lower bound']

        DB.append(sol) 
        daily_throughput.append(value(model.tp))
        daily_degrad.append(2.71e-5 * value(model.tp) + value(model.max_energy) * calendar_deg / 365) #2.71e-5 taken from Scott Peterson paper
        
    YB.append(sum(DB) / ((1 + model.discount) ** n))
    Y_degrad.append(sum(daily_degrad))
    Y_tp.append(sum(daily_throughput))
    print("Successfully executed year:", n+1)
    


LB.append(sum(YB))
print("Number of years completed:", len(YB))


for i in range(1, len(YB)):
    YB[-i] = YB[-i] - YB[-(i + 1)]
    Y_degrad[-i] = Y_degrad[-i] - Y_degrad[-(i + 1)]
    Y_tp[-i] = Y_tp[-i] - Y_tp[-(i + 1)]


#LCOD can be avoided since degradation is updated at the end of each day.
df = 0.07 
CAPEX = 70 # in $/kWh #Need a better estimate for CAPEX
LCOD = ((sum(daily_degrad) / rated_energy) * CAPEX * rated_energy) / sum(Y_degrad[i] * ((1 + df) ** i) for i in range(0, len(YB)))
Y_profit = []

for i in range(0, len(YB)):
    #Y_OM_cost.append((rem_energy - Y_degrad[i]) / epr * om_cost)
    Y_Deg_cost.append(Y_degrad[i] * LCOD)
    #Y_profit.append(YB[i] - Y_OM_cost[i] - Y_Deg_cost[i])


#ipdb.set_trace()

df1 = pd.DataFrame(index=np.arange(0, len(YB)), columns=('Year', 'Savings', 'Degradation', 'Throughput', 'Deg Cost', 'SOH', 'OM Cost', 'NPV', 'Misc'))

for i in range(len(YB)):
    df1.iloc[i, 0] = i
    df1.iloc[i, 1] = YB[i]
    df1.iloc[i, 2] = Y_degrad[i]
    df1.iloc[i, 3] = Y_tp[i]
    df1.iloc[i, 4] = Y_Deg_cost[i] / ((1 + df) ** i)
    if i == 0:
        df1.iloc[i, 5] = 0.8
    else:
        df1.iloc[i, 5] = df1.iloc[i - 1, 5] - Y_degrad[i - 1] / rated_energy
    df1.iloc[i, 6] = (om_cost * (df1.iloc[i, 5] * rated_energy / epr)) / ((1 + df) ** i)
    if i == 0:
        df1.iloc[i, 7] = df1.iloc[i, 1] - df1.iloc[i, 6] - CAPEX * rem_energy
    else:
        df1.iloc[i, 7] = df1.iloc[i - 1, 7] + df1.iloc[i, 1] - df1.iloc[i, 6]

et2 = time.time()
total_time = et2 - st
print(total_time)
df1.iloc[0, 8] = total_time


df1.to_csv('Downloads\99520\Final_Week\Com_epr{}_soc1_size{}_cap{}.csv'.format(epr, osize, CAPEX))
