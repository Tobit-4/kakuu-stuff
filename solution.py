
import pyomo.environ as pyo
import numpy as np
from pyomo.opt import SolverFactory, TerminationCondition

def build_model():
    """Complete model construction with stochastic elements"""
    model = pyo.ConcreteModel()
    
    # ========== SETS ==========
    num_parts = 18  # From your Excel data
    num_machines = 2
    num_batches = 5
    num_customers = 4
    num_scenarios = 100
    
    model.I = pyo.Set(initialize=range(num_parts))          # Parts
    model.J = pyo.Set(initialize=range(num_batches))        # Batches
    model.M = pyo.Set(initialize=range(num_machines))       # Machines
    model.U = pyo.Set(initialize=range(num_customers))      # Customers
    model.Omega = pyo.Set(initialize=range(num_scenarios))  # Scenarios

    # ========== PARAMETERS ==========
    # Machine parameters (from Excel)
    model.A = pyo.Param(model.M, initialize={0:625, 1:1600})     # Area capacity
    model.H = pyo.Param(model.M, initialize={0:32.5, 1:40})      # Height capacity
    model.vt = pyo.Param(model.M, initialize={0:0.030864, 1:0.030864})  # Volume time
    model.rt = pyo.Param(model.M, initialize={0:1.4, 1:0.7})     # Recoat time
    model.set_m = pyo.Param(model.M, initialize={0:2, 1:1})       # Setup time
    model.tau = pyo.Param(model.M, initialize={0:60, 1:80})       # Cost rate
    model.epsPR = pyo.Param(model.M, initialize={0:6, 1:6.25})    # Emissions

    # Part parameters (from Excel - replace with actual values)
    part_data = {
        'height': [4.27, 2.18, 29.58, 18.99,10.77, 26.67, 14.38, 3.5, 3, 19.14, 16.78, 34.2, 1.18, 34.61, 13.87, 22.73, 17.04, 27.94],
        'area': [122.62, 178.34, 273.83, 89.68, 269.75, 258.54, 114.56, 454.89, 615.12, 33.58, 248.15, 666.63, 71.04, 51.41, 71.15, 161.49, 99.53,  56.85],
        'volume': [102.83, 214.79, 840.17, 683.06, 1928.6, 1375.9, 989.53, 683.48, 722.91, 164.78, 802.8, 8777.6, 44.74, 295.45, 670.69, 991.45, 703.08, 272.92],
        'due_date': [145, 129, 151, 158, 119, 185, 186, 223, 177, 175, 197, 178, 246, 208, 241, 235, 239, 252],
        'delay_cost': [1.5, 1.5, 1.5, 1.5, 1.75, 1.75, 1.75, 1.75, 1.75, 1.5, 1.5, 1.5, 1.5, 1.5, 2, 2, 2.0]
    }
    
    model.h = pyo.Param(model.I, initialize=part_data['height'])
    model.a = pyo.Param(model.I, initialize=part_data['area'])
    model.v = pyo.Param(model.I, initialize=part_data['volume'])
    model.dd = pyo.Param(model.I, initialize=part_data['due_date'])
    model.dc = pyo.Param(model.I, initialize=part_data['delay_cost'])

    # Transportation parameters
    model.dist = pyo.Param([0,1], model.U, initialize={
        (0,0):100, (0,1):75, (0,2):150, (0,3):200,
        (1,0):50, (1,1):25, (1,2):225, (1,3):125
    })
    model.unitTC = pyo.Param(initialize=0.1)
    model.epsilon_T = pyo.Param(initialize=0.05)

    # Stochastic parameters
    np.random.seed(42)
    model.mu = pyo.Param(model.I, initialize={i: model.v[i]*0.03 for i in model.I})
    model.sigma = pyo.Param(model.I, initialize={i: 0.1 for i in model.I})
    model.T_max = pyo.Param(model.M, initialize={m: 100 for m in model.M})
    model.p_m = pyo.Param(model.M, initialize={m: 0.05 for m in model.M})
    
    # Scenario generation
    model.xi = pyo.Param(model.I, model.Omega, 
                        initialize={(i,w): np.random.normal(0,1) 
                                   for i in model.I for w in model.Omega})
    model.delta = pyo.Param(model.I, model.Omega,
                          initialize={(i,w): np.random.lognormal(1, 0.5)
                                     for i in model.I for w in model.Omega})

    # ========== VARIABLES ==========
    model.X = pyo.Var(model.I, model.J, model.M, within=pyo.Binary)  # Assignment
    model.Y = pyo.Var(model.J, model.M, within=pyo.Binary)           # Batch usage
    
    # Continuous variables
    model.PC = pyo.Var(model.J, model.M, within=pyo.NonNegativeReals)  # Prod cost
    model.SC = pyo.Var(model.J, model.M, within=pyo.NonNegativeReals)  # Setup cost
    model.TC = pyo.Var(model.U, within=pyo.NonNegativeReals)           # Transport cost
    
    model.He = pyo.Var(model.J, model.M, within=pyo.NonNegativeReals)  # Batch height
    model.vol = pyo.Var(model.J, model.M, within=pyo.NonNegativeReals) # Batch volume
    
    model.s = pyo.Var(model.J, model.M, within=pyo.NonNegativeReals)   # Start time
    model.p = pyo.Var(model.J, model.M, within=pyo.NonNegativeReals)   # Process time
    model.c = pyo.Var(model.J, model.M, within=pyo.NonNegativeReals)   # Completion
    
    model.ct = pyo.Var(model.I, within=pyo.NonNegativeReals)          # Part completion
    model.tt = pyo.Var(model.I, within=pyo.NonNegativeReals)          # Tardiness
    
    model.TV = pyo.Var([0,1], model.U, within=pyo.NonNegativeReals)   # Transport vol
    model.TE = pyo.Var([0,1], model.U, within=pyo.NonNegativeReals)   # Transport emissions
    model.PE = pyo.Var(model.M, within=pyo.NonNegativeReals)          # Process emissions
    
    model.failure_ind = pyo.Var(model.M, model.Omega, within=pyo.Binary) # Failure indicator

    # ========== OBJECTIVE ==========
    f1_norm = 100000  # Normalization factors
    f2_norm = 5000
    alpha = 0.5        # Weight parameter
    
    def cost_rule(model):
        return (sum(model.PC[j,m] + model.SC[j,m] for j in model.J for m in model.M) + 
                sum(model.TC[u] for u in model.U) + 
                sum(model.dc[i] * model.tt[i] for i in model.I))
    
    def emissions_rule(model):
        return sum(model.TE[p,u] for p in [0,1] for u in model.U) + sum(model.PE[m] for m in model.M)
    
    def weighted_sum(model):
        return alpha*(cost_rule(model)/f1_norm) + (1-alpha)*(emissions_rule(model)/f2_norm)
    
    model.obj = pyo.Objective(rule=weighted_sum, sense=pyo.minimize)

    # ========== CONSTRAINTS ==========
    # Assignment constraints
    def assign_rule(model, i):
        return sum(model.X[i,j,m] for j in model.J for m in model.M) == 1
    model.assign_constr = pyo.Constraint(model.I, rule=assign_rule)
    
    # Capacity constraints
    def area_rule(model, j, m):
        return sum(model.a[i]*model.X[i,j,m] for i in model.I) <= model.A[m]
    model.area_constr = pyo.Constraint(model.J, model.M, rule=area_rule)
    
    # Batch processing constraints
    def height_rule(model, j, m):
        return model.He[j,m] >= max(model.h[i]*model.X[i,j,m] for i in model.I)
    model.height_constr = pyo.Constraint(model.J, model.M, rule=height_rule)
    
    # Time constraints
    def completion_rule(model, j, m):
        return model.c[j,m] >= model.s[j,m] + model.p[j,m]
    model.completion_constr = pyo.Constraint(model.J, model.M, rule=completion_rule)
    
    # Stochastic constraints
    def stochastic_time(model, i, w):
        return model.ct[i] <= model.dd[i] + model.delta[i,w]
    model.stochastic_constr = pyo.Constraint(model.I, model.Omega, rule=stochastic_time)
    
    # ... (add all other constraints from your formulation)

    return model

def solve_model(model):
    """Solve with configured solver"""
    solver = SolverFactory('glpk')  # Replace with 'gurobi' if available
    solver.options['tmlim'] = 600   # 10 minute timeout
    
    results = solver.solve(model, tee=True)
    
    if results.solver.termination_condition == TerminationCondition.optimal:
        print("\nOPTIMAL SOLUTION FOUND")
        print("="*50)
        print(f"Total Cost: £{pyo.value(model.obj):.2f}")
        print(f"- Production: £{sum(pyo.value(model.PC[j,m]) for j in model.J for m in model.M):.2f}")
        print(f"- Tardiness: £{sum(pyo.value(model.dc[i]*model.tt[i]) for i in model.I):.2f}")
        
        print(f"\nTotal Emissions: {pyo.value(model.total_emissions):.1f} kg CO2")
        
        # Print assignments
        print("\nASSIGNMENTS:")
        for i in model.I:
            for j in model.J:
                for m in model.M:
                    if pyo.value(model.X[i,j,m]) > 0.5:
                        print(f"Part {i+1} → Batch {j+1} → Machine {m+1}")
    else:
        print("No optimal solution found")
        print("Status:", results.solver.status)

if __name__ == "__main__":
    print("Building optimization model...")
    model = build_model()
    
    print("\nSolving model...")
    solve_model(model)