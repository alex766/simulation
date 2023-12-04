from simulation import parse_params, run_and_plot

xcel_params = parse_params("logit_interactions.xlsx")
# for p in xcel_params.keys():
#   print(p, xcel_params[p])

bundle_generation_options = ["uber"]
pricing_algs = ["zero_all", "uber_low", "uber_avg", "uber_high", "max_all", "dp_alg 2"]
num_loads = 100
carriers_to_loads_ratio = 1.5
model_params = xcel_params["Model 1"]

params_ratio = [1]
booking_ratio = [1]

run_and_plot(bundle_generation_options, pricing_algs, num_loads, carriers_to_loads_ratio, model_params, 
                booking_ratio = booking_ratio, ratio = params_ratio, penalty_ratio = 2)