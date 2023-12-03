from datetime import datetime, timedelta
import math
from numpy import False_
from structures import Load, Bundle, haversine_distance

def find_dist(x1, y1, x2, y2):
    y_mult = 54.6 #longitude
    x_mult = 69 #latitude
    x_miles = (x2-x1)*x_mult
    y_miles = (y2-y1)*y_mult
    return math.sqrt(x_miles*x_miles + y_miles*y_miles)

def uber_generate_bundles(deliveries, AVG_SPEED, LOAD_TIME, UNLOAD_TIME, MAX_IDLE_TIME):
    # NOTES:
    #    - formulation doesn't enforce that drop off time is within the dropoff window, only pickup
    #    - should empty miles be calculated from carrier starting point ?? to see if > expected_deadhead ...

    bundles = []
    for i in range(len(deliveries)):
        bundles.append(Bundle([deliveries[i]]))
        for j in range(len(deliveries)):
            if i == j:
                continue
            
            if check_constraints(deliveries[i], deliveries[j], AVG_SPEED, LOAD_TIME, UNLOAD_TIME, MAX_IDLE_TIME, max_deadhead_constraint = True):
                bundles.append(Bundle([deliveries[i], deliveries[j]]))

    return bundles

def check_constraints(load1, load2, AVG_SPEED, LOAD_TIME, UNLOAD_TIME, MAX_IDLE_TIME, max_deadhead_constraint = False):
    # i dropoff != j pickup market, not ok to bundle
    if load1.dst_market != load2.org_market:
        return False
    
    deadhead = haversine_distance(load1.dropoff, load2.pickup)
    # expected_deadhead = {"TX_ANT": 31.7, "TX_FTW": 30.9, "TX_HOU": 29.5, "TX_AUS": 47.8, "TX_DAL": 30.9}
    expected_deadhead = {"SanAntonio": 31.7, "Houston": 29.5, "Austin": 47.8, "Dallas": 30.9}
    if max_deadhead_constraint:
        if deadhead > expected_deadhead[load1.dst_market]:
            return False

    earliest_pickup_i = load1.pickup_TW.x

    deliver_i_time = load1.d / AVG_SPEED
    i_to_j_time = deadhead / AVG_SPEED
    earliest_pickup_j = earliest_pickup_i + LOAD_TIME + deliver_i_time + UNLOAD_TIME + i_to_j_time
    if earliest_pickup_j > load2.pickup_TW.y:
        return False

    if load2.pickup_TW.x - earliest_pickup_j > MAX_IDLE_TIME:
        return False
    return True


def generate_2_3_bundles(loads, AVG_SPEED, LOAD_TIME, UNLOAD_TIME, MAX_IDLE_TIME):
    bundles = []
    for i in range(len(loads)):
        for j in range(len(loads)):
            if i == j:
                continue
            if check_constraints(loads[i], loads[j], AVG_SPEED, LOAD_TIME, UNLOAD_TIME, MAX_IDLE_TIME):
                bundles.append(Bundle([loads[i], loads[j]]))
                for k in range(len(loads)):
                    if i == k or j == k:
                        continue
                    if check_constraints(loads[j], loads[k], AVG_SPEED, LOAD_TIME, UNLOAD_TIME, MAX_IDLE_TIME):
                        bundles.append(Bundle([loads[i], loads[j], loads[k]]))
    return bundles
