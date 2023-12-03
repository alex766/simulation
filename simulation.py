import numpy as np
import pandas as pd
import copy
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
import csv
import math
import warnings
import time
from statistics import mean

from uber_formulation import uber_generate_bundles, generate_2_3_bundles
from carriers import Carriers, analyze_sessions, str_to_time, get_frequency_distribution
from pricing import StaticPricing, generate
from structures import Vector2, Load, Bundle, Session, haversine_distance

warnings.filterwarnings('ignore')

carrier_start = datetime(2023, 7, 10)
loads_start = datetime(2023, 7, 17)
loads_end = datetime(2023, 7, 25)
NBR_TOTAL_LOADS = 2395
load_dates = range(loads_start.day,loads_end.day)

carrier_start_time_uniform = True

AVG_SPEED = 40 #mph
LOAD_TIME = 2 #hours, also try 1/2
UNLOAD_TIME = 2 #hours
MAX_IDLE_TIME = 10 #hours

file_path = ""

low_prices = {
    "TX_ANT": {"TX_ANT": 230, "TX_AUS": 350, "TX_DAL": 740, "TX_HOU": 540},
    "TX_AUS": {"TX_ANT": 390, "TX_AUS": 280, "TX_DAL": 440, "TX_HOU": 500},
    "TX_DAL": {"TX_ANT": 520, "TX_AUS": 460, "TX_DAL": 250, "TX_HOU": 510},
    "TX_HOU": {"TX_ANT": 490, "TX_AUS": 510, "TX_DAL": 600, "TX_HOU": 240}}

high_prices = {
    "TX_ANT": {"TX_ANT": 280, "TX_AUS": 430, "TX_DAL": 910, "TX_HOU": 660},
    "TX_AUS": {"TX_ANT": 480, "TX_AUS": 350, "TX_DAL": 540, "TX_HOU": 610},
    "TX_DAL": {"TX_ANT": 630, "TX_AUS": 560, "TX_DAL": 300, "TX_HOU": 630},
    "TX_HOU": {"TX_ANT": 600, "TX_AUS": 620, "TX_DAL": 740, "TX_HOU": 290}}

price_zero = 0
price_max = 1000

markets = ['TX_ANT', 'TX_HOU', 'TX_AUS', 'TX_DAL', 'TX_FTW']

market_mappings = { "TX_ANT": "SanAntonio", "TX_AUS": "Austin", "TX_DAL": "Dallas", "TX_HOU": "Houston", "TX_FTW": "Dallas"}
reverse_market_mappings = {"SanAntonio": "TX_ANT", "Austin": "TX_AUS", "Dallas": "TX_DAL", "Houston": "TX_HOU"}

#from Uber sessions
arrival_rate_per_market = {
    "Dallas": 0.45,
    "Houston": 0.38,
    "SanAntonio": 0.11,
    "Austin": 0.06
}

p_m_dict = {
      "TX_ANT": {"TX_ANT": [0,0], "TX_AUS": [0,0], "TX_DAL": [0,0], "TX_HOU": [0,0], "TX_FTW": [0,0]},
      "TX_AUS": {"TX_ANT": [0,0], "TX_AUS": [0,0], "TX_DAL": [0,0], "TX_HOU": [0,0], "TX_FTW": [0,0]},
      "TX_DAL": {"TX_ANT": [0,0], "TX_AUS": [0,0], "TX_DAL": [0,0], "TX_HOU": [0,0], "TX_FTW": [0,0]},
      "TX_HOU": {"TX_ANT": [0,0], "TX_AUS": [0,0], "TX_DAL": [0,0], "TX_HOU": [0,0], "TX_FTW": [0,0]},
      "TX_FTW": {"TX_ANT": [0,0], "TX_AUS": [0,0], "TX_DAL": [0,0], "TX_HOU": [0,0], "TX_FTW": [0,0]}}

def getMYD(d):
    return datetime(d.year, d.month, d.day)

def hours_from_start(t):
    return math.floor((t - carrier_start).total_seconds()/60/60)

def get_uber_avg(org_market, dest_market):
    if org_market == "TX_FTW":
        org_market = "TX_DAL"
    if dest_market == "TX_FTW":
        dest_market = "TX_DAL"
    return (high_prices[org_market][dest_market] + low_prices[org_market][dest_market])/2

def get_deliveries(num_loads, penalty_ratio = 2, show_plot = False):
    all_days = set()
    valid_deliveries = []
    count = 0
    found = True
    idx = -1
    load_id = 0
    pux, puy, dox, doy = [], [], [], []
    file_name = 'MIT_loads_data.csv' #'uf_data_to_mit_batch1.csv"
    intra = 0
    inter = 0

    # load_idxs = np.random.randint(NBR_TOTAL_LOADS, size=num_loads)
    with open(file_path + file_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            idx += 1
            if row[0] == '':
                continue
            s_pu = str_to_time(row[3])
            e_pu = str_to_time(row[4])
            s_do = str_to_time(row[5])
            e_do = str_to_time(row[6])
            created_t = str_to_time(row[13])
            if num_loads != -1 and len(valid_deliveries) >= num_loads:
                break
            if e_do < s_pu or (e_do - s_pu).total_seconds()/3600 > 48 or s_pu.month != carrier_start.month or s_pu.day not in load_dates:
                continue
            pu_lat_col = 9
            if row[pu_lat_col] == '':
                continue
            pu_lat = float(row[pu_lat_col])
            pu_lon = float(row[pu_lat_col + 1])
            do_lat = float(row[pu_lat_col + 2])
            do_lon = float(row[pu_lat_col + 3])
            if [pu_lat, pu_lon] == [do_lat, do_lon]:
                continue

            # # FOR INTRA VS INTER MARKET
            # if market_mappings[row[1]] == market_mappings[row[2]]:
            #     continue

            if num_loads > NBR_TOTAL_LOADS or idx % math.floor(NBR_TOTAL_LOADS/num_loads) == 0: #to get a better distribution of loads
                found = False
            if not found or num_loads == -1:
              all_days.add(getMYD(s_do))
              all_days.add(getMYD(e_pu))
              if show_plot:
                  plt.plot([pu_lon, do_lon], [pu_lat, do_lat], linewidth=0.3)
                  pux.append(pu_lon)
                  puy.append(pu_lat)
                  dox.append(do_lon)
                  doy.append(do_lat)
              found = True
              l1 = Load(id = load_id, pu_x = pu_lon, pu_y = pu_lat, do_x = do_lon, do_y = do_lat,
                        org_market = market_mappings[row[1]], dst_market = market_mappings[row[2]], pu_s = hours_from_start(s_pu),
                        pu_e = hours_from_start(e_pu), do_s = hours_from_start(s_do), do_e = hours_from_start(e_do), created = hours_from_start(created_t), penalty = get_uber_avg(row[1], row[2]) * penalty_ratio)
              valid_deliveries.append(l1)
              load_id += 1
              if l1.org_market == l1.dst_market:
                  intra += 1
              else:
                  inter += 1

    if show_plot:
        plt.scatter(pux, puy, c = "b", s = 0.5)
        plt.scatter(dox, doy, c = "r", s = 0.5)
        plt.title("All Loads on the Platform")
        plt.show()
    dates = sorted(list(all_days))

    print(intra / (inter + intra), inter + intra, len(valid_deliveries))
    return valid_deliveries

def find_avg_distance(carriers, loads):
    dists = {"SanAntonio": [], "Austin": [], "Dallas": [], "Houston": []}
    for i in range(len(carriers)):
        for j in range(len(loads)):
            if market_mappings[carriers[i]["market"]] == loads[j].org_market:
                c_loc = Vector2(carriers[i]["start_x"], carriers[i]["start_y"])
                dists[loads[j].org_market].append(haversine_distance(c_loc, loads[j].pickup))

    avg_dists = {}
    for k in dists.keys():
        avg_dists[k] = mean(dists[k])
    return avg_dists

def calculate_price(bundle_info, pricing_alg):
    d = 0
    market_price = 0
    for i in range(len(bundle_info.loads)):
        load_info = bundle_info.loads[i]
        startingMarket = reverse_market_mappings[load_info.org_market]
        endingMarket = reverse_market_mappings[load_info.dst_market]
        if startingMarket == "TX_FTW":
            startingMarket = "TX_DAL"
        if endingMarket == "TX_FTW":
            endingMarket = "TX_DAL"

        if pricing_alg == "uber_low":
            market_price += max(10, low_prices[startingMarket][endingMarket] - (high_prices[startingMarket][endingMarket] - low_prices[startingMarket][endingMarket])/2)
        elif pricing_alg == "uber_avg":
            market_price += (low_prices[startingMarket][endingMarket] + high_prices[startingMarket][endingMarket])/2
        elif pricing_alg == "uber_high":
            market_price += high_prices[startingMarket][endingMarket] + (high_prices[startingMarket][endingMarket] - low_prices[startingMarket][endingMarket])/2
        elif pricing_alg == "zero_all":
            market_price += price_zero
        elif pricing_alg == "max_all":
            market_price += price_max
        else:
            raise Exception("Pricing algorithm " + pricing_alg + " not defined")

    return market_price

def calc_single_bundle(normalized_utilities, displayed_lens):
    prob_single, prob_bundle = 0, 0
    num_single, num_bundle, avg_single, avg_bundle = 0,0,0,0
    for idx in range(len(normalized_utilities)):
        if displayed_lens[idx] == 1:
            prob_single += normalized_utilities[idx]
            num_single += 1
        elif displayed_lens[idx] == 2:
            prob_bundle += normalized_utilities[idx]
            num_bundle += 1

    if num_single > 0:
        avg_single = prob_single / num_single
    if num_bundle > 0:
        avg_bundle = prob_bundle / num_bundle
    return avg_single, avg_bundle

def calculate_utility(carrier_info: Session, bundle_info: Bundle, p, params, debug = False):
    p_mile_market = []

    dist_to_load = haversine_distance(carrier_info.location, bundle_info.loads[0].pickup)
    empty_miles = dist_to_load + bundle_info.em
    in_load_total = bundle_info.d - bundle_info.em
    total_miles = empty_miles + in_load_total

    utility = params["Booking constant"] + empty_miles * params["Deadhead"] + in_load_total * params["Distance"] + p * params["Price"]
    if "log(deadhead)" in params.keys() and "log(distance)" in params.keys():
        utility += np.log(empty_miles/100) * params["log(deadhead)"] + np.log(in_load_total/100) * params["log(distance)"]
    if len(bundle_info.loads) == 2:
        utility += params["Bundle size 2"]
    elif len(bundle_info.loads) >= 3:
        utility += params["Bundle size 3"]
    if "Price x Distance" in params.keys():
        utility += params["Price x Distance"] * (in_load_total/100)*(p/100)
    if "Deadhead^2" in params.keys():
        utility += params["Deadhead^2"] * (empty_miles/100)**2
        utility += params["Distance^2"] * (in_load_total/100)**2

    carrier_loc = {'TX_ANT': params["In SAN"], 'TX_DAL': params["In DAL"], 'TX_FTW': params["In DAL"], 'TX_HOU': params["In HOU"], 'TX_AUS': 0}
    load_dest = {'TX_ANT': params["To SAN"], 'TX_DAL': params["To DAL"], 'TX_FTW': params["To DAL"], 'TX_HOU': params["To HOU"], 'TX_AUS': 0}
    load_orig = {'TX_ANT': params["From SAN"], 'TX_DAL': params["From DAL"], 'TX_FTW': params["From DAL"], 'TX_HOU': params["From HOU"], 'TX_AUS': 0}
    utility += carrier_loc[reverse_market_mappings[carrier_info.market]]
    utility += load_dest[reverse_market_mappings[bundle_info.loads[-1].dst_market]]
    utility += load_orig[reverse_market_mappings[bundle_info.loads[0].org_market]]

    if "time_to_pickup" in params.keys():
        time_to_pickup = bundle_info.loads[0].pickup_TW.y - carrier_info.time # number of hours, all relative to simulation start
        utility += time_to_pickup * params["time_to_pickup"]

    return utility, p, empty_miles, p_mile_market

def choose_bundle(utilities, all_bundles, params, p_reject, nbr_display):
    np_ut = np.array(utilities)
    n_largest = np.min(np_ut)
    if len(np_ut) > nbr_display:
        n_largest = np.partition(np_ut, -nbr_display)[-nbr_display]

    displayed_utilities, displayed_idxs, displayed_lens = [], [], []
    for j in range(len(utilities)):
        if nbr_display == len(displayed_utilities): #ignore ties
            break
        if utilities[j] >= n_largest:
            displayed_utilities.append(utilities[j])
            displayed_idxs.append(j)

    # add in rejection / leave app option, utility = 0
    reject_u = 0
    if "no_booking" in params.keys():
        reject_u = params["no_booking"]

    displayed_utilities.append(reject_u)
    displayed_idxs.append(-1)

    np_arr = np.array(displayed_utilities)

    # calculate softmax of utilities
    normalized_utilities = np.exp(np_arr)/np.exp(np_arr).sum()

    if p_reject is not None:
        normalized_utilities *= (1-p_reject)
        normalized_utilities[-1] += p_reject

    prob_sum = sum(normalized_utilities) - normalized_utilities[-1]
    prob_len = len(normalized_utilities)-1
    
    chosen_idx = np.random.choice(range(len(displayed_idxs)), p=normalized_utilities)
    chosen_bundle_idx = displayed_idxs[chosen_idx]
    return chosen_bundle_idx, prob_sum, prob_len

def simulate_new2(n_carriers, sessions, valid_deliveries, bundle_generation, pricing_alg, choice_model_params, p_impressions, p_reject=None, homogeneous_p = False):
    nbr_accept, total_empty_miles, nbr_single, nbr_bundled, nbr_reject, nbr_accepted, accepted_price_sum = 0, 0, 0, 0, 0, 0, 0
    ps, es, lead_times, bundles, ps_single, ps_bundle, ps_all, ps_sum, all_accepted_prices = [], [], [], [], [], [], [], [], []
    accepted_lens = [0] * 5
    loads = copy.deepcopy(valid_deliveries)
    nbr_loads = len(loads)
    arr_rate = copy.deepcopy(arrival_rate_per_market)
    c = copy.deepcopy(sessions)
    price_over_time = {}

    last_booked_do = [-1] * n_carriers
    end_times, created_times = [], []
    for i in range(nbr_loads):
        end_times.append(loads[i].pickup_TW.y)

    np_times = np.array(end_times)
    sorted_times_idx = np.argsort(np_times)
    time_idx = 0
    total_hours = (c[-1].time - c[0].time)
    arrival_rate = len(c) / total_hours
    for e in arr_rate:
        arr_rate[e]*= arrival_rate

    sp = StaticPricing(loads, arr_rate, 0, choice_model_params)
    if "dp_alg" in pricing_alg:
        sp.approx = int(pricing_alg.split()[1])
        if homogeneous_p == True:
            sp.avg_dist_to_pickup = find_avg_distance(c, loads)
    if bundle_generation == "single":
        for i in range(nbr_loads):
            bundles.append(Bundle([loads[i]]))
        ps_bundle.append(0)
    elif bundle_generation == "uber":
        bundles = uber_generate_bundles(loads, AVG_SPEED, LOAD_TIME, UNLOAD_TIME, MAX_IDLE_TIME)
    elif bundle_generation == "max_packing":
        feasible_bundles = generate_2_3_bundles(loads, AVG_SPEED, LOAD_TIME, UNLOAD_TIME, MAX_IDLE_TIME)
        bundles = generate(env, sp, feasible_bundles, nbr_loads/2)
        for i in range(nbr_loads):
            bundles.append(Bundle([loads[i]]))
    else:
        raise Exception("Bundle generation type " + bundle_generation + " not defined")

    price_per_mile_per_lane = copy.deepcopy(p_m_dict)
    price_per_mile, interm_price_per_mile = [], []
    avg_prices = []
    created_times_by_market = {'SanAntonio': [], 'Austin': [], 'Dallas': [], 'Houston': []}
    bundles_by_market = {'SanAntonio': [], 'Austin': [], 'Dallas': [], 'Houston': []}
    sorted_created_idx_by_market = {'SanAntonio': [], 'Austin': [], 'Dallas': [], 'Houston': []}
    created_idx_by_market = {'SanAntonio': 0, 'Austin': 0, 'Dallas': 0, 'Houston': 0}

    dummy_carrier = Session(0, Vector2(-96.8, 32.9), "Dallas", -1)

    for bundle in bundles:
        max_created = -1
        for l_idx in range(len(bundle.loads)):
            if bundle.loads[l_idx].created_t > max_created:
                max_created = bundle.loads[l_idx].created_t
        created_times_by_market[bundle.loads[0].org_market].append(max_created)
        bundles_by_market[bundle.loads[0].org_market].append(bundle)
    for k in bundles_by_market.keys():
        sorted_created_idx_by_market[k] = np.argsort(np.array(created_times_by_market[k]))

    for i in range(len(c)):
        session = c[i]
        t = session.time
        while time_idx < nbr_loads and t >= loads[sorted_times_idx[time_idx]].pickup_TW.y:
            load = loads[sorted_times_idx[time_idx]]
            if "dp_alg" in pricing_alg:
                if load in sp.loads_by_market[load.org_market]:
                    sp.remove_load(load, load.pickup_TW.y)
            time_idx += 1

        utilities, valid_bundle_idxs, valid_bundle_lens = [], [], []
        all_bundles = []

        starting_market = session.market
        sorted_created_times_idx = sorted_created_idx_by_market[starting_market]
        created_times = created_times_by_market[starting_market]
        bundles_to_use = bundles_by_market[starting_market]

        while created_idx_by_market[starting_market] < len(bundles_to_use) and t > created_times[sorted_created_times_idx[created_idx_by_market[starting_market]]]:
            created_idx_by_market[starting_market] = created_idx_by_market[starting_market] + 1
        created_idx = created_idx_by_market[starting_market]

        # filter for all valid bundles for this session
        for j in range(created_idx):
            bundle = bundles_to_use[sorted_created_times_idx[j]]
            if bundle.loads[0].org_market != session.market:
                continue
            carrier_last_do = last_booked_do[session.carrier_id]
            if carrier_last_do != -1 and bundle.loads[0].pickup_TW.y < carrier_last_do:
                continue
            not_yet_booked = True
            min_pu_e = -1
            max_created = -1
            for l_idx in range(bundle.num_loads):
                if bundle.loads[l_idx].booked_by != -1:
                    not_yet_booked = False
                    break
                if min_pu_e == -1 or bundle.loads[l_idx].pickup_TW.y < min_pu_e:
                    min_pu_e = bundle.loads[l_idx].pickup_TW.y
                if bundle.loads[l_idx].created_t > max_created:
                   max_created = bundle.loads[l_idx].created_t
            if not_yet_booked:
                if t < bundle.loads[0].pickup_TW.x and t < min_pu_e: # carrier session start is before first load in bundle starts, is valid pairing
                    if t >= max_created: #all loads in bundle have been created at this point
                        all_bundles.append(bundle)

        if len(all_bundles) == 0: #no loads are feasible for this carrier
            nbr_reject += 1
            continue

        nbr_display = np.random.choice(len(p_impressions), p=p_impressions) + 1 # +1 b/c 0th index = 1 impression
        prices = {}
        if "dp_alg" in pricing_alg:
            prices = sp.get_prices(all_bundles, session, nbr_display, homogeneous_p)
        else:
            for j in range(len(all_bundles)):
                prices[all_bundles[j].get_string()] = calculate_price(all_bundles[j], pricing_alg)

        for j in range(len(all_bundles)):
            p_ids = all_bundles[j].get_string()
            if prices[p_ids] < 0:
                prices[p_ids] = 0
            u, p, e, p_m = calculate_utility(session, all_bundles[j], prices[p_ids], choice_model_params)
            utilities.append(u)

        if dummy_carrier.market == session.market:
            dummy_prices = {}
            if "dp_alg" in pricing_alg:
                dummy_carrier.time = t
                dummy_prices = sp.get_prices(all_bundles, dummy_carrier, 100, homogeneous_p)
            else:
                dummy_prices = prices
            for pr in dummy_prices.keys():
                if pr not in price_over_time:
                    price_over_time[pr] = []
                if (len(price_over_time[pr]) > 0 and price_over_time[pr][-1][0] == t) or dummy_prices[pr] == 0:
                    continue
                price_over_time[pr].append([t, dummy_prices[pr]])

        chosen_bundle_idx, prob_sum, prob_len = choose_bundle(utilities, all_bundles, choice_model_params, p_reject, nbr_display)
        ps_all.append(prob_sum/prob_len) #avg of all non-reject utilities
        ps_sum.append(prob_sum)

        if chosen_bundle_idx == -1: #reject, leaving app
            continue

        # Bundle "chosen_bundle_idx" has been booked by session, perform calculations for price/distance analysis
        c[i].booked_bundle_id = chosen_bundle_idx
        bundle = all_bundles[chosen_bundle_idx]
        last_booked_do[session.carrier_id] = bundle.loads[-1].dropoff_TW.x
        bundle_price = prices[bundle.get_string()]
        accepted_price_sum += bundle_price
        all_accepted_prices.append(bundle_price)
        price_per_mile.append(bundle_price / bundle.d)
        inter_market = True
        for j in range(bundle.num_loads):
            if "dp_alg" in pricing_alg:
                sp.remove_load(bundle.loads[j], session.time)
            if bundle.loads[j].org_market == bundle.loads[j].dst_market:
                inter_market = False
            bundle.loads[j].booked_by = i
            price_per_mile_per_lane[reverse_market_mappings[bundle.loads[j].org_market]][reverse_market_mappings[bundle.loads[j].dst_market]][0] += bundle_price * bundle.loads[j].d/ bundle.d
            price_per_mile_per_lane[reverse_market_mappings[bundle.loads[j].org_market]][reverse_market_mappings[bundle.loads[j].dst_market]][1] += bundle.loads[j].d
        if inter_market:
            interm_price_per_mile.append(bundle_price / bundle.d)
        if bundle.num_loads == 1:
            nbr_single += 1
        elif bundle.num_loads >= 2:
            nbr_bundled += bundle.num_loads
        nbr_accepted += bundle.num_loads
        accepted_lens[bundle.num_loads] += 1
        lead_times.append(bundle.loads[0].pickup_TW.x - t)
        total_empty_miles += bundle.total_em_no_dst(session.location)

    penalty_loads = 0
    for i in range(nbr_loads):
        if loads[i].booked_by == -1:
            penalty_loads += loads[i].penalty

    mean_price = 0
    if len(price_per_mile) != 0:
        mean_price = mean(price_per_mile)

    if nbr_accepted != 0:
        total_empty_miles = total_empty_miles/nbr_accepted

    price_per_mile.sort()
    interm_price_per_mile.sort()
    percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    all_percentiles = []
    interm_percentiles = []
    for p in percentiles:
        if len(price_per_mile)>0:
            idx = int(np.floor(p * len(price_per_mile)))
            all_percentiles.append(price_per_mile[idx])
        else:
            all_percentiles.append(0)
        if len(interm_price_per_mile) > 0:
            inter_idx = int(round(p * len(interm_price_per_mile)))
            inter_idx = min(inter_idx, len(interm_price_per_mile) - 1)
            interm_percentiles.append(interm_price_per_mile[inter_idx])
        else:
            interm_percentiles.append(0)

    return nbr_accepted, total_empty_miles, nbr_single, nbr_bundled, lead_times, price_per_mile_per_lane, mean(ps_all), mean(ps_sum), accepted_price_sum + penalty_loads, accepted_lens, mean_price, np.array(all_percentiles), np.array(interm_percentiles), price_over_time

def plot_scatter(x, y, title, x_lab, y_lab, colors, ratio):
    plt.scatter(x, y, color = colors)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    # plt.legend(bbox_to_anchor = (1.05,1), loc='upper left')
    if len(ratio) <= 1:
        for i in range(len(x)):
            i_ratio = ratio[i % len(ratio)]
            label = "{:.2f}".format(y[i])
            if len(ratio) > 1:
                label = label + "-" + str(i_ratio)
            plt.annotate(label, (x[i],y[i]), textcoords="offset points", xytext=(0,7), ha='center')
    plt.show()

def plot_bar(x, y, title, x_lab, y_lab):
    plt.bar(x, y)
    plt.xlabel(x_lab)
    if len(x) > 7:
        plt.xticks(rotation=90)
    plt.ylabel(y_lab)
    plt.title(title)
    for i in range(len(y)):
        plt.text(i, y[i], int(y[i]), ha = 'center')
        # plt.text(i, y[i], round(y[i], 2), ha = 'center')
    plt.show()

def plot_line_scatter(x, y, title, x_lab, y_lab, colors, labels):
    for i in range(len(y)):
        plt.plot(x, y[i], color = colors[i], label = labels[i])
        plt.scatter(x, y[i], color = colors[i])
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.legend()
    plt.title(title)
    plt.show()

def get_labels_colors(len_algs, ratio):
    if len_algs in [5, 6, 7]:
        prices = [0, .25, .5, .75, 1]
        xlabels = ["0", "low", "avg", "high", str(price_max)]
        if len_algs == 6:
            prices = [0, .25, .5, .75, 1, 1.25]
            xlabels = ["0", "low", "avg", "high", str(price_max), "dp_alg"]
        elif len_algs == 7:
            prices = [0, .25, .5, .75, 1, 1.25, 1.50]
            xlabels = ["0", "low", "avg", "high", str(price_max), "dp_alg 1", "dp_alg 2"]

        colors = ["r", "g", "b", "c", "y", "gray", "purple"]
        p_ratios, x_ratios, c_ratios = [], [], []
        for i in range(len(prices)):
            for s in range(len(ratio)):
                p_ratios.append(prices[i])
                if len(ratio) > 1:
                    x_ratios.append(xlabels[i] + "-" + str(ratio[s]))
                else:
                    x_ratios.append(xlabels[i])
                c_ratios.append(colors[s])
        return p_ratios, x_ratios, c_ratios, xlabels
    else:
        return [], [], [], []

def run_and_plot(bundle_gen_ops, pricing_algs, num_loads, carriers_ratio, choice_model_params, print_stats = False,
                 p_m_graph = False, lead_time_graph = False, empty_and_single_bundle = False, booking_ratio = [1], ratio = [1],
                 penalty_ratio = 2, p_reject=None, homogeneous_prices = False, sessions_ratio = 1):
    valid_deliveries = get_deliveries(num_loads, penalty_ratio)
    num_carriers = int(carriers_ratio * len(valid_deliveries))
    all_singles, all_bundles, all_accepts, all_empty_miles, labels = [], [], [], [], []
    mean_s_prob, mean_b_prob, mean_prob, mean_sum, all_costs, all_avg_lens, all_p_m = [], [], [], [], [], [], []
    num_runs = 5
    c_class = Carriers(carrier_start, loads_start, loads_end, carrier_start_time_uniform, num_carriers)
    p_impressions = get_frequency_distribution("impressions_per_session.csv", 185)
    
    if "Price x Distance" not in choice_model_params:
        choice_model_params["Price x Distance"] = 0
    if "Distance^2" not in choice_model_params:
        choice_model_params["Distance^2"] = 0
        choice_model_params["Deadhead^2"] = 0

    for i in range(len(bundle_gen_ops)):
        for p in range(len(pricing_algs)):
            for s in range(len(ratio)):
                print("BUNDLING TYPE: " + bundle_gen_ops[i] + ", PRICING ALG: " + pricing_algs[p] + ", RATIO: " + str(ratio[s]))
                r = ratio[s]
                br = booking_ratio[s]
                new_r_params = {}
                for choice in choice_model_params.keys():
                    if choice=="Booking constant":
                        new_r_params[choice] = choice_model_params[choice] * br
                    else:
                        new_r_params[choice] = choice_model_params[choice] * r
                    if choice == "Price" or choice == "Distance" or choice == "Deadhead":
                        new_r_params[choice] = new_r_params[choice] / 100
                n_accepts, empty_miles_all, single_loads, bundled_loads, all_lead_times, n_rejects, n_bundles, single_probs, bundled_probs, all_probs, all_sum_probs, costs, p_m = [], [], [], [], [], [], [], [], [], [], [], [], []
                accepted_b_lens, price_percentiles, interm_price_percentiles = [], [], []
                for j in range(num_runs):
                    all_sessions = c_class.generate_session_info(num_carriers, sessions_ratio)
                    res = simulate_new2(num_carriers, all_sessions, valid_deliveries, bundle_gen_ops[i], pricing_algs[p], new_r_params, p_impressions, p_reject=p_reject, homogeneous_p = homogeneous_prices)
                    n_accepts.append(res[0])
                    empty_miles_all.append(res[1])
                    single_loads.append(res[2])
                    bundled_loads.append(res[3])
                    all_lead_times += res[4] # concatenate to keep all
                    all_probs.append(res[6])
                    all_sum_probs.append(res[7])
                    costs.append(res[8])
                    accepted_b_lens.append(np.array(res[9]))
                    p_m.append(res[10])
                    price_percentiles.append(res[11])
                    interm_price_percentiles.append(res[12])

                    p_per_m = {}
                    lanes = []
                    price_per_mile = []
                    for m1 in res[5]:
                        for m2 in res[5][m1]:
                            if m1 not in p_per_m:
                                p_per_m[m1] = {}
                            if res[5][m1][m2][1] != 0:
                                p_per_m[m1][m2] = res[5][m1][m2][0] / res[5][m1][m2][1]
                            else:
                                p_per_m[m1][m2] = 0
                            lanes.append(m1 + " to " + m2)
                            price_per_mile.append(p_per_m[m1][m2])

                    if p_m_graph and j == 0: #maybe plot this less, more avged lol
                        plt.bar(lanes, price_per_mile) #, width = 0.4)
                        plt.xlabel("Lane")
                        plt.ylabel("Price per mile")
                        plt.title("Price per mile for fulfilled deliveries")
                        plt.xticks(rotation=90)
                        plt.show()

                # print("all accepted loads", np.array(price_percentiles).mean(axis = 0))
                # print("inter market", np.array(interm_price_percentiles).mean(axis = 0))

                nbr_single = mean(single_loads)
                nbr_bundled = mean(bundled_loads)
                mean_prob.append(mean(all_probs))
                mean_sum.append(mean(all_sum_probs))
                all_costs.append(mean(costs))
                all_p_m.append(mean(p_m))
                if lead_time_graph:
                    plt.title("lead times")
                    plt.hist(all_lead_times)
                    plt.xticks(rotation=45, ha='right')
                    plt.xlabel("# of hours between bundle booked and start time")
                    plt.show()

                bundle_lens = np.array(accepted_b_lens)
                avg_lens = np.mean(bundle_lens, axis = 0)
                # plot_bar(range(len(avg_lens)), avg_lens, "Number of loads in Different Sized Bundles for pricing: " + pricing_algs[p], "Size of Bundle", "Number of Loads")
                all_avg_lens.append(avg_lens)
                all_singles.append(nbr_single)
                all_bundles.append(nbr_bundled)
                all_accepts.append(mean(n_accepts))
                all_empty_miles.append(mean(empty_miles_all))
                labels.append("%s - %s - %.2f" % (bundle_gen_ops[i], pricing_algs[p], ratio[s] * 100))

    p_ratios, x_ratios, c_ratios, xlabels = get_labels_colors(len(pricing_algs), ratio)

    if empty_and_single_bundle:
        for i in range(len(all_singles)):
            plt.scatter(all_singles[i], all_bundles[i], label=labels[i])
        plt.legend(bbox_to_anchor = (1.05,1), loc='upper left')
        plt.title("Single loads vs bundled loads for different methods")
        plt.xlabel("# of Loads booked as single loads")
        plt.ylabel("# of Loads booked in a bundle of 2 loads")
        plt.show()

        for i in range(len(all_accepts)):
            plt.scatter(all_accepts[i], all_empty_miles[i], label=labels[i])
        plt.legend(bbox_to_anchor = (1.05,1), loc='upper left')
        plt.title("Empty miles per number of accepted loads for different methods")
        plt.show()
        plot_bar(x_ratios, all_empty_miles, "Average Empty Miles per Bundle Accepted", "Pricing Method", "Empty miles / number of bundles accepted")

    plot_scatter(p_ratios, mean_sum, title="Total acceptance probability (summed over loads)", x_lab=f"Price ({', '.join(xlabels)})", y_lab="Total acceptance probability", colors=c_ratios, ratio=ratio)
    plot_scatter(p_ratios, mean_prob, title="Average probability of acceptance", x_lab=f"Price ({', '.join(xlabels)})", y_lab="Probability of acceptance", colors=c_ratios, ratio=ratio)

    plot_bar(x_ratios, all_accepts, "Accepted Loads per Pricing Method", "Pricing Method", "Number of accepted loads")
    print("Number of accepted loads for \n[" + ', '.join(xlabels) + "]")
    print(all_accepts)
    # plot_bar(x_ratios, all_costs, "Costs (prices of accepted loads + penalty for not accepted) to platform", "Pricing Method", "Platform Cost")
    # plot_bar(x_ratios, all_p_m, "Price per mile of accepted loads", "Pricing Method", "Price per Mile of Accepted Loads")

def parse_one_df(df):
    new_params = {}
    m_name = df.columns[0]
    for ind in df.index:
        new_params[df[m_name][ind]] = df['Estimate'][ind]
    return new_params

def parse_params(file_name):
    df = pd.read_excel(file_path + file_name)
    splits = df.loc[pd.isna(df[df.columns[0]])] #what happens if there's no empty row... will this throw error
    params = {}
    prev_cutoff = 0
    next_model = df.columns[0]

    for i in range(len(splits.index)):
        cutoff = splits.index[i]
        df1 = df.iloc[prev_cutoff:cutoff, :]
        params[next_model] = parse_one_df(df1)
        prev_cutoff = cutoff + 2
        next_model = df[next_model][cutoff + 1]

    df_last = df.iloc[prev_cutoff:, :]
    params[next_model] = parse_one_df(df_last)
    return params