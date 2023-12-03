import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import math
import csv
from structures import Session, Vector2
# file_path = '/content/gdrive/MyDrive/simulation/'
file_path = ''

def str_to_time(timestr):
    try:
        year = int(timestr[:4])
        month = int(timestr[5:7])
        day = int(timestr[8:10])
        hr = int(timestr[11:13])
        mini = int(timestr[14:16])
        sec = int(timestr[17:])
        my_date = datetime(year, month, day, hr, mini, sec)
    except ValueError:
        my_date = None
    return my_date

#CARRIER START TIME ANALYSIS
def analyze_sessions(carrier_start, carrier_end, to_plot = False):
    xs = []
    ys = []
    times = []
    with open(file_path + 'sessions_sampled.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if row[0] == '':
                continue
            imp_ts = str_to_time(row[1])
            # imp_ts = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")
            lat = float(row[2])
            lng = float(row[3])

            if imp_ts < carrier_start or imp_ts > carrier_end:
                continue

            if imp_ts > carrier_end: # faster b/c table is ordered for now
                break

            xs.append(lng)
            ys.append(lat)

            times.append(imp_ts)

    if to_plot:
        gen_types = {
            "Dallas": {"p": 0.4, "x": -96.8, "y":32.9, "var":0.3},
            "SanAntonio": {"p": 0.2, "x": -98.4, "y":29.5, "var":0.15},
            "Houston": {"p": 0.2, "x": -95.3, "y":29.8, "var":0.2},
            "Austin": {"p": 0.1, "x": -97.8, "y":30.3, "var":0.1},
        }
        org_markets = {"SanAntonio": 0, "Austin": 0, "Dallas": 0, "Houston": 0}

        for i in range(len(xs)):
            types, dists = [], []
            for k in gen_types.keys():
                types.append(k)
                dists.append(find_dist(xs[i], ys[i], gen_types[k]["x"], gen_types[k]["y"]))
            type_s = types[dists.index(min(dists))]
            org_markets[type_s] += 1

        for k in org_markets.keys():
            org_markets[k] = org_markets[k] / len(xs)
        print(org_markets)

        plt.title("Carrier Starting Locations")
        plt.scatter(xs, ys, s = 2)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()

        plt.title("carrier start times")
        plt.hist(times)#bins = int(180/5))
        plt.xticks(rotation=45, ha='right')
        plt.show()

    print(len(xs), len(ys))
    return xs, ys, times

def get_frequency_distribution(file_name, num_rows):
    session_counts = [0] * num_rows
    with open(file_path + file_name, newline='') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
      header = 0
      for row in spamreader:
          if header == 0:
              header += 1
              continue
          if row[1] != '':
              num = int(row[0])
              freq = int(row[1])
              if num < len(session_counts):
                  session_counts[num] = freq
    
    probs = np.array(session_counts)
    probs = probs / np.sum(probs)

    return probs

def find_dist(x1, y1, x2, y2):
    y_mult = 54.6 #longitude
    x_mult = 69 #latitude
    x_miles = (x2-x1)*x_mult
    y_miles = (y2-y1)*y_mult
    return math.sqrt(x_miles*x_miles + y_miles*y_miles)

class Carriers:
  def __init__(self,
          carrier_start,
          loads_start,
          loads_end,
          carrier_start_time_uniform,
          NUM_CARRIERS
        ):
        self.carrier_start = carrier_start
        self.loads_start = loads_start
        self.loads_end = loads_end
        self.carrier_start_time_uniform = carrier_start_time_uniform
        self.NUM_CARRIERS = NUM_CARRIERS
        xs, ys, ts = analyze_sessions(self.carrier_start, self.loads_end)
        self.xs = xs
        self.ys = ys
        self.ts = ts
        self.sessions_dist = get_frequency_distribution("session_per_user.csv", 200)

  def generate_start_times(self, nbr_carriers):
    carrier_start_times = []
    for i in range(nbr_carriers):
        if self.carrier_start_time_uniform:
            t = np.random.rand(1)
        else:
            t = abs(np.random.normal(loc=0, scale=.3)) #flip all negatives to positive, just double of normal on postivies
        scaled_t = math.floor(t * 60 * 24 * 14) #minutes in the range of 2 weeks
        dt = timedelta(minutes=scaled_t)
        start_time = self.carrier_start + dt
        if start_time > self.loads_end:
            start_time = self.loads_end
        carrier_start_times.append(start_time)

    sorted_starts = sorted(carrier_start_times)
    return sorted_starts


  def generate_session_info(self, nbr_carriers, ratio = 1, debug = False):
    startingx = []
    startingy = []
    startingt = []

    gen_types = {
        "Dallas": {"p": 0.4, "x": -96.8, "y":32.9, "var":0.3},
        "SanAntonio": {"p": 0.2, "x": -98.4, "y":29.5, "var":0.15},
        "Houston": {"p": 0.2, "x": -95.3, "y":29.8, "var":0.2},
        "Austin": {"p": 0.1, "x": -97.8, "y":30.3, "var":0.1},
        "other": {"p": 0.1}
    }

    # xs, ys, ts = analyze_sessions(self.carrier_start, self.loads_end)
    # sessions_dist = get_frequency_distribution("session_per_user.csv", 200)

    sessions = []
    for i in range(nbr_carriers):
        nbr_sessions = np.random.choice(len(self.sessions_dist), p = self.sessions_dist)
        nbr_sessions = int(math.floor(nbr_sessions * ratio))
        session_idxs = np.random.choice(len(self.xs), nbr_sessions)
        for j in range(len(session_idxs)):
            idx = session_idxs[j]
            location = Vector2(self.xs[idx], self.ys[idx])
            types, dists = [], []
            for k in gen_types.keys():
                if k == "other":
                    continue
                types.append(k)
                dists.append(find_dist(location.x, location.y, gen_types[k]["x"], gen_types[k]["y"]))
            type_s = types[dists.index(min(dists))]
            t = math.floor((self.ts[idx] - self.carrier_start).total_seconds()/60/60)
            session = Session(t, location, type_s, i)
            sessions.append(session)
            startingx.append(location.x)
            startingy.append(location.y)

    if debug:
      print(self.sessions_dist)
      print(nbr_carriers, len(sessions))

    if debug:
      plt.title("Carrier Starting Locations")
      plt.xlabel("Longitude")
      plt.ylabel("Latitude")
      plt.scatter(startingx, startingy, s=3)
      plt.show()

    sessions.sort(key = lambda x: x.time)
    return sessions

  def generate_carrier_info(self, nbr_carriers):
      startingx = []
      startingy = []
      startingt = []

      gen_types = {
        "Dallas": {"p": 0.4, "x": -96.8, "y":32.9, "var":0.3},
        "SanAntonio": {"p": 0.2, "x": -98.4, "y":29.5, "var":0.15},
        "Houston": {"p": 0.2, "x": -95.3, "y":29.8, "var":0.2},
        "Austin": {"p": 0.1, "x": -97.8, "y":30.3, "var":0.1},
        "other": {"p": 0.1}
    }

      starts = self.generate_start_times(nbr_carriers)

      xs, ys, ts = analyze_sessions(self.carrier_start, self.loads_end) # TODO: can prob store so don't have to re do this multiple times

      sessions = []
      carriers = []
      for i in range(nbr_carriers):
          # Start location based on given estarting data
          idx = np.random.randint(0, len(xs))
          start = [xs[idx], ys[idx]]
          types = []
          dists = []
          for k in gen_types.keys():
              if k == "other":
                continue
              types.append(k)
              dists.append(find_dist(start[0], start[1], gen_types[k]["x"], gen_types[k]["y"]))
          type_s = types[dists.index(min(dists))]
          
          carriers.append({"start_x":start[0], "start_y":start[1], "market": type_s})
          startingx.append(start[0])
          startingy.append(start[1])
          # Generate end location
          if np.random.random()>0.5: # probability that start location = end location
              carriers[i]["end_x"] = start[0]
              carriers[i]["end_y"] = start[1]

              carriers[i]["preference"] = 1 #short haul
          else:
              type = np.random.choice(list(gen_types.keys()), p=[gen_types[gen]["p"] for gen in gen_types])
              if type == type_s:
                  carriers[i]["preference"] = 1 #short haul
              else:
                  carriers[i]["preference"] = 2 #long haul
              if type =="other":
                  end = np.random.uniform(low=[-99,28.5], high=[-93.5,34])
              else:
                  end = np.random.normal(loc=[gen_types[type]["x"],gen_types[type]["y"]], scale=gen_types[type]["var"])
              carriers[i]["end_x"] = end[0]
              carriers[i]["end_y"] = end[1]

          #add start times
          carriers[i]["start_time"] = starts[i]
          carriers[i]["load"] = -1

          #1 = short haul, 2 = long haul, 3 = no preference
          carriers[i]["preference"] = np.random.choice([carriers[i]["preference"], 3], p = [.7, .3])

          startingt.append(carriers[i]["start_time"])

          location = Vector2(start[0], start[1])
          t = math.floor((starts[i] - self.carrier_start).total_seconds()/60/60)
          session = Session(t, location, type_s, i)
          sessions.append(session)

      print(nbr_carriers, len(sessions))

      plt.title("Carrier Starting Locations")
      plt.xlabel("Longitude")
      plt.ylabel("Latitude")
      plt.scatter(startingx, startingy, s=3)
      plt.show()

      # plt.title("carrier start times")
      # plt.hist(startingt)#bins = int(180/5))
      # plt.xticks(rotation=45, ha='right')
      # plt.show()
      # return carriers
      return sessions