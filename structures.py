import numpy as np
import math

class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_string(self):
        return f"{self.x},{self.y}"

def haversine_distance(a: Vector2, b: Vector2):
    R = 6371.0  # Radius of the Earth in kilometers
    lat1_rad, lon1_rad = np.radians(a.y), np.radians(a.x)
    lat2_rad, lon2_rad = np.radians(b.y), np.radians(b.x)
    dlat, dlon = lat2_rad - lat1_rad, lon2_rad - lon1_rad
    alpha = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    distance = 2 * R * math.atan2(np.sqrt(alpha), np.sqrt(1 - alpha))
    return distance * 0.621371

class Load:
    def __init__(self,
          id,           # unique id
          pu_x,         # x coordinate pickup location
          pu_y,         # y coordinate pickup location
          do_x,         # x coordinate dropoff location
          do_y,         # y coordinate dropoff location
          org_market,   # origin market
          dst_market,   # destination market
          pu_s,         # start pickup time window
          pu_e,         # end pickup time window
          do_s,         # start dropoff time window
          do_e,         # end dropoff time window
          created,      # time of creation
          penalty       # penalty cost for failed delivery
        ):
        self.id = id
        self.pickup = Vector2(pu_x, pu_y)
        self.dropoff = Vector2(do_x, do_y)
        self.pickup_TW = Vector2(pu_s, pu_e)
        self.dropoff_TW = Vector2(do_s, do_e)
        self.created_t = created
        self.org_market = org_market
        self.dst_market = dst_market
        self.tau = do_e
        self.penalty = penalty
        self.d = haversine_distance(self.pickup, self.dropoff)
        self.booked_by = -1

    def empty_miles(self, destination):
        """
        Return empty miles from the dropoff location of the load to the destination in argument
        """
        return haversine_distance(self.dropoff, destination)

class Bundle:
    def __init__(self, loads):
        self.loads = loads
        self.d = 0
        self.em = 0
        for i in range(len(loads)):
            self.d += loads[i].d
            if i<len(loads)-1:
                temp = loads[i].empty_miles(loads[i+1].pickup)
                self.em += temp
                self.d += temp
        self.tau = min(loads, key=lambda x:x.tau).tau
        self.id = self.get_string()
        self.num_loads = len(loads)

    def total_dist(self, s):
        """
        s: (ogn: Vector2, dst: Vector2)
        Return distance traveled from the starting location to the end location while delivering the bundle
        """
        return haversine_distance(s[0], self.loads[0].pickup) + self.d + haversine_distance(self.loads[-1].dropoff, s[1])

    def dist_to_dropoff(self, org):
        """
        ogn: Vector2
        Return distance traveled from the starting location to the dropoff location of the last load included in the bundle
        """
        return haversine_distance(org, self.loads[0].pickup) + self.d

    def total_em(self, s: Vector2):
        """
        s: (ogn: Vector2, dst: Vector2)
        Return empty miles traveled from the starting location to the end location while delivering the bundle
        """
        return haversine_distance(s[0], self.loads[0].pickup) + self.em + haversine_distance(self.loads[-1].dropoff, s[1])

    def total_em_no_dst(self, org: Vector2):
        """
        org: Vector2
        Return empty miles traveled from the starting location to the end location while delivering the bundle
        """
        return haversine_distance(org, self.loads[0].pickup) + self.em
    
    def total_dist_no_dst(self, org: Vector2):
        """
        org: Vector2
        Return distance traveled from the starting location to the end location while delivering the bundle
        """
        return haversine_distance(org, self.loads[0].pickup) + self.d

    def is_equal(self, bundle):
        if len(self.loads)!=len(bundle.loads):
            return False
        for i in range(len(self.loads)):
            if self.loads[i]!=bundle.loads[i]:
                return False
        return True

    def get_string(self):
        """
        Return a unique string representation of the bundle
        """
        return "-".join([str(load.id) for load in self.loads])

class Session:
    def __init__(self, time, location: Vector2, market, c_id):
        self.time = time
        self.location = location
        self.market = market
        self.carrier_id = c_id
        self.booked_bundle_id = -1