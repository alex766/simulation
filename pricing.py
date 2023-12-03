# from binascii import b2a_base64
# from IPython.core.display import b2a_hex
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from scipy.special import lambertw

from structures import Load, Bundle
from gurobipy import *


def generate(env, SP_instance, feasible_bundles, nbr_bundles):
    """
    Generate a set of at most {nbr_bundles} bundles using the weighted max packing algorithm
    Return: list of bundles
    """
    m = Model("weighted_max_packing", env=env)

    # Decision variables
    x = {b: m.addVar(vtype = GRB.BINARY, name = "x_"+b.get_string()) for b in feasible_bundles}
    
    # Objective function
    m.setObjective(sum([x[b]*\
    (
        SP_instance.params["Deadhead"]*(SP_instance.avg_dist_to_pickup[b.loads[0].org_market]+b.em)
        +SP_instance.params["Distance"]*(b.d-b.em)\
        +SP_instance.params["log(deadhead)"]*np.log(SP_instance.avg_dist_to_pickup[b.loads[0].org_market]+b.em)
        +SP_instance.params["log(distance)"]*np.log(b.d-b.em)\
        +SP_instance.params["Bundle size 2"]*(len(b.loads)==2)\
        +SP_instance.params["Bundle size 3"]*(len(b.loads)==3)\
        +SP_instance.params["Price"]*sum([SP_instance.marginal_costs[l.org_market][l.id] for l in b.loads])\
    )
    for b in feasible_bundles]), GRB.MAXIMIZE)

    # Constraints
    m.addConstr(sum([x[b] for b in feasible_bundles]) <= nbr_bundles)
    for l in SP_instance.loads:
        m.addConstr(sum([x[b] for b in feasible_bundles if l in b.loads]) <= 1)

    m.optimize()

    # Select bundles using the solution of the MILP
    selected_bundles = []
    for b in feasible_bundles:
      if x[b].x > .5:
        selected_bundles.append(b)

    return selected_bundles

reverse_market_mappings = {"SanAntonio": "TX_ANT", "Austin": "TX_AUS", "Dallas": "TX_DAL", "Houston": "TX_HOU"}



class StaticPricing:
    
    def __init__(self,
          loads,
          arrival_rate_per_market,
          initial_time,
          params,
          approx = 2 #marginal cost estimation; 0: picewise static, 1: one segment then static, 2: static
        ):

        self.lr = 1e-1
        self.U0 = -params["Booking constant"]
        self.beta_1 = params["Distance"]
        self.beta_2 = params["Price"]
        self.beta_bundle = params["Bundle size 2"]
        self.params = params
        self.gamma = 0
        self.loads = loads
        self.arrival_rate_per_market = arrival_rate_per_market
        self.approx = approx

        self.avg_prices = {
                "SanAntonio": {"SanAntonio": 255, "Austin": 390, "Dallas": 825, "Houston": 600},
                "Austin": {"SanAntonio": 435, "Austin": 315, "Dallas": 490, "Houston": 555},
                "Dallas": {"SanAntonio": 575, "Austin": 510, "Dallas": 275, "Houston": 570},
                "Houston": {"SanAntonio": 545, "Austin": 565, "Dallas": 670, "Houston": 265}
            }

        self.avg_dist_to_pickup = {
            'SanAntonio': 18.66801830224827,
            'Austin': 41.330677093446816,
            'Dallas': 50.06450341594679,
            'Houston': 35.30568982183587
        }

        self.loads_by_market = dict()
        for l in loads:
            if l.org_market in self.loads_by_market:
                self.loads_by_market[l.org_market].append(l)
            else:
                self.loads_by_market[l.org_market] = [l]

        self.marginal_costs = dict()
        for market in self.loads_by_market:
            self.__update_marginal_costs(market, initial_time)

    def get_raw_utlility(self, b, dist_to_pickup):
        """
        Return the raw utility of a bundle i.e. the utilty excluding price related components (price and price x distance)
        """
        utility = \
                (dist_to_pickup+b.em) * self.params["Deadhead"]\
            +   np.log((dist_to_pickup+b.em)/100) * self.params["log(deadhead)"]\
            +   (b.d-b.em) * self.params["Distance"]\
            +   np.log((b.d-b.em)/100) * self.params["log(distance)"]\
            +   (len(b.loads)==2) * self.params["Bundle size 2"]\
            +   (len(b.loads)==3) * self.params["Bundle size 3"]\
            +   (((dist_to_pickup+b.em)/100)**2) * self.params["Deadhead^2"]\
            +   (((b.d-b.em)/100)**2) * self.params["Distance^2"]\

        load_dest = {'TX_ANT': self.params["To SAN"], 'TX_DAL': self.params["To DAL"], 'TX_FTW': self.params["To DAL"], 'TX_HOU': self.params["To HOU"], 'TX_AUS': 0}
        load_orig = {'TX_ANT': self.params["From SAN"], 'TX_DAL': self.params["From DAL"], 'TX_FTW': self.params["From DAL"], 'TX_HOU': self.params["From HOU"], 'TX_AUS': 0}
        utility += load_dest[reverse_market_mappings[b.loads[-1].dst_market]]
        utility += load_orig[reverse_market_mappings[b.loads[0].org_market]]
        return utility

    def get_prices(self, B, session, nbr_display, homogeneous_prices = False):
        """
        Return the optimal prices of the best {nbr_display} bundles for session {session}, selected among the set of bundles {B}
        [{homogeneous_prices} = True] to make prices indepent of the session's location
        """
        
        if self.approx>0:
          self.__update_marginal_costs(session.market, session.time)

        B = [b for b in B if b.loads[0].org_market==session.market]

        if homogeneous_prices:
            dist = lambda b: self.avg_dist_to_pickup[session.market] + b.d
        else:
            dist = lambda b: b.dist_to_dropoff(session.location)

        utilities = {b.id: self.get_raw_utlility(b, dist(b)-b.d)\
                +(self.params["Price"]+b.d/100*self.params["Price x Distance"]/100)*(sum([self.marginal_costs[l.org_market][l.id] for l in b.loads]))\
                -self.U0 - 1
                for b in B}

        B = sorted(B, key = lambda b:utilities[b.id], reverse = True)
        B_displayed = B[:nbr_display]
        
        inner_sum = sum([np.exp(utilities[b.id]) for b in B_displayed])
        Gamma = lambertw(inner_sum).real

        prices = {b.id : 0 for b in B}
        for b in B_displayed:
           prices[b.id] = sum([self.marginal_costs[l.org_market][l.id] for l in b.loads]) - (1+Gamma)/self.beta_2

        return prices

    def remove_load(self, load, current_time):
        market = load.org_market
        self.loads_by_market[market].remove(load)
        if self.approx==0:
          self.__update_marginal_costs(market, current_time)

    def __update_marginal_costs(self, market, current_time):
        """
        Update the marginal costs of market {market} using approximation {self.approx}
        """
        self.marginal_costs[market] = dict()
        loads = self.loads_by_market[market]
        arrival_rate = self.arrival_rate_per_market[market]

        if self.approx==2:
            proba_failure = self.__get_proba_failure(loads, arrival_rate, current_time)
            for l in loads:
              self.marginal_costs[market][l.id] = self.avg_prices[l.org_market][l.dst_market]*(1-proba_failure[l.id])\
                                +l.penalty*proba_failure[l.id]

        elif self.approx==1:
            proba_failure = self.__get_proba_failure(loads, arrival_rate, current_time, include_first_segment=False)
            expected_costs = {l: self.avg_prices[l.org_market][l.dst_market]*(1-proba_failure[l.id])\
                                +l.penalty*proba_failure[l.id] for l in loads}
            T0 = min([l.pickup_TW.y for l in loads])-current_time
            _, expected_costs, _ = self.__price_period(loads, arrival_rate, T0, expected_costs)
            for l in loads:
              self.marginal_costs[market][l.id] = expected_costs[l]

        else:
            costs, timestamps, rho = self.__price_market(L = loads, arrival_rate = arrival_rate, current_time=current_time)
            c0 = costs[0]

            for l in loads:
                if len(loads)==1:
                      marginal_cost = c0
                else:
                      costs, timestamps, _ = self.__price_market(L = [l2 for l2 in loads if l2.id!=l.id], arrival_rate = arrival_rate, current_time=current_time, rho = [rho[l2.id] for l2 in loads if l2.id!=l.id])
                      ci = costs[0]
                      marginal_cost = c0 - ci
                self.marginal_costs[market][l.id] = marginal_cost


    def __get_proba_failure(self, L, arrival_rate, current_time, include_first_segment = True):
        T = sorted(list(set([current_time]+[l.pickup_TW.y for l in L])))
        if include_first_segment==False:
          T=T[1:]
        proba_failure = {l.id: 1 for l in L}

        u = {l.id: np.exp(self.get_raw_utlility(Bundle([l]), self.avg_dist_to_pickup[l.org_market]) + (self.params["Price"]+l.d/100*self.params["Price x Distance"]/100)*self.avg_prices[l.org_market][l.dst_market]) for l in L}
        for t in list(range(len(T))[-2::-1]):  
            I = [l for l in L if l.pickup_TW.y>T[t]] # unexpired loads
            ut = {l.id: u[l.id] for l in I}
            u_tot = np.exp(self.U0)+sum(ut.values())
            for l in I:
              proba_failure[l.id] *= np.exp(-(T[t+1]-T[t])*arrival_rate*u[l.id]/u_tot)
        return proba_failure


    def __price_market(self, L, arrival_rate, current_time, rho = None, split=1):
        expected_costs = {l: l.penalty for l in L}
        T_base = sorted(list(set([current_time]+[l.pickup_TW.y for l in L])))
        T = [current_time]
        for t in range(len(T_base))[:-1]:
            T += list(np.linspace(T_base[t],T_base[t+1],split+1))[1:]
        Cs = [0 for _ in range(len(T)-1)]

        if rho==None:
            rho = [0.1/len(L) for i in range(len(L))]

        for t in list(range(len(T))[-2::-1]):  
            I = [i for i in range(len(L)) if L[i].pickup_TW.y>T[t]] # indexes of unexpired loads
            rho_step, expected_costs, C = self.__price_period([L[i] for i in I], arrival_rate, T[t+1]-T[t], expected_costs, initial_rho=[rho[i] for i in I])
            for i in range(len(I)): rho[I[i]] = rho_step[i]
            Cs[t] = C

        rho = {l.id: rho[i] for i,l in enumerate(L)}

        return Cs, T, rho
    
    def __price_period(self, L, arrival_rate, duration, expected_costs, alpha=1, m=0, initial_rho = None):
        I = np.arange(len(L))
        Lambda = arrival_rate*duration
        if initial_rho==None:
            rho = np.array([0.1/len(L) for i in range(len(L))])
        else:
            rho = np.array(initial_rho)
        grad = np.ones(len(I))
        lr = self.lr/len(L)/Lambda
        iter = 0
        K = np.array([self.beta_2*expected_costs[L[i]]-self.U0-1+self.beta_1*L[i].d for i in I]) #TODO -1?
        while(np.mean(np.abs(grad))>1e-3 and iter<1e4):
            iter+=1
            lr*=0.999
            
            one_minus_sum_rho = 1-np.sum(rho)
            sum_one_minus_exp = np.sum(1-np.exp(-Lambda*rho))
     
            grad = Lambda*np.exp(-Lambda*rho)\
                  * (np.log(rho)-np.log(one_minus_sum_rho)-K)\
                  + (1-np.exp(-Lambda*rho))/rho\
                  + sum_one_minus_exp/(one_minus_sum_rho)
                  
            step = -lr*grad
              
            if np.sum(step)>1e-2:
              step *= 1e-2 / np.sum(step)
            
            rho = np.minimum(1-1e-6, np.maximum(1e-6, rho+step))

        prices = [1/self.beta_2*(np.log(rho[i])-np.log(1-np.sum(rho))+self.U0-self.beta_1*L[i].d) for i in I]
        for i in I:
            expected_costs[L[i]] = expected_costs[L[i]]+(1-np.exp(-Lambda*rho[i]))*(prices[i]-expected_costs[L[i]])
        C = sum([expected_costs[L[i]] for i in I])

        return rho, expected_costs, C
                    
