"""

Author: Gareth Jones



"""

import numpy as np


class LifeCycle(object):
    """Implementation of a life cycle environment with ____ utility function.
    
    
    
    """
    def __init__(self, gamma, R, life_span, batch_size, act_dim, num_assets=40, delta=.0001):

        # Global environment parameters
        self.state_size = 2
        self.batch_size = batch_size
        self.life_span = life_span
        self.periods = life_span
        self.gamma = gamma
        self.R = R

        # Action grid
        self.actGrid = np.linspace(delta, 1 - delta, act_dim)
        # Shock grid
        self.shocks = np.array([-.325, 0, .325])
        # Skill grid
        self.skGrid = np.array([-0.245, 0, 0.245])
        # Asset Grid
        aLimit = 0.0
        numA = num_assets
        aMax = 5
        alpha = 0.3
        z = np.linspace(0, 1, numA)
        # self.aGrid = aLimit + aMax * (1 - (z ** alpha))
        self.aGrid = np.linspace(aLimit, aMax, numA)

        # actLimit = delta
        # numAct = 150
        # actMax = 1. - delta
        # alpha = 0.6
        # z = np.linspace(0, 1, numAct)
        # self.actGrid = actMax * (1 - (z ** alpha))
        # self.actGrid[-1] = actLimit

        # Episode parameters
        self.skill = None
        self.age = None
        self.savings = None

    def step(self, action_idx):

        action = self.actGrid[action_idx]
        consumption = self.cash * (1 - action)
        utility = (consumption ** (1 - self.gamma)) / (1 - self.gamma)
        # Update environment
        self.savings = self.cash * action
        self.age += 1

        if (self.age > self.life_span).all():
            term = 0.
        else:
            term = 1.

        return utility, self.state, term

    def reset(self):
        """Reset the environment state with new savings
        """
        self.skill = np.random.choice(self.skGrid, size=self.batch_size)
        self.savings = np.random.choice(self.aGrid, size=self.batch_size)
        self.age = np.ones(self.batch_size)
        term = 1
        return self.state, term

    @property
    def cash(self):

        age = self.age
        skill = self.skill
        savings = self.savings

        age_wage = 0.1 * age - 0.002 * age ** 2
        skill_wage = 0.05 * age * skill

        income = np.exp(age_wage + skill_wage)
        cash = income + self.R * savings

        return cash

    @property
    def state(self):
        # return np.vstack([self.savings, self.age, self.skill, self.cash]).T

        return np.vstack([self.age, self.cash]).T
