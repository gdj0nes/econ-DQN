import numpy as np

class LifeCycle(object):
    def __init__(self, gamma, R, life_span, batch_size, shocks=False):

        # Global environment parameters
        self.batch_size = batch_size
        self.life_span = life_span
        self.periods = life_span
        self.state_size = 4

        self.gamma = gamma
        self.R = R
        self.shocks = shocks
        self.skills = np.array([-0.245, 0, 0.245])

        self.aLimit = 0.0
        self.numA = 150
        self.aMax = 50
        alpha = 0.3
        z = np.linspace(0, 1, self.numA)
        self.aGrid = self.aLimit + self.aMax * (1 - (z ** alpha))

        self.shocks = np.array([-.325, 0, .325])

        # Episode parameters
        self.skill = None
        self.age = None
        self.savings = None

    def calc_cash(self, eval=False, shocks=False):

        age = self.age
        skill = self.skill
        savings = self.savings

        age_wage = 0.1 * age - 0.002 * age ** 2
        skill_wage = 0.05 * age * skill

        if shocks:
            shock = np.random.choice(self.shocks)
        else:
            shock = 0.

        income = np.exp(age_wage + skill_wage + shock)  # All state
        cash = income + self.R * savings  # Cash on hand calculation

        return cash

    def get_state(self, cash):

        return np.hstack([self.savings, self.age, self.skill, cash])

    def step(self, action, shock=False):

        cash = self.calc_cash()
        consumption = cash * (1 - action)
        # utility = np.log(consumption )
        utility = (consumption ** (1 - self.gamma)) / (1 - self.gamma)
        # Update environment
        self.savings = cash * action
        self.age += 1

        if (self.age == self.life_span).all():
            term = 0.
        else:
            term = 1.

        cash = self.calc_cash()
        return utility, self.get_state(cash), term

    def reset(self):
        """Reset the environment state with new savings

        :param skill: 
        :param init_savings: 
        :return: 
        """
        self.skill = np.random.choice(self.skills, size=(self.batch_size, 1))
        self.savings = np.random.choice(self.aGrid, size=(self.batch_size, 1))
        self.age = np.zeros((self.batch_size, 1))

        cash = self.calc_cash()
        term = 1
        return self.get_state(cash), term
