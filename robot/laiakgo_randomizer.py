import laikago_constant
import random
import numpy as np

class LaikagoRobotRandomizer(object):
    def __init__(self, laikago,
                 mass_bound=laikago_constant.MASS_BOUND,
                 inertia_bound=laikago_constant.INERTIA_BOUND,
                 joint_f_bound=laikago_constant.JOINT_F_BOUND,
                 toe_f_bound=laikago_constant.TOE_F_BOUND):
        self.laikago = laikago
        self.mass_bound = mass_bound
        self.inertia_bound = inertia_bound
        self.joint_f_bound = joint_f_bound
        self.toe_f_bound = toe_f_bound


    def randomize(self):
        base_mass = self.laikago.GetBaseMassFromURDF()
        randomized_base_mass = random.uniform(
            np.array(base_mass) * (1.0 + self.mass_bound[0]),
            np.array(base_mass) * (1.0 + self.mass_bound[1]))
        self.laikago.SetBaseMass(randomized_base_mass)

        leg_masses = self.laikago.GetLegMassesFromURDF()
        randomized_leg_masses = random.uniform(
            np.array(leg_masses) * (1.0 + self.mass_bound[0]),
            np.array(leg_masses) * (1.0 + self.mass_bound[1]))
        self.laikago.SetLegMasses(randomized_leg_masses)

        base_inertial = self.laikago.GetBaseInertiaFromURDF()
        randomized_base_inertial = random.uniform(
            np.array(base_inertial) * (1.0 + self.mass_bound[0]),
            np.array(base_inertial) * (1.0 + self.mass_bound[1]))
        self.laikago.SetBaseInertia(randomized_base_inertial)

        leg_inertias = self.laikago.GetLegInertiasFromURDF()
        randomized_leg_inertias = random.uniform(
            np.array(leg_inertias) * (1.0 + self.mass_bound[0]),
            np.array(leg_inertias) * (1.0 + self.mass_bound[1]))
        self.laikago.SetLegInertias(randomized_leg_inertias)

        randomized_toe_friction = random.uniform(
            np.full(4, self.toe_f_bound[0]),
            np.full(4, self.toe_f_bound[1]))
        self.laikago.SetToeFriction(randomized_toe_friction)

        randomized_joint_fraction = random.uniform(
            np.full(12, self.joint_f_bound[0]),
            np.full(12, self.joint_f_bound[1]))
        self.laikago.SetJointFriction(randomized_joint_fraction)

