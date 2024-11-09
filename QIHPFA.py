from numpy import exp, sqrt, sum
from numpy.random import uniform
from numpy.random import uniform, choice
from numpy import where, clip, logical_and, maximum, minimum, power, sin, abs, pi, sqrt, sign, ones, ptp, min, sum, array, ceil, multiply, mean
from numpy.random import uniform, random, normal, choice
from numpy import mean, pi, sin, cos, array,zeros
from math import gamma
from copy import deepcopy
import numpy
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')


ID_WEI = 2
ID_MIN_PROB = 0  # min problem
ID_MAX_PROB = -1  # max problem
ID_POS = 0  # Position
ID_FIT = 1  # Fitness


def QIHPFA(obj_func, lb, ub, problem_size, pop_size, epochs, verbose=False):

    Convergence_curve = numpy.zeros(epochs)

    if not isinstance(lb, list):
        lb = [lb] * problem_size
        lb = numpy.array(lb)
    if not isinstance(ub, list):
        ub = [ub] * problem_size
        ub = numpy.array(ub)

    if not isinstance(lb, numpy.ndarray):
        lb = numpy.array(lb)
    if not isinstance(ub, numpy.ndarray):
        ub = numpy.array(ub)

    pop = [create_solution(lb,ub,obj_func,pop_size) for _ in range(pop_size)]

    pop, g_best = get_sorted_pop_and_global_best_solution(pop=pop, id_fit=ID_FIT, id_best=ID_MIN_PROB)
    gbest_present = deepcopy(g_best)


    for epoch in range(epochs):
        alpha, beta = uniform(1, 2, 2)
        A = uniform(lb, ub) * exp(-2 * (epoch + 1) / epochs)

        temp = gbest_present[ID_POS] + 2 * uniform() * (gbest_present[ID_POS] - g_best[ID_POS]) + A
        temp = amend_position_faster(temp, lb, ub)
        fit = get_fitness_position(obj_func, temp)
        g_best = deepcopy(gbest_present)
        if fit < gbest_present[ID_FIT]:
            gbest_present = [temp, fit]
        pop[0] = deepcopy(gbest_present)

        for i in range(1, pop_size):
            temp = deepcopy(pop[i][ID_POS])
            pos_new = deepcopy(pop[i][ID_POS])

            t1 = beta * uniform() * (gbest_present[ID_POS] - temp)
            for k in range(1, pop_size):
                dist = sqrt(sum((pop[k][ID_POS] - temp)**2)) / problem_size
                t2 = alpha * uniform() * (pop[k][ID_POS] - temp)

                t3 = uniform() * (1 - (epoch + 1) * 1.0 / epochs) * (dist / (ub - lb))
                pos_new += t2 + t3

            pos_new = (pos_new + t1) / pop_size

            pos_new = amend_position_faster(pos_new, lb, ub)
            fit_new = get_fitness_position(obj_func, pos_new)
            if fit_new < pop[i][ID_FIT]:
                pop[i] = [pos_new, fit_new]


        pop, gbest_present = update_sorted_population_and_global_best_solution(pop, ID_MIN_PROB, g_best)

        pop_copy = numpy.zeros((pop_size, problem_size))
        pop_best_copy = numpy.zeros(problem_size)
        for i in range(pop_size):
            for j in range(problem_size):
               pop_copy[i,j] = deepcopy(pop[i][ID_POS][j])

        for i in range(problem_size):
            pop_best_copy[i] = deepcopy(gbest_present[ID_POS][i])
    
        pop_fit = numpy.array([item[ID_FIT] for item in pop])
        pop_copy = QISSA(pop_size, pop_copy, problem_size, epoch, epochs, pop_fit, pop_best_copy, pop_best_copy[ID_FIT], lb[0], ub[0])

        for i in range(pop_size):
            fit_new = get_fitness_position(obj_func, pop_copy[i])
            if fit_new < pop[i][ID_FIT]:
                pop[i] = [pop_copy[i], fit_new]

        pop, gbest_present = update_sorted_population_and_global_best_solution(pop, ID_MIN_PROB, g_best)


        if type(gbest_present[ID_FIT]) == numpy.ndarray:
            gbest_present[ID_FIT] = gbest_present[ID_FIT][0]

        Convergence_curve[epoch] = gbest_present[ID_FIT]


        if verbose:
            print("> Epoch: {}, Best fit: {}".format(epoch + 1, gbest_present[ID_FIT]))

    print(f"{gbest_present[ID_FIT]:e}") 


    return gbest_present[ID_FIT]



def create_solution(lb , ub, obj_func,dim,minmax=0):

    position = uniform(lb,ub)
    fitness = get_fitness_position(obj_func, position=position, minmax=minmax)
    weight = zeros(dim)
    return [position, fitness,weight]

def get_fitness_position(obj_func,position, minmax=0):
    fit_new = obj_func(position)
    return fit_new if minmax == 0 else 1.0 / (fit_new + EPSILON)


def update_global_best_solution(pop=None, id_best=None, g_best=None):
    sorted_pop = sorted(pop, key=lambda temp: temp[ID_FIT])
    current_best = sorted_pop[id_best]
    return deepcopy(current_best) if current_best[ID_FIT] < g_best[ID_FIT] else deepcopy(g_best)

def get_global_best_solution(pop=None, id_fit=None, id_best=None):
    sorted_pop = sorted(pop, key=lambda temp: temp[id_fit])
    return deepcopy(sorted_pop[id_best])


def get_sorted_pop_and_global_best_solution(pop=None, id_fit=None, id_best=None):
    sorted_pop = sorted(pop, key=lambda temp: temp[id_fit])
    return sorted_pop, deepcopy(sorted_pop[id_best])

def update_sorted_population_and_global_best_solution(pop=None, id_best=None, g_best=None):
    sorted_pop = sorted(pop, key=lambda temp: temp[ID_FIT])
    current_best = sorted_pop[id_best]
    g_best = deepcopy(current_best) if current_best[ID_FIT] < g_best[ID_FIT] else deepcopy(g_best)
    return sorted_pop, g_best


def amend_position_faster(position, lb, ub):
    condition = np.logical_and(lb <= position, position<= ub)
    pos_rand = uniform(lb, ub)
    position =  np.where(condition, position, pos_rand)
    return position


def QISSA(N, X, dim, Iteration, Max_iteration, fitness, gbest, gbestFitness, lb, ub):

    import random
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    c1 = 2 * math.exp(-((4 * Iteration / Max_iteration) ** 2))
    X = numpy.transpose(X)
    X_temp = X
    for i in range(N):
        if i < N / 2:
            for j in range(0, dim):
                c2 = random.random()
                c3 = random.random()
                if c3 < 0.5:
                    
                    X_temp[j, i] = gbest[j] + c1 * (
                        (ub[j] - lb[j]) * c2 + lb[j])
                else:
                    X_temp[j, i] = gbest[j] - c1 * (
                        (ub[j] - lb[j]) * c2 + lb[j])
        elif i>=N/2 and i<N+1:
                rand = random.sample(range(N - 1), 2)
                r1 = int(rand[0])
                r2 = int(rand[1])
                for j in range(0, dim):
                    Xr1 = X_temp[:, r1]
                    Xr2 = X_temp[:, r2]
                    Xr1_fitness = fitness[r1]
                    Xr2_fitness = fitness[r2]
                    SS1=(((Xr1[j]-Xr2[j])**2)*gbestFitness+((Xr2[j]-gbest[j])**2)*Xr1_fitness +((Xr1[j]-gbest[j])**2)*Xr2_fitness) 
                    SS2=((Xr1[j]-Xr2[j])*gbestFitness+(Xr2[j]-gbest[j])*Xr1_fitness+(Xr1[j]-gbest[j])*Xr2_fitness)  
                    X_temp[j, i] = 0.5*(SS1/SS2)

                    
    X_temp = numpy.transpose(X_temp)

    condition = np.logical_and(lb <= X_temp, X_temp<= ub)
    pos_rand = uniform(lb, ub)
    X_temp =  np.where(condition, X_temp, pos_rand)


    return X_temp




