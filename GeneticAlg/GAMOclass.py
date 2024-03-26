import osmnx as ox
import networkx as nx
import geopandas as gpd
from collections import deque
from deap import base, creator, tools, algorithms
ox.settings.log_console=True
ox.settings.use_cache=True
from numpy.random import choice
from smart_mobility_utilities.common import randomized_search
import functools
import numpy as np
from GAclass import GeneticAlgorithm

class GAMO(object):


    def __init__(self,subgraph,successors_dict, origin,destination,popsize=50):


        ga =  GeneticAlgorithm(subgraph, successors_dict, source=origin, destination=destination)
        creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        self.subgraph = subgraph
        self.origin = origin
        self.destination = destination
        

        self.toolbox.register("individual", tools.initIterate, creator.Individual,
                              functools.partial(randomized_search, G=self.subgraph, source=self.origin, destination=self.destination))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", ga.fitnessEvaluation)
        self.toolbox.register("mate",  ga.crossover)
        self.toolbox.register("mutate", ga.mutation)
        self.toolbox.register("select", tools.selNSGA2)
        #ind1 = self.toolbox.individual()
        #ind1.fitness.values = self.toolbox.evaluate(ind1)
        self.pop = self.toolbox.population(n=popsize)
        self.hof = tools.HallOfFame(2, similar=np.array_equal)
        # print summary statistics for the population on each iteration
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    def run(self, logs_path, ngen=50, mutation_rate=0.25, crossover_rate=0.85):

        popsize = len(self.pop)

        invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        
        self.pop = self.toolbox.select(self.pop, popsize) #pop + fitness
        self.hof.update(self.pop)  # best 10 chromosomes
        
        
        bestCh = tools.selBest(self.pop, 1)  # best chromosome
        #print('AICI:',self.hof[0].fitness.valid) OK AICI

        #print(f"Best chromosome is: {bestCh[0]} and its fitness is: {self.toolbox.evaluate(bestCh[0])}")

        ## Begin the generational process
        for gen in range(1, ngen + 1):

            offspring = self.toolbox.select(self.pop, popsize) #OK
            #print('OFFSPRINGS:',offspring[0])
            # Vary the population by cloning the selected individuals
            offspring = [self.toolbox.clone(individual) for individual in offspring]
            #print('AICI:',offspring[0].fitness.valid)

            # CROSSOVER
            for i in range(1, len(offspring), 2):
                child1 = offspring[i - 1]
                child2 = offspring[i]
                if 0.1 < crossover_rate:
                    #print(f'Before crossing:{child1}, {child2}')
                    child1, child2 = self.toolbox.mate(child1, child2)

                    #print(f'After crossing:{child1}, {child2}')
                    #del child1.fitness.values
                    #del child2.fitness.values

            # MUTATION
            for i in range(len(offspring)):
                if 0.1 < mutation_rate:
                    #print(f'Before mating:{offspring[i]}')
                    offspring[i] = self.toolbox.mutate(offspring[i])
                    #print(f'After mating:{offspring[i]}')
                    del offspring[i].fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            self.hof.update(offspring)
            # Select the next generation population by combining the parent population with the offspring population
            # using select we get the best individuals from the combined population
            # !!the population size may change from one generation to the next
            self.pop = self.toolbox.select(self.pop + offspring, popsize)
            #self.pareto_front = tools.sortNondominated(self.pop, popsize, first_front_only=True)[0]
            #fitnPareto= [ind.fitness.values for ind in self.pareto_front]
            bestCh = tools.selBest(self.pop, 1)
            print(f"Best chromosome in generation nr. {gen} is: {bestCh[0]} and its fitness is: {self.toolbox.evaluate(bestCh[0])}")
            # Compute the statistics for the population
            #pop_stats = self.stats.compile(self.pop)

        return  bestCh