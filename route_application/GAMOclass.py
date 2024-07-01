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
from .GAclass import GeneticAlgorithm
from .randomized_search_with_bias import randomized_search_biased
from .datasets_creation import load_my_graph
import random
import asyncio
import os
import pandas as pd
''' 
In this script I define the generational process making use of DEAP module.
'''
G, _ =  load_my_graph()
class GAMO(object):


    def __init__(self, successors_dict, origin,destination,popsize=50,training_period='whole', road_type='all_roads'):

        self.G=  G
        self.ga =  GeneticAlgorithm(successors_dict, source=origin, destination=destination,training_period=training_period,road_type=road_type )
        creator.create("FitnessMax", base.Fitness, weights=(-1.0, -1.0,-1.0))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        self.origin = origin
        self.destination = destination
        self.road_type = road_type  # Ensure road_type is correctly assigned

        if self.road_type == 'biased':
            self.toolbox.register("individual", tools.initIterate, creator.Individual,
                                  functools.partial(randomized_search_biased, G=self.G, source=self.origin, destination=self.destination))
        else:
            self.toolbox.register("individual", tools.initIterate, creator.Individual,
                                  functools.partial(randomized_search, G=self.G, source=self.origin, destination=self.destination))
            
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.ga.fitnessEvaluation_google)
        self.toolbox.register("mate",  self.ga.crossover)
        self.toolbox.register("mutate", self.ga.mutation)
        self.toolbox.register("select", tools.selNSGA2)
        #ind1 = self.toolbox.individual()
        #ind1.fitness.values = self.toolbox.evaluate(ind1)
        self.pop = self.toolbox.population(n=popsize)
        self.hof = tools.HallOfFame(2, similar=np.array_equal)
        # Define statistics for each fitness objective
        stats_congestion = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats_time = tools.Statistics(key=lambda ind: ind.fitness.values[2])
        stats_distance = tools.Statistics(key=lambda ind: ind.fitness.values[1])
        self.mstats = tools.MultiStatistics(
            congestion=stats_congestion,
            time=stats_time,
            distance=stats_distance
        )
        self.mstats.register("avg", np.mean)
        self.mstats.register("std", np.std)
        self.mstats.register("min", np.min)
        self.mstats.register("max", np.max)
        # Initialize the logbook
        self.logbook = tools.Logbook()
        self.logbook.header = ["gen", "nevals"]
        for stat_name, stat_obj in self.mstats.items():
            self.logbook.header.extend(f"{stat_name}_{field}" for field in stat_obj.fields)

    async def run(self, logs_path, ngen=50, mutation_rate=0.1, crossover_rate=0.9):

        popsize = len(self.pop)

        invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        #fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        fitnesses = await asyncio.gather(*[self.toolbox.evaluate(ind) for ind in invalid_ind])
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            #print(ind.fitness.values)
        
        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        #print(self.pop)

        self.pop = self.toolbox.select(self.pop, popsize) #pop + fitness
        self.hof.update(self.pop)  # best 10 chromosomes
        
        
        bestCh = tools.selBest(self.pop, 1)  # best chromosome
        #print('AICI:',self.hof[0].fitness.valid) #OK AICI

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
                if random.random() < crossover_rate:
                    self.toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values

            #MUTATION
            for i in range(len(offspring)):
                if random.random() < mutation_rate:
                    self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            #fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            fitnesses = await asyncio.gather(*[self.toolbox.evaluate(ind) for ind in invalid_ind])
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            self.hof.update(offspring)
            
            self.pop = self.toolbox.select(self.pop + offspring, popsize)

            bestCh = tools.selBest(self.pop, 1)
            bestChFitness = bestCh[0].fitness.values
            print(f"Best chromosome in generation nr. {gen} is: {bestCh[0]} and its fitness is: {bestChFitness}")
            # Gather all the fitnesses in one list and update the statistics object
            record = self.mstats.compile(self.pop)
            print(record)
            log_record = {"gen": gen, "nevals": len(invalid_ind)}
            for stat_name, stat_values in record.items():
                for field, value in stat_values.items():
                    log_record[f"{stat_name}_{field}"] = value
            self.logbook.record(**log_record)
            print(self.logbook.stream)

        log_df = pd.DataFrame(self.logbook)
        if not os.path.exists('results'):
            os.makedirs('results')

        log_df.to_csv(f'results/evolution_statistics_route_{self.origin}_to_{self.destination}_{popsize}ind_{ngen}gen.csv', index=False)

        return  bestCh, bestChFitness