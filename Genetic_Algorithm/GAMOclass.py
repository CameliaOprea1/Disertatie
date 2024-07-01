import osmnx as ox
from deap import base, creator, tools
ox.settings.log_console=True
ox.settings.use_cache=True
from numpy.random import choice
from smart_mobility_utilities.common import randomized_search
import functools
import numpy as np
from GAclass import GeneticAlgorithm
import random
import asyncio
from datasets_creation import load_my_graph
import os
import pandas as pd
''' 
In this script I define the generational process making use of DEAP module.

'''
G, _ =  load_my_graph()
class GAMO:
    

    def __init__(self,successors_dict, origin,destination,popsize=50):
        self.G=  G
        self.ga =  GeneticAlgorithm(successors_dict, source=origin, destination=destination)
        creator.create("FitnessMax", base.Fitness, weights=(-1.0, -1.0,-1.0))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        
        self.origin = origin
        self.destination = destination
        
        self.toolbox.register("individual", tools.initIterate, creator.Individual,
                              functools.partial(randomized_search, G=self.G, source=self.origin, destination=self.destination))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.ga.fitnessEvaluation_google)
        self.toolbox.register("mate",  self.ga.crossover)
        self.toolbox.register("mutate", self.ga.mutation)
        self.toolbox.register("select", tools.selNSGA2)
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
        # Initialize the logbook
        self.logbook = tools.Logbook()
        self.logbook.header = ["gen", "nevals"]
        for stat_name, stat_obj in self.mstats.items():
            self.logbook.header.extend(f"{stat_name}_{field}" for field in stat_obj.fields)


    async def run(self, ngen=50, mutation_rate=0.5, crossover_rate=0.5):

        popsize = len(self.pop)
        invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        fitnesses = await asyncio.gather(*[self.toolbox.evaluate(ind) for ind in invalid_ind])

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        self.pop = self.toolbox.select(self.pop, popsize)
        self.hof.update(self.pop)  # best 10 chromosomes
        bestCh = tools.selBest(self.pop, 1)  # best chromosome

        # Begin the generational process
        for gen in range(1, ngen + 1):

            offspring = self.toolbox.select(self.pop, popsize) 
            offspring = [self.toolbox.clone(individual) for individual in offspring]
            
            # CROSSOVER
            for i in range(1, len(offspring), 2):
                if random.random() < crossover_rate:
                    self.toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values

            # MUTATION
            # Apply mutation
            for i in range(len(offspring)):
                if random.random() < mutation_rate:
                    self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
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

        log_df.to_csv(f'results/evolution_statistics_route_{self.origin}_to_{self.destination}.csv', index=False)


        return  bestCh, bestChFitness