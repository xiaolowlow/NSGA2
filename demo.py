from nsga2 import NSGA2
import numpy as np

class MyNSGA2(NSGA2):
    def initialize_population(self):
        pop_size = 10
        var_num = 5
        population = np.random.rand(pop_size, var_num)
        return population

    def evaluate_population(self, population):
        def zdt1(x):
            f1 = x[0]
            g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
            f2 = g * (1 - (f1 / g)**0.5)
            return [f1, f2]
        fitness = [zdt1(ind) for ind in population]
        return fitness

    def crossover(self):
        pass

    def mutate(self, offspring):
        pass

    def select(self, population, fitness):
        pass


mynasga2 = MyNSGA2()
print(mynasga2.init_pop)
print(mynasga2.evaluate_population(mynasga2.init_pop))