from abc import ABC, abstractmethod
import numpy as np

class NSGA2(ABC):
    def __init__(self, num_generations=100):
        """
        初始化 NSGA-II 算法框架
        :param num_generations: 最大代数
        """
        self.num_generations = num_generations
        self.init_pop = self.initialize_population()
        self.pop_size = self.init_pop.shape[0]
        self.var_num = self.init_pop.shape[1]
    
    def run(self):
        """
        运行 NSGA-II 算法
        :return: 最终的帕累托前沿
        """
        population = self.initialize_population()
        for generation in range(self.num_generations):
            # 评估种群
            fitness = self.evaluate_population(population)

            # 快速非支配排序
            fronts = self.fast_non_dominated_sort(fitness)

            # 计算拥挤距离
            for front in fronts:
                self.calculate_crowding_distance(front, fitness)

            # 选择操作
            mating_pool = self.select(population, fitness)

            # 交叉操作
            offspring = self.crossover(mating_pool)

            # 变异操作
            offspring = self.mutate(offspring)

            # 合并父代和子代，选择下一代
            combined_population = np.vstack((population, offspring))
            combined_fitness = self.evaluate_population(combined_population)
            sorted_fronts = self.fast_non_dominated_sort(combined_fitness)
            population = self.select_next_generation(combined_population, sorted_fronts)

        # 返回最终的帕累托前沿
        fitness = self.evaluate_population(population)
        return self.fast_non_dominated_sort(fitness)[0]
    


    @abstractmethod
    def initialize_population(self):
        """
        初始化种群，子类需要实现此方法。
        :return: 初始化的种群
        """
        pass

    @abstractmethod
    def evaluate_population(self, population):
        """
        评估种群的目标函数值，子类需要实现此方法。
        :param population: 种群
        :return: 目标函数值矩阵
        """
        pass

    def _evaluate(self, population):
        pass


    @abstractmethod
    def crossover(self, parents):
        """
        执行交叉操作，子类需要实现此方法。
        :param parents: 父代种群
        :return: 子代种群
        """
        pass

    @abstractmethod
    def mutate(self, offspring):
        """
        执行变异操作，子类需要实现此方法。
        :param offspring: 子代种群
        :return: 变异后的子代种群
        """
        pass

    @abstractmethod
    def select(self, population, fitness):
        """
        执行选择操作，子类需要实现此方法。
        :param population: 种群
        :param fitness: 适应度值（目标函数值）
        :return: 选择后的种群
        """
        pass

    def fast_non_dominated_sort(self, fitness):
        """
        快速非支配排序算法
        :param fitness: 种群的目标函数值矩阵
        :return: 非支配排序后的前沿列表
        """
        fronts = [[]]
        domination_count = [0] * len(fitness)
        dominated_solutions = [[] for _ in range(len(fitness))]

        for i in range(len(fitness)):
            for j in range(i + 1, len(fitness)):
                if self.dominates(fitness[i], fitness[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self.dominates(fitness[j], fitness[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        for i in range(len(fitness)):
            if domination_count[i] == 0:
                fronts[0].append(i)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]  # 返回所有非支配前沿

    def dominates(self, solution1, solution2):
        """
        判断解1是否支配解2
        :param solution1: 解1的目标函数值
        :param solution2: 解2的目标函数值
        :return: True 如果解1支配解2，False 否则
        """
        return all(a <= b for a, b in zip(solution1, solution2)) and any(a < b for a, b in zip(solution1, solution2))

    def calculate_crowding_distance(self, front, fitness):
        """
        计算拥挤距离，子类需要实现此方法。
        :param front: 当前非支配前沿
        :param fitness: 适应度值（目标函数值）
        :return: 拥挤距离数组
        """
        pass

    def select_next_generation(self, population, fronts):
        """
        基于非支配排序和拥挤距离选择下一代
        :param population: 合并后的种群
        :param fronts: 排序后的前沿
        :return: 选择后的种群
        """
        selected_population = []
        for front in fronts:
            if len(selected_population) + len(front) <= self.population_size:
                selected_population.extend(front)
            else:
                # 拥挤距离选择剩余个体
                break
        return np.array(selected_population)
