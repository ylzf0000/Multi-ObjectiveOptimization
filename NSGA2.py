# Importing required modules
import math
import random
import matplotlib.pyplot as plt
import numpy as np


# 默认为最小化问题

def recombination(a, b):
    return (a + b) / 2


def clamp(a, l, r):
    if a < l:
        a = l
    elif a > r:
        a = r
    return a


def mutation(a, bounds):
    rs = [random.random() for i in range(len(bounds))]
    for i in range(len(a)):
        a[i] = bounds[i][0] + (bounds[i][1] - bounds[i][0]) * rs[i]
        # a[i] = clamp(a[i], bounds[i][0], bounds[i][1])
    return a


# 检测是否a支配b
def is_dominate(obj_a, obj_b):
    less = False
    for i in range(len(obj_a)):
        fa, fb = obj_a[i], obj_b[i]
        if fa > fb:
            return False
        if fa < fb:
            less = True
    return less


def crowding_distance(pop, fronts, obj_vals):
    dist = [0 for i in range(0, pop.shape[0])]
    obj_vals = obj_vals.T  # obj_vals[i][j]表示第i个目标函数在第j个个体的函数值
    for obj_val in obj_vals:
        f_max = max(obj_val)
        f_min = min(obj_val)
        delta = f_max - f_min
        for f in fronts:
            t = f[:]
            t.sort(key=lambda i: obj_val[i])
            dist[t[0]] = float("inf")
            dist[t[-1]] = float("inf")
            for i in range(1, len(t) - 1):
                dist[t[i]] += (obj_val[t[i + 1]] - obj_val[t[i - 1]]) / delta
    return dist


def non_dominated_sort(pop, obj_vals):
    pop_size = pop.shape[0]
    # S[i]存放被第i个个体支配的个体的索引集合
    S = [[] for i in range(0, pop_size)]
    # n[i]表示支配第i个个体的个体数量
    n = [0 for i in range(0, pop_size)]
    # fronts[i]表示第i层的个体索引
    fronts = [[]]
    # ranks[i]表示第i个个体是第几层
    ranks = [0 for i in range(0, pop_size)]
    for pi in range(0, pop_size):
        for qi in range(0, pop_size):
            # p, q = pop[pi], pop[qi]
            if is_dominate(obj_vals[pi], obj_vals[qi]):
                S[pi].append(qi)
            elif is_dominate(obj_vals[qi], obj_vals[pi]):
                n[pi] += 1
        if n[pi] == 0:
            ranks[pi] = 0
            # if pi not in fronts[0]:
            fronts[0].append(pi)
    i = 0
    while len(fronts[i]) > 0:
        Q = []
        for pi in fronts[i]:
            for qi in S[pi]:
                n[qi] -= 1
                if n[qi] == 0:
                    ranks[qi] = i + 1
                    if qi not in Q:
                        Q.append(qi)
        i += 1
        fronts.append(Q)
    del fronts[-1]  # 最后一个面是空的
    return fronts


def sat_constraint(p, constraint):
    for c in constraint:
        if not c(p):
            return False
    return True


def get_front_vals(front, pop, num_objs, objs):
    front_vals = np.empty((len(front), num_objs))
    for i in range(len(front)):
        for oi in range(num_objs):
            front_vals[i][oi] = objs[oi](pop[front[i]])
    return front_vals


def get_obj_vals(pop, objs):
    pop_size, num_objs = pop.shape[0], len(objs)
    obj_vals = np.empty((pop_size, num_objs))
    for i in range(pop_size):
        for j in range(num_objs):
            obj_vals[i][j] = objs[j](pop[i])
    return obj_vals


def show_scatter(front_vals):
    f1 = [val[0] for val in front_vals]
    f2 = [val[1] for val in front_vals]
    plt.xlabel('F1', fontsize=15)
    plt.ylabel('F2', fontsize=15)
    plt.grid()
    plt.legend('Pareto front')
    plt.scatter(f1, f2)
    plt.show()


def main_loop(max_gen, pop_size, num_objs, objs, dim, bounds, constraint, mutation_rate):
    """
    :param max_gen: 种群最大代数
    :param pop_size: 种群大小
    :param num_objs: 目标函数数量
    :param objs: 目标函数list
    :param dim: 变量维度
    :param bounds: 变量定义域,bounds[i][0,1]表示第i个变量取值范围
    :param constraint: 约束条件函数
    :param mutation_rate: 变异率
    :return:
    """
    # 初始化种群
    front_vals = None
    pop = np.empty((pop_size, dim), dtype=np.float32)
    for i in range(pop_size):
        p = [bounds[j][0] + (bounds[j][1] - bounds[j][0]) * random.random()
             for j in range(dim)]
        while not sat_constraint(p, constraint):
            p = [bounds[j][0] + (bounds[j][1] - bounds[j][0]) * random.random()
                 for j in range(dim)]
        pop[i] = p

    # 开始迭代
    num_gen = 0
    while num_gen < max_gen:
        mix_pop = np.empty((pop_size * 2, dim), dtype=np.float32)
        mix_pop[:pop_size] = pop
        for i in range(pop_size, pop_size * 2):
            ai = random.randint(0, pop_size - 1)
            bi = random.randint(0, pop_size - 1)
            # ci = random.randint(0, pop_size - 1)
            mix_pop[i] = recombination(pop[ai], pop[bi])
            if random.random() < mutation_rate:
                mix_pop[i] = mutation(mix_pop[i], bounds)
            while not sat_constraint(mix_pop[i], constraint):
                mix_pop[i] = recombination(pop[ai], pop[bi])
                if random.random() < mutation_rate:
                    mix_pop[i] = mutation(mix_pop[i], bounds)
        # 为当前种群个体生成目标函数值
        obj_vals = get_obj_vals(mix_pop, objs)
        next_pop = []
        # 快速非支配排序
        fronts = non_dominated_sort(mix_pop, obj_vals)
        dist = crowding_distance(mix_pop, fronts, obj_vals)
        for f in fronts:
            size_next_pop = len(next_pop)
            size_f = len(f)
            if size_next_pop + size_f <= pop_size:
                for i in f:
                    next_pop.append(mix_pop[i])
            else:
                t = f[:]
                t.sort(key=lambda i: dist[i], reverse=True)
                i = 0
                while len(next_pop) < pop_size:
                    next_pop.append(mix_pop[t[i]])
                    i += 1
                break
        pop = np.array(next_pop)
        print(f'num_gen: {num_gen}')
        num_gen += 1
    # 返回第1个面的目标函数值
    front_vals = get_front_vals(non_dominated_sort(pop, get_obj_vals(pop, objs))[0], pop, num_objs, objs)

    return front_vals


class TestFunction:
    def __init__(self, objs, constraints, bounds):
        self.objs = objs
        self.constraints = constraints
        self.bounds = bounds


if __name__ == '__main__':
    max_gen = 100
    pop_size = 100
    BinhAndKornFunction = TestFunction(
        objs=[
            lambda x: 4 * x[0] ** 2 + 4 * x[1] ** 2,
            lambda x: (x[0] - 5) ** 2 + (x[1] - 5) ** 2,
        ],
        constraints=[
            lambda x: (x[0] - 5) ** 2 + x[1] ** 2 - 25 <= 0,
            lambda x: (x[0] - 8) ** 2 + (x[1] + 3) ** 2 - 7.7 >= 0,
        ],
        bounds=[[0, 5], [0, 5]],
    )
    ChankongAndHaimesFunction = TestFunction(
        objs=[
            lambda x: 2 + (x[0] - 2) ** 2 + (x[1] - 1) ** 2,
            lambda x: 9 * x[0] - (x[1] - 1) ** 2,
        ],
        constraints=[
            lambda x: x[0] ** 2 + x[1] ** 2 - 225 <= 0,
            lambda x: x[0] - 3 * x[1] + 10 <= 0,
        ],
        bounds=[[-20, 20], [-20, 20]],
    )
    FonsecaFlemingFunction = TestFunction(
        objs=[
            lambda x: 1 - math.exp(-(x[0] - 1 / math.sqrt(2)) ** 2 - (x[1] - 1 / math.sqrt(2)) ** 2),
            lambda x: 1 - math.exp(-(x[0] + 1 / math.sqrt(2)) ** 2 - (x[1] + 1 / math.sqrt(2)) ** 2),
        ],
        constraints=[
        ],
        bounds=[[-4, 4], [-4, 4]],
    )
    using_function = FonsecaFlemingFunction
    objs = using_function.objs
    constraints = using_function.constraints
    bounds = using_function.bounds
    mutation_rate = 0.6
    front_vals = main_loop(max_gen, pop_size, len(objs), objs, len(bounds), bounds, constraints, mutation_rate)
    show_scatter(front_vals)
