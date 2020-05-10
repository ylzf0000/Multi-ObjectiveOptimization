import math


class TestFunction:
    def __init__(self, objs, constraints, bounds):
        self.objs = objs
        self.constraints = constraints
        self.bounds = bounds


BinhAndKornfunction = TestFunction(
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
ChankongAndHaimesfunction = TestFunction(
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
FonsecaFlemingfunction = TestFunction(
    objs=[
        lambda x: 1 - math.exp(-(x[0] - 1 / math.sqrt(2)) ** 2 - (x[1] - 1 / math.sqrt(2)) ** 2),
        lambda x: 1 - math.exp(-(x[0] + 1 / math.sqrt(2)) ** 2 - (x[1] + 1 / math.sqrt(2)) ** 2),
    ],
    constraints=[
    ],
    bounds=[[-4, 4], [-4, 4]],
)
Testfunction4 = TestFunction(
    objs=[
        lambda x: x[0] ** 2 - x[1],
        lambda x: -0.5 * x[0] - x[1] - 1,
    ],
    constraints=[
        lambda x: 6.5 - x[0] / 6 - x[1] >= 0,
        lambda x: 7.5 - 0.5 * x[0] - x[1] >= 0,
        lambda x: 30 - 5 * x[0] - x[1] >= 0,
    ],
    bounds=[[-7, 4], [-7, 4]],
)
SchafferfunctionN = TestFunction(
    objs=[
        lambda x: x[0] ** 2,
        lambda x: (x[0] - 2) ** 2,
    ],
    constraints=[
    ],
    bounds=[[-10e5, 10e5]],
)
