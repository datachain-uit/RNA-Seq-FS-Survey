import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class GASVMSelector:

    def __init__(self,
                 pop_size=20,
                 num_generations=30,
                 elite_size=2,
                 mutation_rate=0.01,
                 test_size=0.2,
                 random_state=42):

        self.pop_size = pop_size
        self.num_generations = num_generations
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.test_size = test_size
        self.random_state = random_state

        self.best_mask = None
        self.num_evals = 0
        self.best_score = 0.0

    def fit_transform(self, X, y):

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame for GA-SVM.")

        X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        num_features = X_train_fs.shape[1]
        rng = np.random.default_rng(self.random_state)
        random.seed(self.random_state)

        def create_individual():
            position = rng.integers(0, 2, size=num_features).tolist()
            return {"position": position, "fitness": 0.0}

        def fitness(position):

            mask = np.array(position, dtype=bool)

            if np.sum(mask) == 0:
                return 0.0

            self.num_evals += 1

            X_tr_sel = X_train_fs.iloc[:, mask]
            X_te_sel = X_test_fs.iloc[:, mask]

            clf = SVC(C=0.1, kernel='linear', random_state=self.random_state)
            clf.fit(X_tr_sel, y_train_fs)
            return float(clf.score(X_te_sel, y_test_fs))

        def crossover(parent1, parent2):
            point = rng.integers(1, num_features - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2

        def mutate(position):
            return [
                bit if random.random() > self.mutation_rate else 1 - bit
                for bit in position
            ]

        population = [create_individual() for _ in range(self.pop_size)]
        gbest_position = population[0]["position"]
        gbest_score = 0.0

        for gen in range(self.num_generations):
            for ind in population:
                fit = fitness(ind["position"])
                ind["fitness"] = fit
                if fit > gbest_score:
                    gbest_score = fit
                    gbest_position = ind["position"][:]

            elites = sorted(population, key=lambda x: x["fitness"], reverse=True)[: self.elite_size]

            new_pop = elites[:]
            while len(new_pop) < self.pop_size:
                parents = random.sample(population, 2)
                c1, c2 = crossover(parents[0]["position"], parents[1]["position"])
                c1 = mutate(c1)
                c2 = mutate(c2)
                new_pop.append({"position": c1, "fitness": 0.0})
                if len(new_pop) < self.pop_size:
                    new_pop.append({"position": c2, "fitness": 0.0})

            population = new_pop[: self.pop_size]

        mask = np.array(gbest_position, dtype=bool)
        self.best_mask = mask

        X_sel = X.loc[:, mask]
        return X_sel, mask, self.num_evals
