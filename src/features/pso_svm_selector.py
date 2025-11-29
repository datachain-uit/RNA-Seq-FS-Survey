import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class PSOSVMSelector:

    def __init__(
        self,
        pop_size=20,
        num_generations=30,
        w=0.7,
        c1=1.5,
        c2=1.5,
        test_size=0.2,
        random_state=42
    ):
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.test_size = test_size
        self.random_state = random_state

        self.num_evals = 0
        self.best_mask = None
        self.best_score = 0.0

    def fit_transform(self, X, y):

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame for PSO-SVM FS.")

        X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        num_features = X_train_fs.shape[1]
        rng = np.random.default_rng(self.random_state)
        random.seed(self.random_state)

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        def create_particle():
            pos = rng.integers(0, 2, size=num_features).tolist()
            vel = rng.uniform(-1, 1, size=num_features).tolist()
            return {
                "position": pos,
                "velocity": vel,
                "pbest": pos[:],
                "pbest_score": 0.0
            }

        def fitness(position):
            mask = np.array(position, dtype=bool)

            if np.sum(mask) == 0:
                return 0.0

            self.num_evals += 1

            X_tr_sel = X_train_fs.iloc[:, mask]
            X_te_sel = X_test_fs.iloc[:, mask]

            clf = SVC(C=0.1, kernel="linear", random_state=self.random_state)
            clf.fit(X_tr_sel, y_train_fs)
            return float(clf.score(X_te_sel, y_test_fs))

        swarm = [create_particle() for _ in range(self.pop_size)]
        gbest = swarm[0]["position"]
        gbest_score = 0.0

        for _ in range(self.num_generations):

            for particle in swarm:
                score = fitness(particle["position"])

                if score > particle["pbest_score"]:
                    particle["pbest_score"] = score
                    particle["pbest"] = particle["position"][:]

                if score > gbest_score:
                    gbest_score = score
                    gbest = particle["position"][:]

            for i, particle in enumerate(swarm):
                for d in range(num_features):
                    r1, r2 = random.random(), random.random()

                    inertia = self.w * particle["velocity"][d]
                    cognitive = self.c1 * r1 * (particle["pbest"][d] - particle["position"][d])
                    social = self.c2 * r2 * (gbest[d] - particle["position"][d])

                    v_new = inertia + cognitive + social
                    particle["velocity"][d] = v_new

                    particle["position"][d] = 1 if sigmoid(v_new) > random.random() else 0

        mask = np.array(gbest, dtype=bool)
        self.best_mask = mask

        X_selected = X.loc[:, mask]
        return X_selected, mask, self.num_evals
