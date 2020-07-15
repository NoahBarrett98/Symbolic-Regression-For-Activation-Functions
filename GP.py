from deap import tools, base, gp, creator, algorithms
import deap
import operator
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

np.seterr(all='raise')

### HYPERPARAMS ###
NUM_STEPS = 10
POPULATION = 5
MAX_HEIGHT = 1

### MU-Lambda
MU = POPULATION
# number of offspring produced by pop (recombination pool)
LAMBDA = 250
CXPB = 0.5
MUTPB = 0.3

# intialize pop with min height 2 and max height 6
init_pop = (2, 6)

### PRIMITIVES ###
def protectedDiv(left, right):
    try:
        if right==0:
            return 1.0
        return left / right
    except FloatingPointError:
        return 1.0

def protected_sqrt(x):
    try:
        return np.sqrt(x)
    except FloatingPointError:
        return 1.0

def cos(x):
    return np.cos(x)

def sin(x):
    return np.sin(x)

def tan(x):
    return np.tan(x)

pset = deap.gp.PrimitiveSet("MAIN", 1)

pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
# pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
# exponential linear unit
pset.addPrimitive(K.elu, 1)
# relu
pset.addPrimitive(K.relu, 1)
# sigmoid
pset.addPrimitive(K.sigmoid, 1)
# tanh
pset.addPrimitive(K.tanh, 1)
pset.renameArguments(ARG0='x')

### Creator ###
# symbolic regression needs at least two object types
# an individual containing the genotype and a fitness

# this is a minimization problem so the weights is negative
creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))
# tree representation of individual
# from a list of expressions, make a prefix tree
creator.create("Individual", deap.gp.PrimitiveTree, fitness=deap.creator.FitnessMin)

### Toolbox ###
toolbox = base.Toolbox()
# create an individual using half and half
# half and half: Generate an expression with a PrimitiveSet pset. Half the time,
# the expression is generated with genGrow(), the other half, the expression is generated with genFull().
# this will produce a list of expressions which then can be converted to a primitive tree (prefix tree/Trie)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=init_pop[0], max_=init_pop[1])
# repeat individual initalizing
toolbox.register("individual", tools.initIterate, deap.creator.Individual, toolbox.expr)
# a population contains many individuals, created using gen half and half
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

### DL EVAL ###
# DL evaluation (function in simple conv) minimize loss

def activation_fitness(activation):

    activation = toolbox.compile(expr=activation)

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128),
      tf.keras.layers.Activation(activation),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10),
      tf.keras.layers.Softmax()
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    #index = int(fitness == 'loss')
    # return loss, we want to minimize loss
    return [model.evaluate(x_test,  y_test, verbose=2)[0]]

toolbox.register("evaluate", activation_fitness)

# tournament of size 3 for selection
toolbox.register("select", tools.selTournament, tournsize=3)
# one point crossover with uniform probability over all the nodes
toolbox.register("mate", gp.cxOnePoint)
# uniform probability mutation which may append a new full sub-tree to a node
# Randomly select a point in the tree individual,
# then replace the subtree at that point as a root by the expression generated using method expr().
# in this case our expression is genFull-> Generate an expression where each leaf has the same depth between min and max
toolbox.register("expr_mut", gp.genFull, min_=init_pop[0], max_=init_pop[1])
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# limit the height of generated individuals, done ot avoid bloat
# Koza reccomends max depth of 17
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT))

### statistics ###
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

### launch evolution ###
"""
When creating a population we want to ensure a diverse population:

Premature Convergence may occur if diversity is not maintained. This is when the convergence of the algorithm occurs prior
to finding the optimal solution.
This is caused by:
    1) loss of diversity: diversity refers to the amount of solutions and how different they are (distance between them)
    in this occurence all solutions resemble the best one.
    2) too strong selective pressure towards the best solution
    3) too much exploitation of hte existing building blocks from the current population 
        (i.e. recombining or not mutating them enough)

    There are various strategies for preventing premature convergence:
        i) Increasing population size. 
        ii) Uniform crossover. 
        iii) Replacement of similar individuals. 
        iv) Segmentations of individuals of similar fitness (known as fitness sharing) are some of them.
"""
pop = toolbox.population(n=POPULATION)

# structure that contains the best individuals
hof = tools.HallOfFame(5)

### DEAP algorithms:
### first select from pop
### use varand method to apply cross over and mutation
### evaluate the off spring -> in our case mse to determine its accuracy

pop, log = algorithms.eaSimple(pop,
                               toolbox,
                               CXPB,
                               MUTPB,
                               NUM_STEPS,
                               stats=mstats,
                               halloffame=hof,
                               verbose=True)

### view best solution ###
avg = log.chapters["fitness"].select("avg")
min = log.chapters["fitness"].select("min")
max = log.chapters["fitness"].select("max")
fig, ax =plt.subplots(1,1, figsize=(5,5))
ax.plot(avg, color="k", label='avg')
ax.plot(min, color="b", label='min')
ax.plot(max, color="r", label='max')
ax.set_title("Fitness")
plt.legend()
plt.show()