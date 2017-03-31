import copy
import numpy as np
import random
from nn import NeuralNet

class TournSelect (object):
    def __init__ (self, pop_size, input_size, output_size, k):
        # Start an id iterator to track new children in population
        self.id_counter = pop_size
        self.k = k

        # Generate initial population (id:net)
        self.pop = {i: NeuralNet(input_size,0,output_size) for i in range(pop_size)} 

    def get_pop(self):
        return self.pop

    def next_gen (self, fit_scores):
        # Error checking
        num_creats = len( self.pop.keys() )
        if self.k > num_creats:
            self.k = num_creats

        k_groups = [[] for i in range(self.k)]

        # Split creatures into categories
        for i, creature in enumerate(self.pop.iteritems()):
            k_groups[ i % self.k ].append( creature )

        # New empty pop
        newpop = {}

        for i, group in enumerate(k_groups):
            # Randomly choose parents
            idx_choices = np.random.choice([i for i in range(len(group))], 2, replace=True)
            p1 = group[ idx_choices[0] ][1]
            p2 = group[ idx_choices[1] ][1]

            # Make children
            c1 = self.crossover(p1, p2)
            self.mutate(c1)
            c2 = self.crossover(p1, p2)
            self.mutate(c2)

            # Get sorted fitnesses of group (least fit to most)
            group_ids = [i[0] for i in group]
            group_fits = [ (net_id, fit_scores[net_id]) for net_id in group_ids]
            group_fits.sort(key=lambda tup: tup[1])

            # Replace 2 least fit with children
            # Add all nets in group to pop
            id1 = group_fits[0][0]
            id2 = group_fits[1][0]
            for x in group:
                if (x[0] == id1):
                    x = (self.id_counter, c1) # Replace net with c1
                    self.id_counter += 1
                elif (x[0] == id2):
                    x = (self.id_counter, c2) # Replace net with c2
                    self.id_counter += 1

                newpop[ x[0] ] = x[1]

        self.pop = newpop
        return self.pop

    def crossover(self, p1, p2):
        assert (p1.in_size == p2.in_size), "Parent net input sizes don't match"
        assert (p1.out_size == p2.out_size), "Parent net output sizes don't match"

        # Random submatrix for crossover
        top_left_x_point = (random.randint(0, int(p1.in_size*.75)),
                            random.randint(0, p1.out_size-1))
        bot_right_x_point = (random.randint(top_left_x_point[0], p1.in_size-1),
                             random.randint(top_left_x_point[1], p1.out_size-1))

        # Make child
        c = copy.deepcopy(p1)

        # Perform crossover
        # Replace random submatrix of p1 to p2
        for i in range(top_left_x_point[0], bot_right_x_point[0]+1):
            for j in range(top_left_x_point[1], bot_right_x_point[1]+1):
                c.w_inp_out[i][j] = p2.w_inp_out[i][j]

        return c

    def mutate (self, net):
        mutate_prob = .2

        # Generate random matrix for perturbation chance
        rand_choice = np.random.rand(net.in_size, net.out_size)

        # Random chance to mutate
        for i in range(net.in_size ):
            for j in range(net.out_size ):
                if (rand_choice[i][j] < mutate_prob):
                    net.w_inp_out[i][j] = random.random()

