import gym
from mltools import ga

k = 4
pop_size = 10
input_size = 4
output_size = 1

env = gym.make('CartPole-v1')

solver = ga.TournSelect(pop_size, input_size, output_size, k)

# Get the initial generated population
pop = solver.get_pop()

for epoch in range(100):
    fit_scores = {}

    # Run simulations
    for net_id, net in pop.items():
        obs = env.reset()

        # Fitness is the sum of rewards during the evaluation
        sum_reward = 0

        for _ in range(5000):
            #if epoch == 9:
            #env.render()

            action = net.activate( obs )[0]
            obs, reward, done, info = env.step( action )

            sum_reward += reward

            if done:
                break

        fit_scores[net_id] = sum_reward

    # Get the next generation based on evaluated fitness scores
    pop = solver.next_gen(fit_scores)

    print ( 'epoch {0}: top fitness - {1}'.format(epoch, max(fit_scores.values())) )
