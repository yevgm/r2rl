import numpy as np


def to_one_hot(s, size):
    transformed_state = np.zeros(size * size)
    idx = s[0] * size + s[1]
    transformed_state[int(idx)] = 1
    return transformed_state


def evaluate_policy(env, model, render, env_idx, turns=3):
    scores = 0
    for j in range(turns):
        s, done, ep_r, steps = env.reset(), False, 0, 0
        while not done:
            # if it's maze, transform to one-hot representation
            if env_idx == 3:
                maze_size = env.observation_space.high[0] + 1
                s = to_one_hot(s, maze_size)

            # Take deterministic actions at test time
            a = model.select_action(s, deterministic=True)
            s_prime, r, done, info = env.step(a)
            ep_r += r
            steps += 1
            s = s_prime
            if render:
                env.render()
        scores += ep_r
    return int(scores / turns)


def evaluate_policy_shock(env, model, env_updater, shock_time=100, turns=3):
    scores = 0
    for j in range(turns):
        s, done, ep_r, steps = env.reset(), False, 0, 0
        env_updater.reset()
        step_cntr = 0
        while not done:
            # Take deterministic actions at test time
            a = model.select_action(s, deterministic=True)
            s_prime, r, done, info = env.step(a)
            ep_r += r
            steps += 1
            s = s_prime
            if step_cntr == shock_time:
                env_updater.update_environment(env)

            step_cntr += 1

        scores += ep_r
    return int(scores / turns)


