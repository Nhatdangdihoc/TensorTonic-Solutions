def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    num_states = len(values)
    new_values = []

    for s in range(num_states):
        q_values = []

        for a in range(len(rewards[s])):
            # R(s,a) + γ · Σ T(s,a,s') · V(s')
            future = sum(transitions[s][a][s_prime] * values[s_prime]
                        for s_prime in range(num_states))
            
            q_values.append(rewards[s][a] + gamma * future)

        new_values.append(max(q_values))

    return new_values