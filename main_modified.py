from agent_modified import Agent
import matplotlib.pyplot as plt

EPISODE_LIMIT = 250
NO_RUNS = 100
def main():
    scores = []
    average = []
    i = 0
    # attempts to run the agent for 100 trials, if one fails calculates next
    # trial without losing all previous trials
    while i < NO_RUNS:
        print(f"trial {i}")
        agent = Agent()
        try:
            agent.train()
        except ValueError as e:
            continue
        except:
            print('ValueError')
            continue
        trial = agent.average_reward
        if len(agent.average_reward) > EPISODE_LIMIT:
            trial = trial[0:EPISODE_LIMIT]
        scores.append(trial)
        i += 1

    for trial in range(EPISODE_LIMIT):
        total = 0
        no_runs = 0
        for run in scores:
            if len(run) <= trial:
                continue 
            total += run[trial]
            no_runs += 1
        average.append(total/no_runs)
    
    print(f"Score: {scores} \nAverage: {average}")
    plt.clf()
    plt.plot(range(EPISODE_LIMIT), average, marker='', linewidth=0.8, alpha=0.9, label='Reward')
    plt.title("CartPole", fontsize=14)
    plt.xlabel("episode", fontsize=12)
    plt.ylabel("score", fontsize=12)

    plt.savefig('average_score.png')

if __name__ == '__main__':
    main()