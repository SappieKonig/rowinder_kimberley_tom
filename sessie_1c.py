import gym
env =gym.make("CartPole-v0")
#for x in range(10):
env.reset()
done= False

#totaal over variabelen heen
observation_total = []
reward_total = []
is_done_total = []
info_total = []

for run in range(3):
    env.reset()
    #variabele totalen per spel
    observations = []
    rewards = []
    is_dones = []
    infos = []

    while 1:
        observation, reward, is_done, info= env.step(0)
        #env.render()
        #voeg uit env gekomen waarde toe aan list van dit spel
        observations.append(observation)
        rewards.append(reward)
        is_dones.append(is_done)
        infos.append(info)
        if is_done:
            observation_total.append(observations)
            reward_total.append(rewards)
            is_done_total.append(is_dones)
            infos.append(infos)
            break
    #print(run)
print(reward_total)