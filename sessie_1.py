import numpy as np
import gym
import tensorflow as tf

env =gym.make("CartPole-v0")
#for x in range(10):
env.reset()
done= False #gebruik ik lekker niet

def decay(rewards, decay_factor):
    """
    Berekent de echte rewards aan de hand van de verkregen rewards van een episode op elk tijdstip en een decay_factor

    :param rewards: een array/list met rewards per stap
    :param decay_factor: getal tussen 0 en 1 dat het belang van de toekomst aangeeft
    :return: een array met rewards waar de toekomst WEL in mee is genomen

    VB: decay([1, 0, 1], .9) --> [(1+0.9*0+0.9^2*1), (0+0.9*1), 1] --> [1.81, .9, 1]
    """
    decayed_rewards = np.zeros(len(rewards))
    decayed_rewards[-1] = rewards[-1]
    for i in reversed(range(len(rewards) - 1)):
        decayed_rewards[i] = rewards[i] + decay_factor * decayed_rewards[i + 1]
    return decayed_rewards

def decay_loose(reward_list, decay_factor):
    """
        Same als reguliere decay, maar los aan te roepen
    """
    decayed_reward_list=[]
    for rewards in reward_list:
        decayed_rewards = np.zeros(len(rewards))
        decayed_rewards[-1] = rewards[-1]
        for i in reversed(range(len(rewards) - 1)):
            decayed_rewards[i] = rewards[i] + decay_factor * decayed_rewards[i + 1]
        decayed_reward_list.append(decayed_rewards)
    return decayed_reward_list

def decay_and_normalize(total_rewards, decay_factor):
    """
    Past decay toe op een batch van episodes en normaliseert over het geheel
    Berekent per waarde distance SD van mean

    :param total_rewards: list van lists/arrays, waar de inner lists rewards bevatten
    :param decay_factor: getal tussen 0 en 1 dat het belang van de toekomst aangeeft
    :return: één nieuwe array met nieuwe rewards waar de toekomst in mee is genomen en die genormaliseerd is

    VB: decay_and_normalize([[0, 1], [1, 1, 1]], .9)
        eerst decay --> [[.9, 1], [2.71, 1.9, 1]]
        dan normaliseren --> [-0.85, -0.71, 1.71, 0.56, -0.71]
    """
    for i, rewards in enumerate(total_rewards):
        total_rewards[i] = decay(rewards, decay_factor)

    total_rewards = np.concatenate(total_rewards)

    return (total_rewards - np.mean(total_rewards)) / np.std(total_rewards)

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
        if len(is_dones) >500: #anders stop je niet als het ooit perfect is? xD
            is_done=True

        if is_done:
            observation_total.append(observations)
            reward_total.append(rewards)
            is_done_total.append(is_dones)
            infos.append(infos)
            break
    #print(run)

#nog een keer overwegen om het per reeks op een losse regel te printen
print('reward_total')
print(reward_total)
print(is_done_total) #testen of het klopt dat we ook een reward=1 krijgen indien we is_done zijn >.> klopt, code erboven is niet gek
print()
print('decayed waardes')
print(decay_loose(reward_total,.9))
print()
print('genormaliseerde decayed waardes')
print(decay_and_normalize(reward_total,.9))



