#regel 86 tm 91 een keer sterk verbeteren
#heb er volgens mij geen random shit inzitten om een epsilon na te jagen

# regel 102 klopt niet helemaal (infos.append(infos)). infos boeit gelukkig niet ;) Het is nog een beetje een rotzooitje maar het klopt allemaal wel
# jullie eigen commentaar zegt eigenlijk alles al
import numpy as np
import gym
import tensorflow as tf
import random

env =gym.make("CartPole-v0")

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
step_values_total = []

observation = tf.keras.layers.Input(4)
X = tf.keras.layers.Dense(12, "relu")(observation)
X = tf.keras.layers.Dense(48, "relu")(X)
X = tf.keras.layers.Dense(48, "relu")(X)
output = tf.keras.layers.Dense(1, "sigmoid")(X)


model = tf.keras.models.Model(inputs=[observation], outputs=[output])
model.compile(loss='mse', optimizer='adam')


for run in range(3):
    #variabele totalen per spel
    observations = []
    rewards = []
    is_dones = []
    infos = []
    step_values = []
    observation = env.reset()
    observation.reshape(1, 4)

    while 1:
        predicted_waarde = model.predict(observation.reshape((1, -1)))[0][0]
        # predicted_waarde_action=np.round(predicted_waarde).astype(int)
        predicted_waarde_action = 0 if random.random() > predicted_waarde else 1
        print(predicted_waarde_action)
        observation, reward, is_done, info = env.step(predicted_waarde_action)
        # env.render()
        # voeg uit env gekomen waarde toe aan list van dit spel
        observations.append(observation)
        rewards.append(reward)
        is_dones.append(is_done)
        infos.append(info)
        step_values.append(predicted_waarde_action)

        if is_done:
            observation_total.append(observations)
            reward_total.append(rewards)
            is_done_total.append(is_dones)
            infos.append(infos)
            step_values_total.append(step_values)
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

print('step_values_total')
for a in step_values_total:
    print(a)

