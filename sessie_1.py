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

loss_total=[]

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

def decay_min_valuehead_normalized(total_rewards, value_head, decay_factor):
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
    total_rewards_2 = [a - b for a, b in zip(total_rewards, value_head)]

    for i, rewards in enumerate(total_rewards_2):
        total_rewards[i] = decay(rewards, decay_factor)


    total_rewards = np.concatenate(total_rewards)

    return (total_rewards - np.mean(total_rewards)) / np.std(total_rewards)

observation = tf.keras.layers.Input(4)
X = tf.keras.layers.Dense(48, "relu")(observation)
X = tf.keras.layers.Dense(48, "relu")(X)
#X = tf.keras.layers.Dense(48, "relu")(X)
output = tf.keras.layers.Dense(1, "sigmoid")(X)
output_2=tf.keras.layers.Dense(1)(X)

model = tf.keras.models.Model(inputs=[observation], outputs=[output,output_2])
#model.compile(loss='mse', optimizer='adam')
optimizer = tf.keras.optimizers.Adam(3e-3)

for epoch in range(2):

    #totaal over variabelen heen
    observation_total = []
    reward_total = []
    is_done_total = []
    info_total = []
    step_values_total = []
    actions_total=[]
    predicted_values_before_random_total=[]
    value_head=[]


    for run in range(2):
        #variabele totalen per spel
        observations = []
        rewards = []
        is_dones = []
        infos = []
        step_values = []

        #step_values.append(0)

        predicted_values_before_random=[]
        pre_value_head=[]

        #predicted_values_before_random.append(0)

        observation = env.reset()
        observation.reshape(1, 4)
        reward=[1]
        is_done=False
        info=[]

        while 1:
            #nader te bepalen comment
            observations.append(observation)
            rewards.append(reward)
            is_dones.append(is_done)
            infos.append(info)

            predicted_waarde = model.predict(observation.reshape((1, -1)))[0][0]
            predicted_waarde_2 = model.predict(observation.reshape((1, -1)))[1][0]
            print(predicted_waarde,predicted_waarde_2)
            # predicted_waarde_action=np.round(predicted_waarde).astype(int)
            predicted_waarde_action = 0 if random.random() > predicted_waarde else 1
            #print(predicted_waarde_action)
            observation, reward, is_done, info = env.step(predicted_waarde_action)
            # env.render()


            step_values.append(predicted_waarde_action)
            predicted_values_before_random.append(predicted_waarde)
            pre_value_head.append(predicted_waarde_2)

            if is_done:
                #observations.append(observation)
                #rewards.append(reward)
                #is_dones.append(is_done)
                #infos.append(info)

                observation_total.append(observations)
                reward_total.append(rewards)
                is_done_total.append(is_dones)
                infos.append(infos)
                actions_total.append(step_values)
                predicted_values_before_random_total.append(predicted_values_before_random)
                value_head.append(pre_value_head)

                break
        #print(run)
    '''
    #nog een keer overwegen om het per reeks op een losse regel te printen
    print('reward_total')
    print(reward_total)
    print(is_done_total) #testen of het klopt dat we ook een reward=1 krijgen indien we is_done zijn >.> klopt, code erboven is niet gek
    print()
    print('decayed waardes')
    print(decay_loose(reward_total,.97))
    print()
    print('genormaliseerde decayed waardes')
    '''
    new_decayed_reward = decay_min_valuehead_normalized(reward_total, value_head, .97)
    decay_and_normalized_rewards=decay_and_normalize(reward_total,.97)

    #print(decay_and_normalized_rewards)

    #print('actions_total - genomen stappen')
    #for a in actions_total:
    #    print(a)
    #print()
    '''
    print('observation_total')
    for b in observation_total:
        print(b)
    '''

    #zet format arrays om
    observations_reshaped=np.concatenate(observation_total)
    actions_reshaped=np.concatenate(actions_total)
    predicted_values_reshaped=np.concatenate(predicted_values_before_random_total)
    reward_reshaped=np.concatenate(reward_total)
    #print(reward_reshaped)

    '''
    for lengte_obs in range(len(observations_reshaped)):
        print() #witregel
        print('(zero based) observation: ' + str(lengte_obs))  # witregel
        print(observations_reshaped[lengte_obs]) #obs
        print(actions_reshaped[lengte_obs]) #genomen stappen
        print(predicted_values_reshaped[lengte_obs]) #input voor rolled values
        print(decay_and_normalized_rewards[lengte_obs]) #rewards decayed and normalized
        print(reward_reshaped[lengte_obs]) #rewards decayed niet normalized
        '''

    with tf.GradientTape() as tape:
        predictions = model(observations_reshaped)
        predictions = tf.keras.backend.clip(predictions, 1e-7, 1 - 1e-7)
        #print(actions_reshaped.shape())
        #print(predictions.shape())
        #loss = tf.keras.losses.categorical_crossentropy(actions_reshaped, predictions) * decay_and_normalized_rewards
        loss = tf.keras.losses.mse(actions_reshaped, predictions)*decay_and_normalized_rewards
        print()
        #print(loss)
        loss_total.append(loss.shape)
        print(loss.shape)
    train_vars = model.trainable_variables
    grads = tape.gradient(loss, train_vars)
    optimizer.apply_gradients(zip(grads, train_vars))

    #new_decayed_reward=decayed_reward