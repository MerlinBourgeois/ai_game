import random
import tensorflow as tf
import numpy as np
import gc

largeur = 800
hauteur = 600
# Paramètres du personnage
personnage_largeur = 50
personnage_hauteur = 50
personnage_x = 25
personnage_y = hauteur // 2 - personnage_hauteur // 2
personnage_vitesse = 5

# Paramètres des obstacles
obstacle_largeur = 100
obstacle_hauteur = 100
obstacle_x = largeur
obstacle_y = random.randint(0, hauteur - obstacle_hauteur)
obstacle_vitesse = 20
save_interval = 1000

# Modèle TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(6,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='linear')  # Utilisation de l'activation linéaire pour l'apprentissage par renforcement
])

# Optimiseur
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Mémoire de relecture
replay_memory = []

# Paramètres de l'apprentissage par renforcement
gamma = 0.99  # Facteur de remise
batch_size = 32  # Taille de l'échantillon pour l'apprentissage par lots

# Charger le modèle si un fichier de sauvegarde existe
model_path = "model_save.h5"
try:
    model = tf.keras.models.load_model(model_path)
    print("Le modèle a été chargé avec succès.")
except (OSError, IOError):
    print("Le modèle n'a pas pu être chargé. Un nouveau modèle sera utilisé.")

# Fonction de récompense
def get_reward(distance, survived_time):
    collision_penalty = -100  # Pénalité pour la collision avec un obstacle
    distance_reward = 10 * (1 - distance)  # Récompense basée sur la distance par rapport à l'obstacle
    #time_reward = survived_time  # Récompense basée sur la durée de survie

    reward = collision_penalty + distance_reward #+ time_reward
    return reward

# Exploration-exploitation
def exploration_exploitation(action, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 2)  # Exploration aléatoire
    else:
        return action  # Exploitation du modèle

# Prédire l'action en fonction de l'état actuel
def prendre_decision(x, y, obstacle_x, obstacle_y):
    etat = np.array([x / largeur, y / hauteur, obstacle_x / largeur, obstacle_y / hauteur,
                     personnage_x / largeur, personnage_y / hauteur])
    action = np.argmax(model.predict(etat.reshape(1, 6))[0])
    return exploration_exploitation(action, epsilon)

# Boucle principale de l'entraînement
running = True
iterations = 1  # Nombre d'itérations d'apprentissage supplémentaires

survived_time = 0  # Temps de survie du personnage

# Paramètres pour la libération de mémoire
memory_clear_interval = 1000  # Intervalles d'itération avant de libérer la mémoire
iteration_count = 0  # Compteur d'itérations

epsilon = 1.0  # Paramètre d'exploration initial
epsilon_decay = 0.999  # Taux de décroissance de l'exploration
epsilon_min = 0.01  # Valeur minimale de l'exploration

while running:
    for _ in range(iterations):
        # Gestion des contrôles du personnage
        action = prendre_decision(personnage_x, personnage_y, obstacle_x, obstacle_y)
        if action == 0 and personnage_y > 0:
            personnage_y -= personnage_vitesse
        elif action == 1 and personnage_y < hauteur - personnage_hauteur:
            personnage_y += personnage_vitesse

        # Déplacement de l'obstacle
        obstacle_x -= obstacle_vitesse
        if obstacle_x < -obstacle_largeur:
            obstacle_x = largeur
            obstacle_y = random.randint(0, hauteur - obstacle_hauteur)

        # Collision entre le personnage et l'obstacle
        if obstacle_x < personnage_x + personnage_largeur and obstacle_x + obstacle_largeur > personnage_x \
                and obstacle_y < personnage_y + personnage_hauteur and obstacle_y + obstacle_hauteur > personnage_y:
            # Relancer la partie en réinitialisant les positions
            personnage_x = 25
            personnage_y = hauteur // 2 - personnage_hauteur // 2
            obstacle_x = largeur
            obstacle_y = random.randint(0, hauteur - obstacle_hauteur)

        # Enregistrement de l'expérience dans la mémoire de relecture
        distance = abs(obstacle_x - personnage_x) / largeur
        reward = get_reward(distance, survived_time)
        etat = np.array([personnage_x / largeur, personnage_y / hauteur, obstacle_x / largeur, obstacle_y / hauteur,
                         personnage_x / largeur, personnage_y / hauteur])
        experience = (etat, action, reward)
        replay_memory.append(experience)

        # Mise à jour du modèle avec l'apprentissage par renforcement
        if len(replay_memory) >= batch_size:
            batch = random.sample(replay_memory, batch_size)
            etats = np.array([exp[0] for exp in batch])
            actions = np.array([exp[1] for exp in batch])
            rewards = np.array([exp[2] for exp in batch])

            next_states = np.array([exp[0] for exp in replay_memory[-batch_size:]])
            next_qs = model.predict(next_states)
            max_next_qs = np.max(next_qs, axis=1)

            targets = rewards + gamma * max_next_qs

            with tf.GradientTape() as tape:
                q_values = model(etats)
                action_masks = tf.one_hot(actions, depth=3)
                selected_q_values = tf.reduce_sum(action_masks * q_values, axis=1)
                loss = tf.reduce_mean(tf.square(targets - selected_q_values))

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Exploration-exploitation : décroissance de l'exploration
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Libérer la mémoire à chaque intervalle défini
        if iteration_count % memory_clear_interval == 0:
            replay_memory = []
            tf.keras.backend.clear_session()
            gc.collect()
            print("Mémoire libérée à l'itération", iteration_count)

        iteration_count += 1

        # Sauvegarder le modèle à chaque intervalle défini
        if iteration_count % save_interval == 0:
            model.save(model_path)
            print("Modèle sauvegardé à l'itération", iteration_count)

    survived_time += 1  # Incrémente le temps de survie du personnage
