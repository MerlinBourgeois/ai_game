import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

largeur = 800
hauteur = 600

# Paramètres du personnage
personnage_largeur = 50
personnage_hauteur = 50
personnage_x = 25
personnage_y = hauteur // 2 - personnage_hauteur // 2
personnage_vitesse = 5

# déclaration des variables pour la courbe d'aprentissage
temps_de_survie = []
recompenses_totales = []
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_xlabel('Temps de survie')
ax.set_ylabel('Récompense totale')
ax.set_title('Courbe d\'apprentissage')


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
    tf.keras.layers.Dense(3, activation='linear')
])

# Optimiseur
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

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
    reward = collision_penalty + distance_reward
    return reward

# Exploration-exploitation équilibrée
def exploration_exploitation(action, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 2)  # Exploration aléatoire
    else:
        return action  # Exploitation du modèle

# Prédire l'action en fonction de l'état actuel
def prendre_decision(x, y, obstacle_x, obstacle_y, epsilon):
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
epsilon_min = 0.01  # Paramètre d'exploration minimal
epsilon_decay = 0.9999  # Taux de décroissance d'exploration

lines = []
colors = []


plt.ion()

while running:

    # Prendre une décision
    action = prendre_decision(personnage_x, personnage_y, obstacle_x, obstacle_y, epsilon)

    # Effectuer l'action
    if action == 0:
        personnage_y -= personnage_vitesse
    elif action == 1:
        personnage_y += personnage_vitesse

    # Mettre à jour les positions de l'obstacle et du personnage
    obstacle_x -= obstacle_vitesse

    # Calculer la distance entre le personnage et l'obstacle
    distance = abs(personnage_y - obstacle_y) / hauteur

    # Obtenir la récompense en fonction de la distance et du temps de survie
    reward = get_reward(distance, survived_time)

    # Enregistrer l'expérience dans la mémoire de relecture
    replay_memory.append((np.array([personnage_x / largeur, personnage_y / hauteur, obstacle_x / largeur, obstacle_y / hauteur,
                                    personnage_x / largeur, personnage_y / hauteur]), action, reward))

    # Mettre à jour le temps de survie
    survived_time += 1

    # Si l'obstacle sort de l'écran, le réinitialiser
    if obstacle_x < -obstacle_largeur:
        obstacle_x = largeur
        obstacle_y = random.randint(0, hauteur - obstacle_hauteur)
        survived_time = 0

    # Libérer périodiquement la mémoire de relecture pour éviter une consommation excessive de mémoire
    if iteration_count % memory_clear_interval == 0:
        replay_memory = replay_memory[-save_interval:]

    # Vérifier si suffisamment de données sont disponibles pour l'apprentissage
    if len(replay_memory) >= batch_size:
        # Échantillonner aléatoirement un lot de données de la mémoire de relecture
        batch = random.sample(replay_memory, batch_size)

        # Séparer les états, actions et récompenses
        etats = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        recompenses = np.array([x[2] for x in batch])

        # Prédire les valeurs Q pour les états actuels
        q_valeurs = model.predict(etats)

        # Prédire les valeurs Q pour les états suivants en utilisant le modèle cible
        q_valeurs_suivantes = model.predict(etats)

        # Mettre à jour les valeurs Q pour les actions choisies
        for i in range(len(batch)):
            action = actions[i]
            reward = recompenses[i]
            q_valeurs[i][action] = reward + gamma * np.max(q_valeurs_suivantes[i])

        # Effectuer une étape d'apprentissage sur le modèle
        with tf.GradientTape() as tape:
            q_predictions = model(etats, training=True)
            loss = tf.keras.losses.MeanSquaredError()(q_valeurs, q_predictions)

        # Calculer les gradients et mettre à jour les poids du modèle
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Mettre à jour le paramètre d'exploration
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Sauvegarder le modèle à intervalles réguliers
    if iterations % save_interval == 0:
        model.save(model_path)
        print("Le modèle a été sauvegardé.")
        
    temps_de_survie.append(survived_time)
    recompenses_totales.append(np.sum(reward))
    if reward <= -100:
        # Créer une nouvelle ligne pour la courbe
        line, = ax.plot(temps_de_survie, recompenses_totales, color=colors[len(lines)])

        # Ajouter la nouvelle ligne et couleur aux listes
        lines.append(line)
        colors.append('#{:06x}'.format(random.randint(0, 0xFFFFFF)))

        # Réinitialiser les listes pour la prochaine partie
        temps_de_survie = []
        recompenses_totales = []

    iterations += 1
    iteration_count += 1

    line.set_data(temps_de_survie, recompenses_totales)
    ax.relim()
    ax.autoscale_view()

    # Mettre à jour la fenêtre
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.0000001) 


# Sauvegarder le modèle à la fin de l'entraînement
model.save(model_path)
print("Le modèle a été sauvegardé.")

