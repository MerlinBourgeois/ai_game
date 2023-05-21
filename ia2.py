import pygame
import random
import tensorflow as tf
import numpy as np

# Initialisation de Pygame
pygame.init()

# Définition de la taille de la fenêtre
largeur = 800
hauteur = 600
fenetre = pygame.display.set_mode((largeur, hauteur))

# Couleurs
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)

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

# Modèle TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(6,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Optimiseur
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Charger le modèle si un fichier de sauvegarde existe
checkpoint_path = "model_checkpoint.ckpt"
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

# Fonction de récompense
def get_reward():
    if obstacle_y < personnage_y or obstacle_y + obstacle_hauteur > personnage_y + personnage_hauteur:
        return -1  # Pénalité si l'obstacle est trop haut ou trop bas par rapport au personnage
    else:
        return -10  # Pénalité supplémentaire pour la collision avec un obstacle

# Prédire l'action en fonction de l'état actuel
def prendre_decision(x, y, obstacle_x, obstacle_y):
    etat = np.array([x / largeur, y / hauteur, obstacle_x / largeur, obstacle_y / hauteur,
                     personnage_x / largeur, personnage_y / hauteur])
    action = model.predict(etat.reshape(1, 6))[0]
    return np.argmax(action)

# Boucle principale du jeu
running = True
clock = pygame.time.Clock()
iterations = 1  # Nombre d'itérations d'apprentissage supplémentaires

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_s:
                # Sauvegarder le modèle
                checkpoint.save(file_prefix=checkpoint_path)
                print("Modèle sauvegardé.")

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

        # Mise à jour du modèle
        reward = get_reward()
        with tf.GradientTape() as tape:
            etat = np.array([personnage_x / largeur, personnage_y / hauteur, obstacle_x / largeur, obstacle_y / hauteur,
                             personnage_x / largeur, personnage_y / hauteur])
            prediction = model(np.expand_dims(etat, axis=0))
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=[reward], logits=prediction))

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Libérer la mémoire
        del gradients, tape

    # Effacement de l'écran
    fenetre.fill(BLANC)

    # Dessin du personnage
    pygame.draw.rect(fenetre, NOIR, (personnage_x, personnage_y, personnage_largeur, personnage_hauteur))

    # Dessin de l'obstacle
    pygame.draw.rect(fenetre, NOIR, (obstacle_x, obstacle_y, obstacle_largeur, obstacle_hauteur))

    # Rafraîchissement de l'écran
    pygame.display.flip()

    clock.tick(60)

# Quitter Pygame
pygame.quit()
