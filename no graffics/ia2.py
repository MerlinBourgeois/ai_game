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
save_interval = 1000

# Modèle TensorFlow
model_path = "model_save.h5"
model = tf.keras.models.load_model(model_path)

# Fonction de récompense
def get_reward(distance, survived_time):
    collision_penalty = -100  # Pénalité pour la collision avec un obstacle
    distance_reward = 10 * (1 - distance)  # Récompense basée sur la distance par rapport à l'obstacle
    #time_reward = survived_time  # Récompense basée sur la durée de survie

    reward = collision_penalty + distance_reward #+ time_reward
    return reward

# Prédire l'action en fonction de l'état actuel
def prendre_decision(x, y, obstacle_x, obstacle_y):
    etat = np.array([x / largeur, y / hauteur, obstacle_x / largeur, obstacle_y / hauteur,
                     personnage_x / largeur, personnage_y / hauteur])
    action = model.predict(etat.reshape(1, 6))[0]
    return np.argmax(action)

# Boucle principale du jeu
running = True
clock = pygame.time.Clock()
survived_time = 0  # Temps de survie du personnage

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

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

    # Effacement de l'écran
    fenetre.fill(BLANC)

    # Dessin du personnage
    pygame.draw.rect(fenetre, NOIR, (personnage_x, personnage_y, personnage_largeur, personnage_hauteur))

    # Dessin de l'obstacle
    pygame.draw.rect(fenetre, NOIR, (obstacle_x, obstacle_y, obstacle_largeur, obstacle_hauteur))

    # Rafraîchissement de l'écran
    pygame.display.flip()

    clock.tick(60)

    survived_time += 1  # Incrémente le temps de survie du personnage

    # Recharger l'IA à intervalle défini
    if survived_time % save_interval == 0:
        model = tf.keras.models.load_model(model_path)
        print("IA rechargée à l'itération", survived_time)

# Quitter Pygame
pygame.quit()
