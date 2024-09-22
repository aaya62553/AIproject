Tennis video analyzis : 

- Récupération des bounding boxes des joueurs, de la balle, des keypoints du terrain en utilisant 3 modeles différents
- Filtration des joueurs par rapport aux autres humains sur la vidéo en utilisant la distance du terrain par rapport aux joueurs
- Mise en place d'un mini cour sur le coté en utilisant les dimensions réelles d'un terrain, recupération des keypoints, des points des joueurs et de la balle 
- Calcul des frames ou la balle est frappé par 1 des joueurs en utilisant la dérivée du mouvement de la balle selon l'axe y (hauteur), élimination des "outsiders" en vérifiant que la dérivée ne s'annule pas pendant au moins 25 frames
- Calcul des différentes vitesses de la balle (par joueur), du joueur, la moyene de la vitesse de la balle, moyenne de la vitesse du joueur