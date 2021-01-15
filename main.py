from search import search

import random


def search_with_knn(base, nom_images, nb_revelant):

    # search(base, nom_images, "KDTREE", nb_revelant, 10)
    # search(base, nom_images, "KDTREE", nb_revelant, 2)
    # search(base, nom_images, "KDTREE", nb_revelant, 5)
    # search(base, nom_images, "KDTREE", nb_revelant, 20)

    search(base, nom_images, "LINEAR", nb_revelant, 10)


# Base Corel
# imageCorel1 = random.choice(list(open('Images/COREL/_liste_COREL_queries.txt'))).rstrip()
# search_with_knn("COREL", imageCorel1, 7)
#
# imageCorel2 = random.choice(list(open('Images/COREL/_liste_COREL_queries.txt'))).rstrip()
# search_with_knn("COREL", imageCorel2, 7)
#
# # Base NISTER
# imageNister1 = random.choice(list(open('Images/NISTER/_liste_NISTER_queries.txt'))).rstrip()
# search_with_knn("NISTER", imageNister1, 3)
#
# imageNister2 = random.choice(list(open('Images/NISTER/_liste_NISTER_queries.txt'))).rstrip()
# search_with_knn("NISTER", imageNister2, 3)

# Base base1
imageBase11 = "1006-Monceau"
# search("base1", imageBase11, "KDTREE", 20, 10)
search_with_knn("base1", imageBase11, 20)

imageBase12 = "1265-Amsterdam"
search_with_knn("base1", imageBase12, 68)
#
# # Base Fickr
# imageFickr1 = "F_2007-01-16_0000009"
# search_with_knn("Fickr", imageFickr1, 4)
#
# imageFickr2 = "F_2007-01-16_0000028"
# search_with_knn("Fickr", imageFickr2, 4)



# search("COREL", "corel_0000009563_512", "LINEAR", 7)
