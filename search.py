import string
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import sys
from timeit import default_timer as timer


def search(database: string, query: string, indextype: string, relevant: int, knn: int):

    print("------------------------------------------------------------------------------------")
    print("Image Query :" + query + "\tTypeIndexation " + indextype + " \t Knn " + str(knn))

    img_path = "Images/"  # [TO UPDATE]

    img_dir = img_path + database + "/"
    if database == "COREL" or database == "NISTER":
        img_dir = img_path + database + "_queries/"
    output_dir = "results/" + database
    resfilename = "results/" + query + "-" + indextype

    ## Load query image
    query_filename = img_dir + query + ".jpg"
    if not os.path.isfile(query_filename):
        print("Path to the query " + query_filename + " is not found -- EXIT\n")
        sys.exit(1)

    queryImage = cv2.imread(query_filename)

    plt.figure(0), plt.title("Image requete")
    plt.imshow(cv2.cvtColor(queryImage, cv2.COLOR_BGR2RGB))

    ## Compute query descriptors
    ### Opencv 2.4.13
    # sift = cv2.SIFT()
    ### Opencv 3
    sift = cv2.xfeatures2d.SIFT_create()

    kp, qdesc = sift.detectAndCompute(queryImage, None)
    print("Number of query descriptors :", len(qdesc))

    #######################
    ## Search for similar descriptors in the database
    # 1/ load the descriptors for final distance calculations
    #    load the data to map image numbers to image names for final display
    # 2/ load index and run K-NN query
    #
    #######################

    ## Load database descriptors **this is not the index**
    start = timer()
    dataBaseDescriptors = np.load(output_dir + "_DB_Descriptors.npy")
    imageBasePaths = np.load(output_dir + "_imagesPaths.npy")
    imageBaseIndex = np.load(output_dir + "_imagesIndex.npy")
    end = timer()
    print("time to load descriptors: " + str(end - start))

    ## Load database index (computed offline)
    # algorithm = 254 is the parameter to use in order to LOAD AN EXISTING INDEX !!!
    start = timer()
    index_params = dict(algorithm=254, filename=output_dir + "_flann_index-" + indextype + ".dat")
    ### Opencv 2.4.13
    fl = cv2.flann_Index(np.asarray(dataBaseDescriptors, np.float32), index_params)
    ### Opencv 3
    # fl = cv2.flann.Index(np.asarray(dataBaseDescriptors,np.float32), index_params)
    end = timer()
    print("time to load the index: " + str(end - start))

    ## Search on the database index
    start = timer()
    # knn = 10
    idx, dist = fl.knnSearch(np.asarray(qdesc, np.float32), knn, params={})
    end = timer()
    print("time to run the knn (k=" + str(knn) + ") search: " + str(end - start))
    # timeKnnExecution = end - start

    #######################
    ## Compute image scores (voting mechanism)
    #######################

    scores = np.zeros(len(imageBasePaths))
    for qnn in idx:
        for index in qnn:
            scores[imageBaseIndex[index]] += 1

    sortedScore = list(zip(scores, imageBasePaths))
    sortedScore.sort(reverse=True)

    # filter out the images with null score
    filtered_scores = [x for x in sortedScore if x[0] > 0]
    print(len(filtered_scores), "images have received a vote amongst ", len(sortedScore))

    # save results in file
    resfile = open(resfilename + "_ranked_list.txt", 'w')
    resfile.writelines(["%f %s\n" % item for item in filtered_scores])

    #######################
    ## Display the top images
    #######################
    top = 10
    plt.figure(1), plt.title(indextype)
    for i in range(top):
        img = cv2.imread(filtered_scores[i][1])
        score = filtered_scores[i][0]
        plt.subplot(2, 5, i + 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('rank ' + str(i + 1)), plt.xticks([]), plt.yticks([]), plt.xlabel(str(score))

    plt.savefig(resfilename + "_top" + str(top) + ".png")
    plt.show()

    #######################
    ## Evaluation : TO COMPLETE
    #######################

    def getImageId(imname):
        if database == "COREL":
            Id = imname.split('_')[1]
        elif database == "NISTER":
            Id = imname.split('-')[1]
        elif database == "Copydays":
            Id = imname.split('_')[-2]
        elif database == "base1":
            Id = imname.split('-')[1].split('.')[0]
        else:
            Id = imname.split('.')[-1]

        return Id

    queryId = getImageId(query)
    print("Identifiant de la requete : ", queryId)

    rpFile = open(resfilename + "_rp.dat", 'w')
    precision = np.zeros(len(filtered_scores), dtype=float)
    recall = np.zeros(len(filtered_scores), dtype=float)

    nbRelevantImage = relevant
    nb_pertinent = 0
    i = 0

    for elem in filtered_scores:
        # print("elem"+elem[1])
        # print("getimageid "+getImageId(elem[1]))
        # print("query "+queryId)
        if getImageId(elem[1]) in queryId:
            nb_pertinent += 1
            precision[i] = nb_pertinent / (i + 1)
            recall[i] = nb_pertinent / nbRelevantImage
            rpFile.write(str(precision[i]) + '\t' + str(recall[i]) + '\n')

        i += 1

    # print("precision")
    # print(precision)
    # print("recal")
    # print(recall)

    print("AP")
    print(sum(precision) / nbRelevantImage)

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall, precision, lw=2, color='navy',
             label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall for ' + query)
    plt.legend(loc="upper right")
    plt.savefig(resfilename + "_rp.png")
    plt.savefig(output_dir + query + "_rp.pdf", format='pdf')
    plt.show()

    # return timeKnnExecution
