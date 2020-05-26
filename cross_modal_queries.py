from utils.util import getNumToTagsMap, bgr2rgb
from matplotlib import pyplot as plt
import os
import torch
import numpy as np

def crossModalQueries(embeddings='savedEmbeddings.pt', topk=5, mode1="au", mode2="im", use_tags=False):
    finalTag = getNumToTagsMap()
    print(finalTag)

    for r, di, files in os.walk("data/test_audio"):
        audioFiles = sorted(files)

    t = torch.load(embeddings)
    for i in range(len(t)):
        t[i] = np.concatenate(t[i])
        
    # Generalize here
    if len(t) == 6:
        imgList, audList, imgEmbedList, audEmbedList, vidTagList, audTagList = t
    elif len(t) == 7:
        imgList, audList, imgEmbedList, audEmbedList, vidTagList, audTagList, audioSampleList = t
    elif len(t) == 4:
        imgList, audList, imgEmbedList, audEmbedList = t
    else:
        raise ValueError(
                "Invalid number of items: Found {} in 'savedEmbeddings.pt'".format(len(t))
            )

    print("Loaded embeddings.")

    #imgList = bgr2rgb(imgList)
    flag = True
    print(imgList.shape[0])

    # Open a file and store your queries here
    res = open("results/results_{0}_{1}.txt".format(mode1, mode2), "w+")

    assert(mode1 != mode2)
    
    for i in range(imgEmbedList.shape[0]):
        if mode1 == "im":
            embed = imgEmbedList[i]
        else:
            embed = audEmbedList[i]

        # Compute distance
        if mode2 == "im":
            dist = ((embed - imgEmbedList)**2).sum(1)
        else:
            dist = ((embed - audEmbedList)**2).sum(1)

        # Sort arguments
        idx = dist.argsort()[:topk]
        if use_tags:
            print(vidTagList[idx])
        plt.clf()
        num_fig = idx.shape[0]

        # Actual query
        ax = plt.subplot(2, 3, 1)
        if use_tags:
            ax.set_title("Query: " + finalTag[vidTagList[i, 0]])
        plt.axis("off")
        plt.imshow(imgList[i].transpose(1,2,0))

        # Top 5 matches
        for j in range(num_fig):
            ax = plt.subplot(2, 3, j+2)
            if use_tags:
                ax.set_title(finalTag[vidTagList[idx[j], 0]])
            plt.imshow(imgList[idx[j]].transpose(1,2,0))
            plt.axis("off")

        # plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
        if flag:
            input()
            flag = False
        # res = input("Do you want to save?")
        # if res == "y":

        if mode1 == "au":
            res.write(audioFiles[audioSampleList[i, 0]] + "\n")
        else:
            tmpFiles = map(lambda x: audioFiles[audioSampleList[x, 0]], idx)
            line = ", ".join(tmpFiles)
            res.write(line + "\n")

        plt.savefig("results/embed_{0}_{1}_{2}.png".format(mode1, mode2, i))
    res.close()

if __name__ == "__main__":
    embedding_path = './save/embeddings/savedEmbeddings.pt'
    crossModalQueries(embeddings=embedding_path, topk=5, mode1="au", mode2="im", use_tags=False)
