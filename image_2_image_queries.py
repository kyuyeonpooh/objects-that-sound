from matplotlib import pyplot as plt
from utils.util import getNumToTagsMap, bgr2rgb
import torch
import numpy as np


def imageToImageQueries(embeddings='savedEmbeddings.pt', topk=5, use_tags=False):

    finalTag = getNumToTagsMap()
    #print(finalTag)

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

        # Open a file and store your queries here
    res = open("results/results_{0}_{1}.txt".format("im", "im"), "w+")

    for i in range(imgEmbedList.shape[0]):
        embed = imgEmbedList[i]
        dist  = ((embed - imgEmbedList)**2).sum(1)
        idx   = dist.argsort()[:topk]
        if use_tags:
            print(vidTagList[idx])
        plt.clf()
        num_fig = idx.shape[0]
        ax = plt.subplot(1, 3, 1)
        
        if use_tags:
            ax.set_title(finalTag[vidTagList[idx[0]]])
            res_query = finalTag[vidTagList[idx[0]]]
        plt.axis("off")
        plt.imshow(imgList[idx[0]].transpose(1,2,0))
        
        res_tags = []
        for j in range(1, num_fig):
            ax = plt.subplot(2, 3, j+1 + int(j/3))
            if use_tags:
                ax.set_title(finalTag[vidTagList[idx[j]]])
                res_tags.append(finalTag[vidTagList[idx[j]]])
            plt.imshow(imgList[idx[j]].transpose(1,2,0))
            plt.axis("off")

        # plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
        if flag:
            input()
            flag = False
        # res = raw_input("Do you want to save?")
        # if res == "y":
        plt.savefig("results/embed_im_im_{0}.png".format(i))
        
        res_input = "query: " + res_query + ", result: " + (",".join(res_tags))
        res.write(res_input + "\n")
    res.close()

if __name__ == "__main__":
    embedding_path = './save/embeddings/savedEmbeddings.pt'
    imageToImageQueries(embeddings=embedding_path, topk=5, use_tags=True)
