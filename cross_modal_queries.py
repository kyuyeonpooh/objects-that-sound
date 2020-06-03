from utils.util import getNumToTagsMap, bgr2rgb, save_result
from matplotlib import pyplot as plt
import os
import torch
import numpy as np


def crossModalQueries(embeddings=None, 
                      topk=5, 
                      mode1="au", 
                      mode2="im", 
                      use_tags=False,
                      result_path=None,
                      plot=False
                      ):
    
    if plot and topk != 5:
        raise ValueError("When plot is True, topk must be 5.")
    
    
    finalTag = getNumToTagsMap()
    #print(finalTag)

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
        raise ValueError("Invalid number of items: Found {} in 'savedEmbeddings.pt'".format(len(t)))

    print("Loaded embeddings.")

    # imgList = bgr2rgb(imgList)
    
    print("Size of data : " + str(imgList.shape[0]))

    # Open a file and store your queries here
    res = open("results/results_{0}_{1}.txt".format(mode1, mode2), "w+")

    assert mode1 != mode2
    
    res_queries = []
    res_tags = []
    for i in range(imgEmbedList.shape[0]):
        if mode1 == "im":
            embed = imgEmbedList[i]
        else:
            embed = audEmbedList[i]

        # Compute distance
        if mode2 == "im":
            dist = ((embed - imgEmbedList) ** 2).sum(1)
        else:
            dist = ((embed - audEmbedList) ** 2).sum(1)

        # Sort arguments
        idx = dist.argsort()[:topk]
        if use_tags:
            #print(vidTagList[idx])
            pass
        if plot:
            plt.clf()
        num_fig = idx.shape[0]

        # Actual query
        if use_tags:
            if plot:
                ax = plt.subplot(2, 3, 1)
                ax.set_title("Query: " + finalTag[vidTagList[i]])
            res_query = finalTag[vidTagList[i]]
        
        if plot:
            plt.axis("off")
            plt.imshow(imgList[i].transpose(1, 2, 0))

        # Top k matches
        res_tag = []
        for j in range(num_fig):
            if use_tags:
                if plot:
                    ax = plt.subplot(2, 3, j + 2)
                    ax.set_title(finalTag[vidTagList[idx[j]]])
                res_tag.append(finalTag[vidTagList[idx[j]]])
            if plot:
                plt.imshow(imgList[idx[j]].transpose(1, 2, 0))
                plt.axis("off")

        # plt.tight_layout()
        
        if plot:
            plt.draw()
            plt.pause(0.001)
            flag = True
            if flag:
                input()
                flag = False
        # ans = input("Do you want to save? (quit: q): ")
        # if ans == "q":
        #     break
        # elif ans == "y":
        #     if mode1 == "au":
        #         res.write(audioFiles[audioSampleList[i]] + "\n")
        #     else:
        #         tmpFiles = map(lambda x: audioFiles[audioSampleList[x]], idx)
        #         line = ", ".join(tmpFiles)
        #         res.write(line + "\n")
        #     plt.savefig("results/embed_{0}_{1}_{2}.png".format(mode1, mode2, i))
        
        res_queries.append(res_query)
        res_tags.append(res_tag)
    save_result(result_path, res_queries, res_tags)
    res.close()


if __name__ == "__main__":
    embedding_path = "save/embeddings/savedEmbeddings.pt"
    result_path = './results/results_au_im.pickle'
    crossModalQueries(embeddings=embedding_path, 
                      topk=5, 
                      mode1="im", 
                      mode2="au", 
                      use_tags=True,
                      result_path=result_path,
                      plot=False # Warning: when topk is not 5, plot should be False
                      )
