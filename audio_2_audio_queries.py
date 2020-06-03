from matplotlib import pyplot as plt
from utils.util import getNumToTagsMap, bgr2rgb, save_result
import torch
import numpy as np


def AudioToAudioQueries(embeddings=None, 
                        topk=5, 
                        use_tags=False, 
                        result_path=None,
                        plot=False
                        ):

    if plot and topk != 5:
        raise ValueError("When plot is True, topk must be 5.")

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

    print("Size of data : " + str(audList.shape[0]))

    res_queries = []
    res_tags = []
    for i in range(audEmbedList.shape[0]):
        embed = audEmbedList[i]
        dist  = ((embed - audEmbedList)**2).sum(1)
        idx   = dist.argsort()[:topk]
        if use_tags:
            print(audTagList[idx])
    
        num_fig = idx.shape[0]
        if plot:
            plt.clf()
            ax = plt.subplot(1, 3, 1)
        
        if use_tags:
            if plot:
                ax.set_title(finalTag[audTagList[idx[0]]])
            res_query = finalTag[audTagList[idx[0]]]
        if plot:
            plt.axis("off")
            plt.imshow(audList[idx[0]].transpose(1,2,0))
        
        res_tag = []
        for j in range(1, num_fig):
            ax = plt.subplot(2, 3, j+1 + int(j/3))
            if use_tags:
                if plot:
                    ax.set_title(finalTag[audTagList[idx[j]]])
                res_tag.append(finalTag[audTagList[idx[j]]])
            if plot:
                plt.imshow(audList[idx[j]].transpose(1,2,0))
                plt.axis("off")

        # plt.tight_layout()
        #plt.draw()
        #plt.pause(0.001)
        #flag = True
        #if flag:
        #    input()
        #    flag = False
        # res = raw_input("Do you want to save?")
        # if res == "y":
        plt.savefig("results/embed_au_au_{0}.png".format(i))
        
        res_queries.append(res_query)
        res_tags.append(res_tag)    
    save_result(result_path, res_queries, res_tags)


if __name__ == "__main__":
    embedding_path = './save/embeddings/savedEmbeddings.pt'
    result_path = './results/results.pickle'
    AudioToAudioQueries(embeddings=embedding_path, 
                        topk=5,
                        use_tags=True, 
                        result_path=result_path,
                        plot=False # Warning: when topk is not 5, plot should be False
                        )
