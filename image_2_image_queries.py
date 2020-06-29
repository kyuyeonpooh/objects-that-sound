from matplotlib import pyplot as plt
from utils.util import getNumToTagsMap, bgr2rgb, save_result
import torch
import numpy as np


def imageToImageQueries(embeddings=None, topk=5, use_tags=False, result_path=None, plot=False):

    if plot and topk != 5:
        raise ValueError("When plot is True, topk must be 5.")

    finalTag = getNumToTagsMap()
    # print(finalTag)

    t = torch.load(embeddings)
    for i in [2, 3]:
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

    print("Size of data : " + str(len(imgEmbedList)))

    res_queries = []
    res_tags = []
    for i in range(len(imgEmbedList)):
        embed = imgEmbedList[i]
        dist = ((embed - imgEmbedList) ** 2).sum(1)
        idx = dist.argsort()[:topk]
        if use_tags:
            # print(vidTagList[idx])
            pass

        num_fig = idx.shape[0]
        if plot:
            plt.clf()
            ax = plt.subplot(1, 3, 1)

        if use_tags:
            if plot:
                ax.set_title(finalTag[vidTagList[idx[0]]])
            # res_query = finalTag[vidTagList[idx[0]]]
            res_query = [finalTag[x] for x in vidTagList[i]]
        if plot:
            plt.axis("off")
            plt.imshow(imgList[idx[0]].transpose(1, 2, 0))

        res_tag = []
        for j in range(1, num_fig + 1):
            if plot:
                ax = plt.subplot(2, 3, j + 1 + int(j / 3))
            if use_tags:
                if plot:
                    ax.set_title(finalTag[vidTagList[idx[j]]])
                # res_tag.append(finalTag[vidTagList[idx[j]]])
                res_tag_ = [finalTag[x] for x in vidTagList[idx[j - 1]]]
                res_tag.append(res_tag_)
            if plot:
                plt.imshow(imgList[idx[j]].transpose(1, 2, 0))
                plt.axis("off")

        if plot:
            plt.draw()
            plt.pause(0.001)
            flag = True
            if flag:
                input()
            flag = False
            res = input("Do you want to save?")
            if res == "y":
                plt.savefig("results/embed_im_im_{0}.png".format(i))

        res_queries.append(res_query)
        res_tags.append(res_tag)
    save_result(result_path, res_queries, res_tags)


if __name__ == "__main__":
    embedding_path = "./embedding/L3_aug_inst.pt"
    result_path = "./results/L3_aug_inst_i2i.pickle"
    imageToImageQueries(
        embeddings=embedding_path,
        topk=6000,
        use_tags=True,
        result_path=result_path,
        plot=False,  # Warning: when topk is not 5, plot should be False
    )
