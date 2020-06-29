from utils.util import getNumToTagsMap, bgr2rgb, save_result
from matplotlib import pyplot as plt
import os
import torch
import numpy as np


def crossModalQueries(embeddings=None, topk=5, mode1="au", mode2="im", use_tags=False, result_path=None, plot=False):

    if plot and topk != 5:
        raise ValueError("When plot is True, topk must be 5.")

    finalTag = getNumToTagsMap()
    # print(finalTag)

    for r, di, files in os.walk("./data/test/audio"):
        audioFiles = sorted(files)

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

    # imgList = bgr2rgb(imgList)

    print("Size of data : " + str(len(imgEmbedList)))

    # Open a file and store your queries here
    if plot:
        res = open("results/results_{0}_{1}.txt".format(mode1, mode2), "w+")

    assert mode1 != mode2

    res_queries = []
    res_tags = []
    for i in range(len(imgEmbedList)):
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
            # print(vidTagList[idx])
            pass
        if plot:
            plt.clf()
        num_fig = idx.shape[0]

        # Actual query
        if use_tags:
            if plot:
                ax = plt.subplot(2, 3, 1)
                ax.set_title("Query: " + str([finalTag[x] for x in vidTagList[i]]))
            res_query = [finalTag[x] for x in vidTagList[i]]

        if plot:
            plt.axis("off")
            plt.imshow(imgList[i].squeeze().transpose(1, 2, 0))

        # Top k matches
        res_tag = []
        for j in range(num_fig):
            if use_tags:
                res_tag_ = [finalTag[x] for x in vidTagList[idx[j]]]
                if plot:
                    ax = plt.subplot(2, 3, j + 2)
                    ax.set_title(str(res_tag_))
                res_tag.append(res_tag_)
            if plot:
                plt.imshow(imgList[idx[j]].squeeze().transpose(1, 2, 0))
                plt.axis("off")

        # plt.tight_layout()

        if plot:
            plt.draw()
            plt.pause(0.001)
            flag = True
            if flag:
                input()
                flag = False
            ans = input("Do you want to save? (quit: q): ")
            if ans == "q":
                break
            elif ans == "y":
                if mode1 == "au":
                    res.write(audioFiles[audioSampleList[i][0]] + "\n")
                    print(audioFiles[audioSampleList[i][0]])
                else:
                    tmpFiles = map(lambda x: audioFiles[x], idx)
                    line = ", ".join(tmpFiles)
                    print(line)
                    res.write(line + "\n")
                plt.savefig("results/embed_{0}_{1}_{2}.png".format(mode1, mode2, i))

        res_queries.append(res_query)
        res_tags.append(res_tag)
    save_result(result_path, res_queries, res_tags)
    if plot:
        res.close()


if __name__ == "__main__":
    embedding_path = "./embedding/L3_aug_inst.pt"
    result_path = "./results/L3_aug_inst_a2i.pickle"
    crossModalQueries(
        embeddings=embedding_path,
        topk=6000,
        mode1="au",
        mode2="im",
        use_tags=True,
        result_path=result_path,
        plot=False,  # Warning: when topk is not 5, plot should be False
    )
