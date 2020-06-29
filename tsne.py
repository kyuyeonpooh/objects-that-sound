import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
from utils import util

tagmap = util.getNumToTagsMap()
embeddings = torch.load("embedding/AVE_aug_inst.pt")

img_embed = np.asarray(embeddings[2]).squeeze()
aud_embed = np.asarray(embeddings[3]).squeeze()
img_tags = embeddings[4]
aud_tags = embeddings[5]
img_tags = np.array([img_tags[i][0] for i in range(len(img_tags))])
aud_tags = np.array([aud_tags[i][0] for i in range(len(aud_tags))])
counter = Counter(img_tags)
counter = sorted(counter, key=counter.get, reverse=True)
top5 = [counter[0], counter[3], counter[5], counter[7], counter[12]]
tagmaplist = [tagmap[x] for x in top5]
print(top5, tagmaplist)
idxlist = []
for i in range(5):
    idxlist.append(np.where(img_tags == top5[i]))
embedlist = tuple(img_embed[idxlist[i]] for i in range(5))
embedlist_aud = tuple(aud_embed[idxlist[i]] for i in range(5))
img_embeddings = np.asarray(np.vstack(embedlist))
aud_embeddings = np.asarray(np.vstack(embedlist_aud))
label = []
for i in range(5):
    label += [i] * len(embedlist[i])


plt.tick_params(
    axis="both", which="both", bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False
)
tsne = TSNE(n_iter=10000)
imgsubspace = tsne.fit_transform(img_embeddings)
xs = imgsubspace[:, 0]
ys = imgsubspace[:, 1]
sc1 = plt.scatter(xs, ys, c=label)
audsubspace = tsne.fit_transform(aud_embeddings)
xsaud = audsubspace[:, 0]
ysaud = audsubspace[:, 1]
sc2 = plt.scatter(xsaud, ysaud, c=label, marker="x")
plt.legend(handles=sc1.legend_elements()[0], labels=tagmaplist)
plt.savefig("tsne.pdf")
plt.show()
