import json


def getNumToTagsMap():
    with open("./metadata/tags.cls") as fi:
        taglist = map(lambda x: x[:-1], fi.readlines())

    with open("./metadata/mappings.json") as fi:
        mapping = json.loads(fi.read())

    finalTag = list(map(lambda x: mapping[x], taglist))
    return finalTag


def bgr2rgb(img):
    res = img + 0.0
    if len(res.shape) == 4:
        res[:, 0, :, :] = img[:, 2, :, :]
        res[:, 2, :, :] = img[:, 0, :, :]
    else:
        res[0, :, :] = img[2, :, :]
        res[2, :, :] = img[0, :, :]
    return res


def reverseTransform(img, aud):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i in range(3):
        img[:, i, :, :] = img[:, i, :, :] * std[i] + mean[i]

    return img, aud


# convert string to boolean variable
def stob(bool_str, config_name):
    if isinstance(bool_str, bool):
        return bool_str
    if bool_str == "True" or bool_str == "true":
        return True
    elif bool_str == "False" or bool_str == "false":
        return False
    else:
        raise ValueError(
            "Configuration {} will only accept one among True, true, False, and false.".format(config_name)
        )
