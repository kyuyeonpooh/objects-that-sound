import os
import json
import numpy as np


class Ontology:
    def __init__(self, data_dir):
        # 0. read AudioSet Ontology data
        with open(os.path.join(data_dir, "ontology.json")) as data_file:
            raw_aso = json.load(data_file)

        # 1. format data as a dictionary
        # aso["/m/0dgw9r"] > {
        #     "restrictions": ["abstract"],
        #     "child_ids": ["/m/09l8g", "/m/01w250", ..., "/m/04xp5v", "/t/dd00012"],
        #     "name": "Human sounds"
        # }
        self.aso = {}
        for category in raw_aso:
            tmp = {}
            tmp["name"] = category["name"]
            tmp["restrictions"] = category["restrictions"]
            tmp["child_ids"] = category["child_ids"]
            tmp["parents_ids"] = None
            self.aso[category["id"]] = tmp

        # 2. fetch higher_categories > ["/m/0dgw9r","/m/0jbk","/m/04rlf","/t/dd00098","/t/dd00041","/m/059j3w","/t/dd00123"]
        for cat in self.aso:  # find parents
            for c in self.aso[cat]["child_ids"]:
                self.aso[c]["parents_ids"] = cat

        # higher_categories = []  # higher_categories are the ones without parents
        for cat in self.aso:
            if self.aso[cat]["parents_ids"] == None:
                self.aso[cat]["parents_ids"] = "Ontology"

        with open(os.path.join(data_dir, "genre_tag_map.json"), "r") as fin:
            genre_tag_dict = json.load(fin)

        self.tag_genre_dict = {val: key for key, val in genre_tag_dict.items()}

    def show_path(self, path):
        print("[ root ", end="")
        for i in range(1, len(path)):
            print('"' + self.aso[path[i]]["name"] + '" ', end="")
        print("]")

    def get_tag(self, class_name):
        return self.tag_genre_dict[class_name]

    def get_tree_path(self, classname):
        if classname not in self.aso:
            print("The audioset do not include the class %s" % classname)
            return []
        path_list = []
        top_class = classname
        while top_class != "Ontology":
            path_list.insert(0, top_class)
            top_class = self.aso[top_class]["parents_ids"]
        path_list.insert(0, top_class)
        return path_list

    def tree_min_distance(self, class1, class2):
        class1_path = self.get_tree_path(class1)
        class2_path = self.get_tree_path(class2)
        inter_count = 0
        for i in range(min(len(class1_path), len(class2_path))):
            if class1_path[i] == class2_path[i]:
                inter_count += 1
            else:
                break
        return len(class1_path) + len(class2_path) - 2 * inter_count

    def get_min_distance(self, class1, class2):
        # class1, class2: tag (e.g. u'/m/026t6')
        if class1.find("/") >= 0 and class2.find("/") >= 0:
            return self.tree_min_distance(class1, class2)
        # class1, class2: name (e.g. Drum)
        else:
            return self.tree_min_distance(self.tag_genre_dict[class1], self.tag_genre_dict[class2])


if __name__ == "__main__":
    data_dir = "json"
    ontology = Ontology(data_dir)

    # How to show tree path - Example 1. Acoustic guitar
    tag = ontology.get_tag("Acoustic guitar")
    tree_path = ontology.get_tree_path(tag)
    ontology.show_path(tree_path)

    # How to show tree path - Example 2. Drum
    tag = ontology.get_tag("Drum")
    tree_path = ontology.get_tree_path(tag)
    ontology.show_path(tree_path)

    # Calculate minimum distance
    distance = ontology.get_min_distance("Acoustic guitar", "Drum")
    print("Minimum distance: ", distance)
