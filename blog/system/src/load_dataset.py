import numpy as np


class Sampler_hdf5_news_cut(object):
    """
    category가 나뉘어져있는 형태의 hdf5파일을 받지만, 개별적인 뉴스 안에서 input data를 sampling함
    """

    def __init__(self, group, seed=None):

        self.group = group
        self.dataset = []
        self.category_count = []

        for int_name in range(len(group)):
            name_dataset = str(int_name)
            self.dataset.append(group[name_dataset])
            self.category_count.append(len(group[name_dataset]))

        self.rs = np.random.RandomState(seed=seed)
        self.category_count = np.array(self.category_count)
        self.category_prob = list(self.category_count/sum(self.category_count))

    def sample(self, length):
        selected_category = np.random.choice(len(self.group), 1, self.category_prob)[0]
        news_selected_category = np.random.choice(len(self.dataset[selected_category]), 1)[0]

        index = self.rs.randint(0, len(self.dataset[selected_category][news_selected_category]) - length + 1)
        return self.dataset[selected_category][news_selected_category][index:length+index]


class Sampler_crawled_data(object):
    def __init__(self, group, seed=None):
        self.group = group
        self.dataset = []

        for key in group.keys():
            self.dataset.append(group[key])

        self.rs = np.random.RandomState(seed=seed)

    def sample(self, length):
        index = self.rs.randint(0, len(self.dataset[0]) - length + 1)
        return self.dataset[0][index:length+index]
