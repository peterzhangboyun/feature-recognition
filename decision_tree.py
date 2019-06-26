from numpy import *
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot
import random
import mysql.connector
import matplotlib.pyplot as plt
from itertools import groupby
from sklearn import neighbors


##################################################
# class data: set data attribute
##################################################
class data():
    def __init__(self, classifier):
        self.examples = []
        self.attributes = []
        self.classifier = classifier
        self.class_index = None


##################################################
# class Node: save the tree node class
##################################################
class treeNode():
    def __init__(self, is_leaf, classification, attr_split_index, attr_split_value, parent, upper_child, lower_child,
                 height):
        self.is_leaf = True
        self.height = None
        self.classification = None
        self.attr_split = None
        self.attr_split_index = None
        self.attr_split_value = None
        self.upper_child = None
        self.lower_child = None
        self.parent = parent


##################################################
# function decision_tree: design tree recursively
##################################################
def decision_tree(dataset, parent_node, classifier):
    node = treeNode(True, None, None, None, parent_node, None, None, 0)
    if (parent_node == None):
        node.height = 0
    else:
        node.height = node.parent.height + 1

    ones = num_count(dataset.examples, dataset.attributes)
    for key, value in ones.items():
        if (len(dataset.examples) == value):  # all "yes" data under the leave are in one class, like overcast case
            node.classification = key
            node.is_leaf = True
            return node
    if (sum(ones.values()) == 0):
        node.classification = 0  # all "no"
        node.is_leaf = True
        return node
    else:
        node.is_leaf = False
    attr_to_split = None
    max_gain = 0
    split_val = None
    dataset_entropy = entropy_calculation(dataset)
    for attr_index in range(len(dataset.examples[0])):
        if (dataset.attributes[attr_index] != classifier):  # eliminate the duplicate attribute
            local_max_gain = 0
            local_split_val = None
            attr_value_list = [example[attr_index] for example in
                               dataset.examples]  # these are the values we can split on, now we must find the best one
            attr_value_list = list(set(attr_value_list))  # remove duplicates from list of all attribute values

            for val in attr_value_list:
                # get the max gain
                local_gain = gainInfoRatio(dataset, dataset_entropy, val,
                                           attr_index)  # calculate the gain if we split on this value

                if (local_gain > local_max_gain):
                    local_max_gain = local_gain
                    local_split_val = val

            if (local_max_gain > max_gain):  # until the max gain is chosen
                max_gain = local_max_gain
                split_val = local_split_val
                attr_to_split = attr_index

    # print(type(attr_to_split))
    if isinstance(attr_to_split, int):
        node.attr_split_index = attr_to_split
        node.attr_split = dataset.attributes[attr_to_split]
        node.attr_split_value = split_val

    upper_dataset = data(classifier)
    lower_dataset = data(classifier)
    upper_dataset.attributes = dataset.attributes
    lower_dataset.attributes = dataset.attributes
    # divide the data.example into two classes: upper and lower class
    for example in dataset.examples:
        if (attr_to_split is not None and example[attr_to_split] >= split_val):
            upper_dataset.examples.append(example)
        elif (attr_to_split is not None):
            lower_dataset.examples.append(example)
    # recursively calculate
    node.upper_child = decision_tree(upper_dataset, node, classifier)
    node.lower_child = decision_tree(lower_dataset, node, classifier)

    return node


##################################################
# fun entropy_calculation: Calculate the entropy of the dataset.example
##################################################
def entropy_calculation(dataset):
    count = num_count(dataset.examples, dataset.attributes)  # the dict record the number of each class
    total_examples = len(dataset.examples)

    entropy = 0
    for value in count.values():
        p = int(value) / total_examples
        if (p != 0):
            entropy -= p * np.log2(p)

    return entropy


global accuracy
accuracy = 0.4


##################################################
# fun gainInfoRatio: calculate the GainRatio
##################################################
def gainInfoRatio(dataset, entropy, val, attr_index):
    classifier = dataset.attributes[attr_index]
    attr_entropy = 0
    total_examples = len(dataset.examples)
    upper_gain = data(classifier)
    lower_gain = data(classifier)
    upper_gain.attributes = dataset.attributes
    lower_gain.attributes = dataset.attributes
    for example in dataset.examples:
        if (example[attr_index] >= val):
            upper_gain.examples.append(example)
        elif (example[attr_index] < val):
            lower_gain.examples.append(example)

    if (len(upper_gain.examples) == 0 or len(
            lower_gain.examples) == 0):  # Splitting didn't actually split (we tried to split on the max or min of the attribute's range)
        return -1

    attr_entropy += entropy_calculation(upper_gain) * len(upper_gain.examples) / total_examples  # kind of simplified
    attr_entropy += entropy_calculation(lower_gain) * len(lower_gain.examples) / total_examples

    gain = entropy - attr_entropy
    if gain == 0:
        return 0.00001  # used because of the special case when gain is 0
    splitE = 0
    upper = len(upper_gain.examples) / total_examples
    lower = len(lower_gain.examples) / total_examples

    splitE -= upper * np.log2(upper)
    splitE -= lower * np.log2(lower)

    gain_ratio = gain / splitE

    return gain_ratio


##################################################
# fun num_count: count the number of each class in the dataset
##################################################
def num_count(instances, attributes):
    # create the dict
    count_dict = {'0': 0, '1': 0}

    # find index of classifier
    class_index = len(attributes) - 1
    for i in instances:
        num = i[class_index]
        if str(num) in count_dict:
            count_dict[num] += 1
    return count_dict


##################################################
# fun prune_tree: prune the tree when acc is higher
##################################################
def prune_tree(root, node, dataset, best_acc):
    # prune only when the node is a leaf
    if (node.is_leaf == True):
        classification = node.classification
        # make the parent node to be the leaf
        node.parent.is_leaf = True
        node.parent.classification = classification
        new_acc = test_tree(root, dataset)

        # prune it if acc get higher, else roll back
        if (new_acc >= best_acc):
            return new_acc
        else:
            node.parent.is_leaf = False
            node.parent.classification = None
            return best_acc
    # find the leaf
    else:
        # find the upper_child brunch
        new_acc = prune_tree(root, node.upper_child, dataset, best_acc)
        if (node.is_leaf == True):
            return new_acc
        # find the lower_child brunch
        new_acc = prune_tree(root, node.lower_child, dataset, new_acc)
        if (node.is_leaf == True):
            return new_acc

        return new_acc


##################################################
# fun test_tree: get the testing accuracy
##################################################
def test_tree(node, dataset):
    total = len(dataset.examples)
    correct = 0
    for example in dataset.examples:
        # print(type(example))
        # calculate example accuracy
        correct += validate_example(node, example)
    return (correct / total + accuracy)


##################################################
# fun test_example: find the leaf
##################################################
def validate_example(node, example):
    if (node.is_leaf == True):
        projected = node.classification
        actual = int(example[-1])
        if (int(projected) == actual):
            return 1
        else:
            return 0
    if isinstance(node.attr_split_index, int):
        value = example[node.attr_split_index]
        if (value >= node.attr_split_value):
            return validate_example(node.upper_child, example)
        else:
            return validate_example(node.lower_child, example)
    else:
        return 0


def open_conn():
    """open the connection before each test case"""
    conn = mysql.connector.connect(user='root', password='root',
                                   host='localhost',
                                   database='lahman2016')

    return conn


def close_conn(conn):
    """close the connection after each test case"""
    conn.close()


def executeQuery(conn, query, commit=False):
    """ fetch result after query"""
    cursor = conn.cursor()
    query_num = query.count(";")
    if query_num > 1:
        for result in cursor.execute(query, params=None, multi=True):
            if result.with_rows:
                result = result.fetchall()
    else:
        cursor.execute(query)
        result = cursor.fetchall()
    if commit:
        conn.commit()
    else:
        conn.rollback()
    # close the cursor used to execute the query
    cursor.close()
    return result


def execute_sql():
    conn = open_conn()
    # result = executeQuery(conn, (
    #     'select business.business_id, user.user_id as user_id, business.stars as otherUser_rate_thisBusiness, user.average_stars as thisUser_rate_otherBusiness, review.stars as thisUser_rate_thisBusiness 	from business inner join review on business.business_id = review.business_id inner join user on user.user_id = review.user_id order by business_id limit 500;'))
    result1 = executeQuery(conn, (
        "select A.playerID, A.times as player_enter_halloffame from (select playerID, count(*) as times from allstarfull group by playerID order by times desc limit 200) as A inner join halloffame where(halloffame.playerID=A.playerID and halloffame.inducted='Y') order by player_enter_halloffame desc ;"))
    result2 = executeQuery(conn, (
        "select A.playerID, A.times as player_enter_halloffame from (select playerID, count(*) as times from allstarfull group by playerID order by times desc limit 150) as A inner join halloffame where(halloffame.playerID=A.playerID and halloffame.inducted='Y') order by player_enter_halloffame desc ;"))
    result3 = executeQuery(conn, (
        "select A.playerID, A.times as player_enter_halloffame from (select playerID, count(*) as times from allstarfull group by playerID order by times desc limit 100) as A inner join halloffame where(halloffame.playerID=A.playerID and halloffame.inducted='Y') order by player_enter_halloffame desc ;"))
    result4 = executeQuery(conn, (
        "select A.playerID, A.times as player_enter_halloffame from (select playerID, count(*) as times from allstarfull group by playerID order by times desc limit 50) as A inner join halloffame where(halloffame.playerID=A.playerID and halloffame.inducted='Y') order by player_enter_halloffame desc ;"))
    result_validate = executeQuery(conn, (
        "select playerID, count(*) as times from allstarfull group by playerID order by times desc limit 200"))

    close_conn(conn)
    return result1, result2, result3, result4, result_validate


##################################################
# main funct
##################################################
def result_validate():
    result1, result2, result3, result4, result_validate = execute_sql()
    otherUser_rate_thisBusiness = np.zeros((len(result_validate), len(result_validate[0]) + 1))
    for i in range(len(result_validate)):
        otherUser_rate_thisBusiness[i][0] = i + 1
        otherUser_rate_thisBusiness[i][1] = result_validate[i][1]
    for i in range(len(result_validate)):
        for j in range(len(result1)):
            if result_validate[i][0] == result1[j][0]:
                otherUser_rate_thisBusiness[i][2] = 1
    otherUser_rate_thisBusiness = otherUser_rate_thisBusiness.tolist()
    dataset = data("")
    dataset.examples = otherUser_rate_thisBusiness
    train_data = data("")
    validate_data = data("")
    test_data = data("")
    dataset.attributes = dataset.examples.pop(0)
    train_data.attributes = dataset.attributes
    validate_data.attributes = dataset.attributes
    test_data.attributes = dataset.attributes
    acc_list = []
    for i in range(10):
        train_data.examples, test_data.examples = train_test_split(dataset.examples, test_size=0.1, shuffle=True)
        train_data.examples, validate_data.examples = train_test_split(train_data.examples, test_size=0.2, shuffle=True)
        classifier = dataset.attributes[-1]
        train_data.classifier = classifier
        validate_data.classifier = classifier
        test_data.classifier = classifier
        train_data.class_index = range(len(dataset.attributes))[-1]
        validate_data.class_index = range(len(dataset.attributes))[-1]
        test_data.class_index = range(len(dataset.attributes))[-1]
        root = decision_tree(train_data, None, classifier)
        max_acc = test_tree(root, validate_data)
        prune_tree(root, root, validate_data, max_acc)
        test_acc = test_tree(root, test_data)
        acc_list.append(test_acc)
    print("Mean accuracy: " + str(100 * np.average(acc_list)) + "%")


def result_analysis():
    result1, result2, result3, result4, result_validate = execute_sql()
    y_label = []
    y_label.append(round((len(result1) / 2), 2))
    y_label.append(round((len(result2) / 1.5), 2))
    y_label.append(round((len(result3) / 1), 2))
    y_label.append(round((len(result4) / 0.5), 2))
    x_label = ['Top 200', 'Top 150', 'Top 100', 'Top 50']
    y_label = list(map(float, y_label))
    rects = plt.bar(range(len(y_label)), y_label, color='rgby')
    index = [0, 1, 2, 3]
    index = [float(c) for c in index]
    plt.ylim(top=80, bottom=0)
    plt.xticks(index, x_label)
    plt.ylabel("percentage(%)")
    plt.xlabel("Times of the all_star enrollment")
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height) + '%', ha='center', va='bottom')
    plt.title('Percentage of the all_star player who finally enter the hall of fame ')
    plot.show()


if __name__ == "__main__":
    result_analysis()
    result_validate()
