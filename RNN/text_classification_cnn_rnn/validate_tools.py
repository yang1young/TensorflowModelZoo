import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable


# get jaccard_similarity
def jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality / float(union_cardinality)


# sort result and formate using prettytable
def result_compare(result1, result2, label):
    label, result1, result2 = (list(x) for x in
                               zip(*sorted(zip(result1, result2, label), key=lambda pair: pair[0], reverse=True)))
    table = PrettyTable(["label", "result1", "result2"])
    table.align["name"] = "l"
    table.padding_width = 1

    for i, j, x in zip(result1, result2, label):
        table.add_row([str(x), str(i), str(j)])
    return table


# matplotlib plot
def plot_two_result(result1, result2, label):
    x = np.linspace(0, len(list(label)), len(list(label)))
    plt.figure(2)
    plt.plot(x, list(result1), label='result1')
    plt.scatter(x, list(result2), color='r', marker='^', s=20, label='result2')
    plt.xlabel('label')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

# def other_plot():
#     Z = [common,ai-common,si-common]
#     plt.figure(figsize=(6, 9))
#     labels = ['common','AST model unique','source model unique']
#     sizes = Z
#     colors = ['red', 'yellowgreen', 'lightskyblue']
#     explode = (0.05, 0.05, 0.05)
#     patches, l_text, p_text = plt.pie(sizes, explode=explode,labels=labels, colors=colors,
#                                       labeldistance=1.1, autopct='%3.1f%%', shadow=False,
#                                       startangle=90, pctdistance=0.6)
#     plt.axis('equal')
#     plt.legend()
#     plt.show()
#
#     plt.figure(4)
#     plt.plot(x, list(result1), label='source code Seq2seq')
#     plt.plot(x, list(result2), label='source replace short VAR accuracy')
#     plt.scatter(x, list(result2), color='r', marker='^', s=20, label='replace VAR Seq2seq')
#     plt.scatter(x, list(result3), color='g', s=20, marker='s', label='repalce VAR classification')
#     plt.scatter(x, list(result4), color='b', s=20, marker='*', label='replace VAR modify vector classification')
#     plt.xlabel('tags')
#     plt.ylabel('accuracy')
#     plt.legend()
#     plt.show()
