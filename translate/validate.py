from prettytable import PrettyTable

path = '/home/qiaoyang/pythonProject/Github_Crawler/data/'

#calculte jaccard similarity
def jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality / float(union_cardinality)


def get_real_and_predict():
    predicts = open(path + 'my_result.txt', 'r').read().split('\n')
    reals = open(path + 'label.test', 'r').read().split('\n')
    print len(predicts)
    print len(reals)
    return predicts,reals


#validate using jaccard
def validate_jaccard(predicts,reals):
    jaccard_all = 0
    for real,predict in zip(reals,predicts):
        r = str(real).split(' ')
        p = str(predict).split(' ')
        jaccard_all +=jaccard_similarity(r,p)
        #print jaccard_similarity(r,p)
    print "all jaccard_similarity is "+ str(jaccard_all/len(reals))


#validate using precision_recall
def validate_precison_recall(predicts,reals):
    labels_list = open(path+'frequecy_library_list').read().split('\n')
    label_size = len(labels_list)
    count_list = []
    catch_list = []
    right_list = []
    precision_list = []
    recall_list = []
    for i in range(label_size):
        count_list.append(0)
        catch_list.append(0)
        right_list.append(0)

    for real, predict in zip(reals, predicts):
        rs = set(real.split(' '))
        ps = set(predict.split(' '))
        index = 0
        for label in labels_list:
            if(label in rs):
                count_list[index] +=1
                if (label in ps):
                    right_list[index] += 1
            if(label in ps):
                catch_list[index] +=1
            index +=1

    for name,count,catch,right in zip(labels_list,count_list,catch_list,right_list):
        if((count<1) or (catch<1)):
           precison = '0'
           recall = '0'
        else:
            precison = str(round(float(right)/float(catch),4))
            recall =str( round(float(right)/float(count),4))
        precision_list.append(precison)
        recall_list.append(recall)

    labels_list,count_list,precision_list,recall_list = (list(x) for x in zip(*sorted(zip(labels_list,count_list,precision_list,recall_list), key=lambda pair: pair[2],reverse=True)))

    table = PrettyTable(["dependency name", "count", "precison", "recall"])
    table.align["dependency name"] = "l"
    table.padding_width = 1

    for l,c,p,r in zip(labels_list,count_list,precision_list,recall_list):
        table.add_row([l,c,p,r])
        #print str(l)+' count is '+str(c)+' precision is '+str(p)+' recall is '+str(r)
    table_txt = table.get_string()
    print table_txt
    with open(path+'result_compare.txt', 'w') as file:
        file.write(table_txt)


if __name__ == "__main__":
    predicts, reals = get_real_and_predict()
    validate_jaccard(predicts,reals)
    validate_precison_recall(predicts,reals)