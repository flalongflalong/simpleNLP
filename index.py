import os
import pickle
import shutil
import jieba
from collections import defaultdict
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer  # TF_IDF向量生成类
from sklearn.datasets.base import Bunch
from sklearn.naive_bayes import MultinomialNB  # 多项式贝叶斯算法

####   通用方法    #####
def readFile(path):
    with open(path, 'r', errors='ignore') as file:  # 文档中编码有些问题，所有用errors过滤错误
        content = file.read()
        return content

def saveFile(path, result):
    with open(path, 'w', errors='ignore') as file:
        file.write(result)

def del_file(path_data):
    for i in os.listdir(path_data) :# os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "\\" + i#当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:#os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            shutil.rmtree(file_data)

def readBunch(path):
    with open(path, 'rb') as file:
        bunch = pickle.load(file)   #从打开的文件对象 文件中读取pickle对象
        # pickle.load(file)
        # 函数的功能：将file中的对象序列化读出。
    return bunch

def writeBunch(path, bunchFile):
    with open(path, 'wb') as file:
        pickle.dump(bunchFile, file)    #将obj对象的编码pickle编码表示写入到文件对象中

def open_dict(Dict, path):
    path = path + '%s.txt' % Dict
    dictionary = open(path, 'r', encoding='utf-8')
    dict = []
    for word in dictionary:
        word = word.strip('\n')
        dict.append(word)
    return dict
#####   通用方法    #####


######  监督式分类   ######
# 通过jieba分词，去除停用词，并保存文件
def segText(inputPath, resultPath):
    fatherLists = os.listdir(inputPath)  # 主目录
    for eachDir in fatherLists:  # 遍历主目录中各个文件夹
        eachPath = inputPath + eachDir + "/"  # 保存主目录中每个文件夹目录，便于遍历二级文件
        each_resultPath = resultPath + eachDir + "/"  # 分词结果文件存入的目录
        if not os.path.exists(each_resultPath):
            os.makedirs(each_resultPath)
        childLists = os.listdir(eachPath)  # 获取每个文件夹中的各个文件
        for eachFile in childLists:  # 遍历每个文件夹中的子文件
            eachPathFile = eachPath + eachFile  # 获得每个文件路径
            #  print(eachFile)
            content = readFile(eachPathFile)  # 调用上面函数读取内容
            # content = str(content)
            result = (str(content)).replace("\r\n", "").strip()  # 删除多余空行与空格
            # result = content.replace("\r\n","").strip()
            cutResult = jieba.cut(result)  # 默认方式分词，分词结果用空格隔开
            saveFile(each_resultPath + eachFile, " ".join(cutResult))  # 调用上面函数保存文件

# 输入分词，输出分词向量
def bunchSave(inputFile, outputFile):
    catelist = os.listdir(inputFile)
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(catelist)  # 将类别保存到Bunch对象中
    for eachDir in catelist:
        eachPath = inputFile + eachDir + "/"
        fileList = os.listdir(eachPath)
        for eachFile in fileList:  # 二级目录中的每个子文件
            fullName = eachPath + eachFile  # 二级目录子文件全路径
            bunch.label.append(eachDir)  # 当前分类标签
            bunch.filenames.append(fullName)  # 保存当前文件的路径
            bunch.contents.append(readFile(fullName).strip())  # 保存文件词向量
    with open(outputFile, 'wb') as file_obj:  # 持久化必须用二进制访问模式打开
        pickle.dump(bunch, file_obj)    #将obj对象的编码pickle编码表示写入到文件对象中

# 求得TF-IDF向量
def getTFIDFMat(inputPath, stopWordList, outputPath):
    bunch = readBunch(inputPath)
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],  vocabulary={})
    # 初始化向量空间
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5)
    transformer = TfidfTransformer()  # 该类会统计每个词语的TF-IDF权值
    # 文本转化为词频矩阵，单独保存字典文件
    tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
    tfidfspace.vocabulary = vectorizer.vocabulary_  # 获取词汇
    # 持久化
    writeBunch(outputPath, tfidfspace)

#求测试集TF-IDF向量空间
def getTestSpace(testSetPath, trainSpacePath, stopWordList, testSpacePath):
    bunch = readBunch(testSetPath)
    # 构建测试集TF-IDF向量空间
    testSpace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[], vocabulary={})
    # 导入训练集的词袋
    trainbunch = readBunch(trainSpacePath)
    # 使用TfidfVectorizer初始化向量空间模型  使用训练集词袋向量
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5, vocabulary=trainbunch.vocabulary)
    transformer = TfidfTransformer()
    testSpace.tdm = vectorizer.fit_transform(bunch.contents)
    testSpace.vocabulary = trainbunch.vocabulary
    # 持久化
    writeBunch(testSpacePath, testSpace)

#使用sklearn中的贝叶斯分类器，进行分类
def bayesAlgorithm(trainPath, testPath):
    trainSet = readBunch(trainPath)
    testSet = readBunch(testPath)

    # 在scikit - learn中，一共有3个朴素贝叶斯的分类算法类。分别是GaussianNB，MultinomialNB和BernoulliNB。其中GaussianNB就是先验为高斯分布的朴素贝叶斯，MultinomialNB就是先验为多项式分布的朴素贝叶斯，而BernoulliNB就是先验为伯努利分布的朴素贝叶斯
    clf = MultinomialNB(alpha=0.001).fit(trainSet.tdm, trainSet.label)   # 加载多项式函数   构造基于数据的分类器

    # alpha:0.001 alpha 越小，迭代次数越多，精度越高
    # print(shape(trainSet.tdm))  #输出单词矩阵的类型
    # print(shape(testSet.tdm))
    predicted = clf.predict(testSet.tdm)     # 预测文档
    total = len(predicted)
    rate = 0
    for flabel, fileName, expct_cate in zip(testSet.label, testSet.filenames, predicted):
        if flabel != expct_cate:
            rate += 1
            print(fileName, "-->预测类别：", expct_cate)
    print("erroe rate:", float(rate) * 100 / float(total), "%")
######  监督式分类   ######


######  情感分析部分   ######
# 通过jieba分词，然后去除停用词
def seg_word(sentence):
    result = (str(sentence)).replace("\r\n", "").strip()  # 删除多余空行与空格
    seg_list = jieba.cut(result)  # 默认方式分词，分词结果用空格隔开
    seg_result = []
    for i in seg_list:
        seg_result.append(i)
    return list(filter(lambda x: x not in stopWordListFilter, seg_result))  #去除停用词

# 找出文本中的情感词、否定词和程度副词
def classify_words(word_list):
    sen_word = dict()
    not_word = dict()
    degree_word = dict()
    # 分类
    for i in range(len(word_list)):
        word = word_list[i]
        if word in sen_dict.keys() and word not in deny_word and word not in degree_dict.keys():
            # 找出分词结果中在情感字典中的词
            sen_word[i] = sen_dict[word]
        elif word in deny_word and word not in degree_dict.keys():
            # 分词结果中在否定词列表中的词
            not_word[i] = -1
        elif word in degree_dict.keys():
            # 分词结果中在程度副词中的词
            degree_word[i] = degree_dict[word]
    # 返回分类结果
    return sen_word, not_word, degree_word

# 计算情感词的分数
# 遍历所有的情感词，查看当前情感词的前面是否有否定词和程度副词，如果没有否定词，就对当前情感词乘以1，如果有否定词或者有多个否定词，可以乘以（-1）^否定词的个数；如果有程度副词，就在当前情感词前面乘以程度副词的程度等级。
def score_sentiment(sen_word, not_word, degree_word, seg_result):
    # 权重初始化为1
    W = 1
    score = 0
    # 情感词下标初始化
    sentiment_index = -1
    # 情感词的位置下标集合
    sentiment_index_list = list(sen_word.keys())
    # 遍历分词结果
    for i in range(0, len(seg_result)):
        # 如果是情感词
        if i in sen_word.keys():
            # 权重*情感词得分
            score += W * float(sen_word[i])
            # 情感词下标加一，获取下一个情感词的位置
            sentiment_index += 1
            if sentiment_index < len(sentiment_index_list) - 1:
                # 判断当前的情感词与下一个情感词之间是否有程度副词或否定词
                for j in range(sentiment_index_list[sentiment_index], sentiment_index_list[sentiment_index + 1]):
                    # 更新权重，如果有否定词，权重取反
                    if j in not_word.keys():
                        W *= -1
                    elif j in degree_word.keys():
                        W *= float(degree_word[j])
        # 定位到下一个情感词
        if sentiment_index < len(sentiment_index_list) - 1:
            i = sentiment_index_list[sentiment_index + 1]
    return score

# 计算一个内容的得分
def sentiment_one_score(sentence):
    # 1.对文档分词
    seg_list = seg_word(sentence)
    # for word in seg_list:
    #     print('word:'+word)

    # 2.将分词结果转换成字典，找出情感词、否定词和程度副词
    sen_word, not_word, degree_word = classify_words(seg_list)

    # 3.计算得分
    score = score_sentiment(sen_word, not_word, degree_word, seg_list)
    return score

# 计算某一文件路径下所有文本的情感分析
def sentiment_score(inputPath):
    fatherLists = os.listdir(inputPath)  # 主目录
    for eachDir in fatherLists:  # 遍历主目录中各个文件夹
        eachPath = inputPath + eachDir + "/"  # 保存主目录中每个文件夹目录，便于遍历二级文件
        # each_resultPath = resultPath + eachDir + "/"  # 分词结果文件存入的目录
        # if not os.path.exists(each_resultPath):
        #     os.makedirs(each_resultPath)
        childLists = os.listdir(eachPath)  # 获取每个文件夹中的各个文件
        for eachFile in childLists:  # 遍历每个文件夹中的子文件
            eachPathFile = eachPath + eachFile  # 获得每个文件路径
            #  print(eachFile)
            content = readFile(eachPathFile)  # 读取内容
            score = sentiment_one_score(content)
            scoreType = '中'
            if score>=1:
                scoreType = '好'
            elif score<=-1:
                scoreType = '差'
            else:
                scoreType = '中'

            print(eachPathFile, content, "-->情感分析：", scoreType)
######  情感分析部分   ######


######  ######设定程序所需的数据    ######
p = os.getcwd()  # 工程路径
stopWordList = open_dict('stopWord', p + "/dict/")  # 获取停用词
deny_word = open_dict('denyWord', p + '/dict/')     # 读取否定词文件
degree_word = open_dict('degreeDict', p + '/dict/') # 读取程度副词 做了一个简单的程度副词标记，大于1，表示情感加强，小于1，表示情感弱化，下面主要按照极其1.8，超1.6，很1.5，较1，稍0.7，欠0.5进行了一个简单的标记，如下所示。也可以根据自己的需求及及进行修改。
posdict = open_dict('positiveWord', p + '/dict/')   # 读取情感词典文件 褒义    情感分值设为  1 , 情感词的情感分值也可以根据自己的需求及及进行修改
negdict = open_dict('negativeWord', p + '/dict/')   # 读取情感词典文件 贬义    情感分值设为  -1

degree_list = []  # 程度副词列表
degree_dict = defaultdict()  # 创建程度副词字典
for i in degree_word:
    degree_dict[i.split(',')[0]] = i.split(',')[1]
    degree_list.append(i.split(',')[0])

sen_dict = defaultdict()  # 创建情感字典
for i in posdict:
    sen_dict[i.split(',')[0]] = 1   # 褒义词典转换成字典对象，key为情感词，value为其对应的权重 1
for i in negdict:
    sen_dict[i.split(',')[0]] = -1  # 贬义词典转换成字典对象，key为情感词，value为其对应的权重 -1

stopWordListFilter = []     # 生成新的停用词表 将否定词或者是程度副词的词典过滤掉
for word in stopWordList:
    if (word not in deny_word) and (word not in degree_list):
        stopWordListFilter.append(word)
######  ######设定程序所需的数据    ######


# 单一内容情感测试
# data1 = '网速有点慢，但比家里wifi稳定，一分钱一分货'
# print(data1, sentiment_one_score(data1))




# 删除临时文件
del_file(p + "/temp/")

# 根据已有分类的关键词进行训练
segText(p + "/keyData/", p + "/temp/segResult/")  # 分词
bunchSave(p + "/temp/segResult/", p + "/temp/train_set.dat")  # 输入分词，输出分词向量
getTFIDFMat(p + "/temp/train_set.dat", stopWordList, p + "/temp/tfidfspace.dat")  # 输入词向量，输出特征空间

# 评论进行分类
segText(p + "/testData/", p + "/temp/test_segResult/")  # 分词
bunchSave(p + "/temp/test_segResult/", p + "/temp/test_set.dat")  # 输入分词，输出分词向量
getTestSpace(p + "/temp/test_set.dat", p + "/temp/tfidfspace.dat", stopWordList, p + "/temp/testspace.dat")
bayesAlgorithm(p + "/temp/tfidfspace.dat", p + "/temp/testspace.dat")

# 情感分析数据
sentiment_score(p + "/testData/")
