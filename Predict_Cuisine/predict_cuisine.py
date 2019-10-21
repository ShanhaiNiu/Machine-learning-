import json
import codecs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data
train_filename='train.json'
train_content = pd.read_json(codecs.open(train_filename, mode='r', encoding='utf-8'))
test_filename = 'test.json'
test_content = pd.read_json(codecs.open(test_filename, mode='r', encoding='utf-8'))

print("菜名数据集一共包含 {} 训练数据 和 {} 测试样例。\n".format(len(train_content), len(test_content)))
if len(train_content)==39774 and len(test_content)==9944:
    print("数据成功载入！")
else:
    print("数据载入有问题，请检查文件路径！")
categories=np.unique(train_content['cuisine'])

train_ingredients = train_content['ingredients']
train_cuisine = train_content['cuisine']

#count ingredients
from collections import Counter
ingredients_array = []
for ingredient in train_ingredients:
    for single in range(len(ingredient)):
        ingredients_array.append(ingredient[single])
sum_ingredients = Counter(ingredients_array)
print(sum_ingredients)

#count italian_ingredients
italian_content = train_content[train_content['cuisine'] == 'italian']
italian_array = []
for ingredient in italian_content['ingredients']:
    for single in range(len(ingredient)):
        italian_array.append(ingredient[single])
italian_ingredients = Counter(italian_array)
print(italian_ingredients)

# clean data 
import re
from nltk.stem import WordNetLemmatizer
import numpy as np

def text_clean(ingredients):
    #remove punctuation
    ingredients= np.array(ingredients).tolist()
    print("菜品佐料：\n{}".format(ingredients[9]))
    ingredients=[[re.sub('[^A-Za-z]', ' ', word) for word in component]for component in ingredients]
    print("去除标点符号之后的结果：\n{}".format(ingredients[9]))
	# sample word
    lemma=WordNetLemmatizer()
    ingredients=[" ".join([ " ".join([lemma.lemmatize(w) for w in words.split(" ")]) for words in component])  for component in ingredients]
    print("去除时态和单复数之后的结果：\n{}".format(ingredients[9]))
    return ingredients

print("\n处理训练集...")
train_ingredients = text_clean(train_content['ingredients'])
print("\n处理测试集...")
test_ingredients = text_clean(test_content['ingredients'])

from sklearn.feature_extraction.text import TfidfVectorizer
# deal with train_tfidf
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1),
                analyzer='word', max_df=.57, binary=False,
                token_pattern=r"\w+",sublinear_tf=False)
train_tfidf = vectorizer.fit_transform(train_ingredients).todense()

# deal with test_tfidf
test_tfidf = vectorizer.transform(test_ingredients)

train_targets=np.array(train_content['cuisine']).tolist()
train_targets[:10]

# split data
from sklearn.model_selection import train_test_split

X_train , X_valid , y_train, y_valid = train_test_split(train_tfidf, train_targets,test_size=0.2, random_state =42)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# create model
parameters = {'C':[1,2,3,4,5,6,7,8,9,10]}
classifier = LogisticRegression()
grid = GridSearchCV(classifier, parameters)
grid = grid.fit(X_train, y_train)

from sklearn.metrics import accuracy_score 
valid_predict = grid.predict(X_valid)
valid_score=accuracy_score(y_valid,valid_predict)
print("验证集上的得分为：{}".format(valid_score))

# test
predictions = grid.predict(test_tfidf)
print("预测的测试集个数为：{}".format(len(predictions)))
test_content['cuisine']=predictions
test_content.head(10)

# save result
submit_frame = pd.read_csv("sample_submission.csv")
result = pd.merge(submit_frame, test_content, on="id", how='left')
result = result.rename(index=str, columns={"cuisine_y": "cuisine"})
test_result_name = "tfidf_cuisine_test.csv"
result[['id','cuisine']].to_csv(test_result_name,index=False)

