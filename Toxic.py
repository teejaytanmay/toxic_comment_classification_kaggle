import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_union
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import ComplementNB
from imblearn.ensemble import EasyEnsembleClassifier

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from matplotlib_venn import venn3
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
sns.set(style="white", context="talk")

from scipy.sparse import hstack
import regex as re
import re as r
import string

class data_viz():
    def __init__(self,trainFile):
        self.train = pd.read_csv(trainFile).fillna(' ')
        self.class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.train['clean'] = (self.train[self.class_names].max(axis=1) == 0).astype(int)
        self.class_names.append('clean')
        self.classes = list(self.class_names)

    def df_distribution(self):
        df_distribution = self.train[self.class_names].sum()\
                            .to_frame()\
                            .rename(columns={0: 'count'})\
                            .sort_values('count')

        df_distribution.plot.pie(y='count', title='Label distribution over comments (without "none" category)',
                                figsize=(10, 10))\
                                .legend(loc='center left', bbox_to_anchor=(1.3, 0.5))

    def per_class_distribution(self):
        x=self.train.iloc[:,2:].sum()
        plt.figure(figsize=(13,7))
        ax= sns.barplot(x.index, x.values, alpha=1)
        plt.title(" Per class Distribution")
        plt.ylabel('Number of Occurrences')
        plt.xlabel('Type ')
        rects = ax.patches
        labels = x.values
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

        plt.show()

    def multi_tag(self):
        rowsums=self.train.iloc[:,2:].sum(axis=1)
        x=rowsums.value_counts()

        plt.figure(figsize=(13,7))
        ax = sns.barplot(x.index, x.values, alpha=0.8)
        plt.title("Multiple tags per comment")
        plt.ylabel('# of Occurrences', fontsize=12)
        plt.xlabel('# of tags ', fontsize=12)

        rects = ax.patches
        labels = x.values
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

        plt.show()

    def corr_mat(self):
        f, ax = plt.subplots(figsize=(9, 6))
        f.suptitle('Correlation matrix for categories')
        sns.heatmap(self.train[self.class_names].corr(), annot=True, linewidths=.5, ax=ax)

    def venn(self):
        t = self.train[(self.train['toxic'] == 1) & (self.train['insult'] == 0) & (self.train['obscene'] == 0)].shape[0]
        i = self.train[(self.train['toxic'] == 0) & (self.train['insult'] == 1) & (self.train['obscene'] == 0)].shape[0]
        o = self.train[(self.train['toxic'] == 0) & (self.train['insult'] == 0) & (self.train['obscene'] == 1)].shape[0]

        t_i = self.train[(self.train['toxic'] == 1) & (self.train['insult'] == 1) & (self.train['obscene'] == 0)].shape[0]
        t_o = self.train[(self.train['toxic'] == 1) & (self.train['insult'] == 0) & (self.train['obscene'] == 1)].shape[0]
        i_o = self.train[(self.train['toxic'] == 0) & (self.train['insult'] == 1) & (self.train['obscene'] == 1)].shape[0]

        t_i_o = self.train[(self.train['toxic'] == 1) & (self.train['insult'] == 1) & (self.train['obscene'] == 1)].shape[0]


        # Make the diagram
        plt.figure(figsize=(8, 8))
        plt.title("Venn diagram for 'toxic', 'insult' and 'obscene'")
        venn3(subsets = (t, i, t_i, o, t_o, i_o, t_i_o),
              set_labels=('toxic', 'insult', 'obscene'))
        plt.show()

    def venn_2(self):
        t = self.train[(self.train['toxic'] == 1) & (self.train['severe_toxic'] == 0)].shape[0]
        s = self.train[(self.train['toxic'] == 0) & (self.train['severe_toxic'] == 1)].shape[0]

        t_s = self.train[(self.train['toxic'] == 1) & (self.train['severe_toxic'] == 1)].shape[0]

        # Make the diagram
        plt.figure(figsize=(8, 8))
        plt.title("Venn diagram for 'toxic' and 'severe_toxic'")
        venn2(subsets = (t, s, t_s),
              set_labels=('toxic', 'severe_toxic'))
        plt.show()

    def comp_corr_mat(self):
        self.train['total_length'] = self.train['comment_text'].str.len()
        self.train['new_line'] = self.train['comment_text'].str.count('\n'* 1)
        self.train['new_small_space'] = self.train['comment_text'].str.count('\n'* 2)
        self.train['new_medium_space'] = self.train['comment_text'].str.count('\n'* 3)
        self.train['new_big_space'] = self.train['comment_text'].str.count('\n'* 4)

        self.train['new_big_space'] = self.train['comment_text'].str.count('\n'* 4)
        self.train['uppercase_words'] = self.train['comment_text'].apply(lambda l: sum(map(str.isupper, list(l))))
        self.train['question_mark'] = self.train['comment_text'].str.count('\?')
        self.train['exclamation_mark'] = self.train['comment_text'].str.count('!')

        FEATURES = ['total_length',
                    'new_line',
                    'new_small_space',
                    'new_medium_space',
                    'new_big_space',
                    'uppercase_words',
                    'question_mark',
                    'exclamation_mark']
        self.class_names += FEATURES

        f, ax = plt.subplots(figsize=(20, 20))
        f.suptitle('Correlation matrix for categories and features')
        sns.heatmap(self.train[self.class_names].corr(), annot=True, linewidths=.5, ax=ax)

    def clean_text(self,text):
        stop = stopwords.words('english')
        text = re.sub('[{}]'.format(string.punctuation), ' ', text.lower())
        return ' '.join([word for word in text.split() if word not in (stop)])

    def wordcloud(self):

        word_counter = {}

        for categ in self.classes:
            d = Counter()
            self.train[self.train[categ] == 1]['comment_text'].apply(lambda t: d.update(self.clean_text(t).split()))
            word_counter[categ] = pd.DataFrame.from_dict(d, orient='index')\
                                                .rename(columns={0: 'count'})\
                                                .sort_values('count', ascending=False)
        for w in word_counter:
            wc = word_counter[w]

            wordcloud = WordCloud(
                  background_color='black',
                  max_words=200,
                  max_font_size=100,
                  random_state=43
                 ).generate_from_frequencies(wc.to_dict()['count'])

            fig = plt.figure(figsize=(12, 8))
            plt.title(w)
            plt.imshow(wordcloud)
            plt.axis('off')

            plt.show()



class Toxicity():
    def __init__(self, trainFile, testFile):
        self.trainFile = trainFile
        self.testFile = testFile
        self.__een = EasyEnsembleClassifier(base_estimator=LogisticRegression(C=6,solver='sag', max_iter=500))
        self.__sgd = SGDClassifier(alpha=.0002, max_iter=180, penalty="l2", loss='modified_huber')
        self.__rforest = RandomForestClassifier(criterion='gini',max_depth=100, max_features=900, n_estimators=20, n_jobs=-1 ,min_samples_leaf=3, min_samples_split=10)
        self.__lr = LogisticRegression(C=6,solver='sag', max_iter=500)
        self.__vot = VotingClassifier(estimators=[ ('een', self.__een), ('sgd', self.__sgd),('lr',self.__lr),('rf',self.__rforest)], voting='soft', weights=[0.9,1.3,0.55,0.65])
        self.train_data = None
        self.test_data = None
        self.all_data = None
        self.train_features = None
        self.test_features = None
        self.test_labels = None
        self.train_labels = None
        self.class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.submissionFile = None
        self.score = {}

    def preprocessing(self, df):
        repl = {
            "yay!": " good ",
            "yay": " good ",
            "yaay": " good ",
            "yaaay": " good ",
            "yaaaay": " good ",
            "yaaaaay": " good ",
            ":/": " bad ",
            ":&gt;": " sad ",
            ":')": " sad ",
            ":-(": " frown ",
            ":(": " frown ",
            ":s": " frown ",
            ":-s": " frown ",
            "&lt;3": " heart ",
            ":d": " smile ",
            ":p": " smile ",
            ":dd": " smile ",
            "8)": " smile ",
            ":-)": " smile ",
            ":)": " smile ",
            ";)": " smile ",
            "(-:": " smile ",
            "(:": " smile ",
            ":/": " worry ",
            ":&gt;": " angry ",
            ":')": " sad ",
            ":-(": " sad ",
            ":(": " sad ",
            ":s": " sad ",
            ":-s": " sad ",
            r"\br\b": "are",
            r"\bu\b": "you",
            r"\bhaha\b": "ha",
            r"\bhahaha\b": "ha",
            r"\bdon't\b": "do not",
            r"\bdoesn't\b": "does not",
            r"\bdidn't\b": "did not",
            r"\bhasn't\b": "has not",
            r"\bhaven't\b": "have not",
            r"\bhadn't\b": "had not",
            r"\bwon't\b": "will not",
            r"\bwouldn't\b": "would not",
            r"\bcan't\b": "can not",
            r"\bcannot\b": "can not",
            r"\bi'm\b": "i am",
            "m": "am",
            "r": "are",
            "u": "you",
            "haha": "ha",
            "hahaha": "ha",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "hasn't": "has not",
            "haven't": "have not",
            "hadn't": "had not",
            "won't": "will not",
            "wouldn't": "would not",
            "can't": "can not",
            "cannot": "can not",
            "i'm": "i am",
            "m": "am",
            "i'll" : "i will",
            "its" : "it is",
            "it's" : "it is",
            "'s" : " is",
            "that's" : "that is",
            "weren't" : "were not",
        }

        keys = [i for i in repl.keys()]
        new_data = []
        ltr = df["comment_text"].tolist()
        for i in ltr:
            arr = str(i).split()
            xx = ""
            for j in arr:
                j = str(j).lower()
                if j[:4] == 'http' or j[:3] == 'www':
                    continue
                if j in keys:
                    j = repl[j]
                xx += j + " "
            new_data.append(xx)
        df["new_comment_text"] = new_data

        trate = df["new_comment_text"].tolist()
        for i, c in enumerate(trate):
            trate[i] = re.sub('[^a-zA-Z ?!]+', '', str(trate[i]).lower())
        df["comment_text"] = trate
        del trate
        df.drop(["new_comment_text"], axis=1, inplace=True)

    def cleanData(self, text):
        clean = bytes(text.lower().encode('utf-8'))
        clean = clean.replace(b"\n", b" ")
        clean = clean.replace(b"\t", b" ")
        clean = clean.replace(b"\b", b" ")
        clean = clean.replace(b"\r", b" ")

        exclude = r.compile(b'[%s]' % re.escape(bytes(string.punctuation.encode('utf-8'))))
        clean = b" ".join([exclude.sub(b'', token) for token in clean.split()])
        clean = r.sub(b"\d+", b" ", clean)
        clean = r.sub(b'\\s+', b' ', clean)
        clean = r.sub(b'\s+', b' ', clean)
        clean = r.sub(b'\s+$', b'', clean)
        clean = r.sub(b'\s*', b'', clean)

        return str(clean).encode('utf-8')

    def prepareToVectorize(self, df):
        df["clean_comment"] = df["comment_text"].apply(lambda x: self.cleanData(x))

    def vectorize(self, data, text):
        word_vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                tokenizer=lambda x: regex.findall(r'[^\p{P}\W]+', x),
                analyzer='word',
                token_pattern='(?u)\\b\\w\\w+\\b\\w{,1}',
                min_df=5,
                norm='l2',
                ngram_range=(1, 2),
                max_features=33000)

        word_vectorizer.fit(text)
        data_word_features = word_vectorizer.transform(data)

        char_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='char',
            token_pattern=None,
            min_df=5,
            ngram_range=(1, 4),
            norm='l2',
            max_features=24000)

        char_vectorizer.fit(text)
        data_char_features = char_vectorizer.transform(data)
        return hstack([data_word_features, data_char_features])


    def trainingData(self):
        df = pd.read_csv(self.trainFile).fillna(' ')
        self.preprocessing(df)
        self.prepareToVectorize(df)
        self.train_data = df["comment_text"]
        self.train_labels = df.drop(columns='comment_text')

    def testingData(self):
        df = pd.read_csv(self.testFile).fillna(' ')
        self.preprocessing(df)
        self.prepareToVectorize(df)
        self.test_data = df["comment_text"]
        self.test_labels = df.drop(columns='comment_text')

    def nbsvm(self, tf, tt):
        p = train_features[tt==tr].sum(0)
        return (p+1) / ((tt==tr).sum()+1)

    def data(self):
        self.trainingData()
        self.testingData()
        self.all_data = pd.concat([self.train_data, self.test_data])
        self.train_features = self.vectorize(self.train_data, self.all_data)
        self.test_features = self.vectorize(self.test_data, self.all_data)
        self.train_features.to_csv("train_features.csv")
        self.test_features.to_csv("test_features.csv")
        self.train_labels.to_csv("train_labels.csv")
        self.submissionFile = pd.DataFrame.from_dict({'Id': self.test_labels['Id']})

    def calc_nbsvm(self,class_label):
        return np.log(nbsvm(1,self.train_labels[class_label])/nbsvm(0,self.train_labels[class_label]))

    def cv_score(self,class_label):
        mul_fac = calc_nbsvm(class_label)
        self.score[class_label] = np.mean(cross_val_score(self.__vot, self.train_features.multiply(mul_fac), self.train_labels[class_label], cv=5, scoring='roc_auc'))

    def trainVotingClassifier(self, class_label):
        mul_fac =  calc_nbsvm(class_label)
        self.__vot.fit(self.train_features.multiply(mul_fac),self.train_labels[class_label])

    def testVotingClassifier(self, class_label):
        self.cv_score(class_label)
        print ('CV Score for class {} is {}'.format(class_label,self.score[class_label]))
        mul_fac = calc_nbsvm(class_label)
        self.submissionFile[class_label] = self.__vot.predict_proba(self.test_features.multiply(mul_fac))[:,1]

    def accuracy(self):
        print ('Total CV Score is {}'.format(np.mean(self.score.values())))

    def makeSubmission(self):
        self.submissionFile.to_csv("submission.csv", index=False)

    def make_pickle(self, class_label):
        model_pickle = open("Toxic_Comment_Classification_Model_"+class_label.title()+".pkl","wb")
        pickle.dump(self.__vot,model_pickle)
        model_pickle.close()

if __name__ == "__main__":
    train_data_name = 'train.csv'
    test_data_name = 'test.csv'

    model = Toxicity(train_data_name,test_data_name)
    model.data()

    visual = data_viz(train_data_name)
    visual.df_distribution()
    visual.multi_tag()
    visual.per_class_distribution()
    visual.corr_mat()
    visual.venn()
    visual.venn_2()
    visual.comp_corr_mat()
    visual.wordcloud()

    for label in model.class_names:
        model.trainVotingClassifier(label)
        model.testVotingClassifier(label)
        model.make_pickle(label)

    model.acuracy()
    model.make_submission()
