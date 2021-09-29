#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
func: 推文处理
"""
import re
import stanza
from nltk.corpus import stopwords
nlp = stanza.Pipeline("en", processors='tokenize,ner')
cut = stanza.Pipeline("en", processors="tokenize, mwt, pos, lemma")
# twitter中常见链接
SPECIAL_LINK = re.compile(r"(https:/…)")
twitter_link = re.compile(r"(https://t.co/[a-zA-Z0-9]+)")


class TwitterPreprocessor:
    def __init__(self):
        self.SPECIAL_SYMBOL_RE = re.compile(r'[^\w\s\u4e00-\u9fa5]+')  # 删除特殊符号
        self.TWEET = re.compile(r"(^|[^@\w])@(\w{1,15})\b")  # 匹配@用户信息
        self.hash_tag = re.compile(r'#\w*')
        self.mention = re.compile(r"( RT | FAV | VIA | rt | fav | via )")  # 推特保留字
        self.link = re.compile(
            r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|'
            r'https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})')
        # 停用词
        self.stopwords = stopwords.words('english')

    def clean_text(self, text: str):
        """
        清理文本
        :param text: str
        :return: str
        """
        text = text.replace('\n', ' ').replace('\r', ' ')  # 去除回车符和换行符
        # 去除推特用户名
        text = re.sub(pattern=self.TWEET, repl=' ', string=text)
        # 去除http链接,顺序不可颠倒
        text = re.sub(pattern=self.link, repl=' ', string=text)
        text = re.sub(pattern=twitter_link, repl=' ', string=text)
        text = re.sub(pattern=SPECIAL_LINK, repl=' ', string=text)
        # 去除特殊字符
        text = re.sub(pattern=self.SPECIAL_SYMBOL_RE, repl=' ', string=text)
        # 去除推特保留词语
        text = re.sub(pattern=self.mention, repl=' ', string=text)
        text = text.strip()
        sentence = text.strip().replace('。', '').replace('」', '').replace('//', '').replace('_', '') \
            .replace('-', '').replace('\t', '').replace('@', '') \
            .replace(r'\\', '').replace("''", '')
        return sentence

    def entity_recognition(self, text: str):
        """
        命名实体识别
        :param text: str
        :return: [entity]
        """
        # 将hashtag作为实体的一部分
        tag = re.findall(pattern=self.hash_tag, string=text)
        text = self.clean_text(text)
        # 命名实体识别
        doc = nlp(text)
        ents = []
        for sentence in doc.sentences:
            for ent in sentence.ents:
                if ent.type in ["ORG", "PERSON", "GPE", 'WORK_OF_ART']:
                    ents.append(ent.text)
        for t in tag:
            ents.append(t[1:])
        return ents  # 实体列表

    def get_token(self, text: str):
        """
        文本分词
        :param text: str
        :return: [words]
        """
        text = self.clean_text(text)
        doc = cut(text)
        words = []
        for sent in doc.sentences:
            for word in sent.words:
                w = word.lemma
                if len(w) > 1:
                    if not w.isdigit() and w.isalpha():
                        w = w.lower()
                        if w not in self.stopwords:
                            words.append(w)
        return words

    def rule_counter(self, text: str):
        """
        根据规则统计文本信息
        :param text:
        :return: False(代表该条文本应该被过滤)，True(代表该文本应该保留)
        """
        # employ the regular expression to match usermentions, hashtags, URL
        res1 = re.findall(pattern=self.link, string=text)
        res2 = re.findall(pattern=SPECIAL_LINK, string=text)
        res3 = re.findall(pattern=twitter_link, string=text)
        if len(res1) + len(res2) + len(res3) > 3:
            return False
        res4 = re.findall(pattern=self.mention, string=text)
        if len(res4) > 3:
            return False
        res5 = re.findall(pattern=self.hash_tag, string=text)
        if len(res5) > 3:
            return False
        # 如果没有命名实体返回false
        ents = self.entity_recognition(text)
        if not ents:
            return False
        # 如果分词数量小于6返回false
        words = self.get_token(text)
        if len(words) < 6:
            return False
        return True
