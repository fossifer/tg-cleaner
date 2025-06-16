import shap
import jieba
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from glob import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

seed = 42

# Set a font that supports Chinese characters
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # Correctly display negative signs

class TelegramMessageClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer(tokenizer=self.tokenize)
        self.classifier = MultinomialNB()
    
    def tokenize(self, message):
        tokens = list(jieba.cut(message, cut_all=False))
        return tokens

    def preprocess_messages(self, messages):
        return [' '.join(self.tokenize(message)) for message in messages]
    
    def load(self, path):
        self.vectorizer, self.classifier = joblib.load(path)

    def train(self, messages, labels, output='models/nb_model_and_vectorizer.joblib'):
        """
        训练分类器
        
        :param messages: 昵称列表
        :param labels: 对应的标签（spam/ham）
        """
        # 预处理昵称
        processed_messages = self.preprocess_messages(messages)
        
        # 将数据分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            processed_messages, labels, test_size=0.2, random_state=seed
        )
        
        # 特征向量化
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # 训练分类器
        self.classifier.fit(X_train_vectorized, y_train)
        
        # 在测试集上进行预测
        y_pred = self.classifier.predict(X_test_vectorized)
        
        # 评估模型
        train_accuracy = self.classifier.score(X_train_vectorized, y_train)
        test_accuracy = self.classifier.score(X_test_vectorized, y_test)
        
        print(f"训练集准确率: {train_accuracy:.2%}")
        print(f"测试集准确率: {test_accuracy:.2%}")

        # 计算精确率、召回率和F1分数
        precision = precision_score(y_test, y_pred, pos_label=True)
        recall = recall_score(y_test, y_pred, pos_label=True)
        f1 = f1_score(y_test, y_pred, pos_label=True)
        
        # 打印详细的分类报告
        print("分类报告:")
        print(classification_report(y_test, y_pred))
        print(f"Spam精确率: {precision:.2%}")
        print(f"Spam召回率: {recall:.2%}")
        print(f"Spam F1分数: {f1:.2%}")

        # 分析错误分类样本
        print("\n错误分类样本分析:")
        misclassified_indices = np.where(y_test != y_pred)[0]

        # 限制打印前20个错误分类样本
        print(f"共有 {len(misclassified_indices)} 个错误分类样本")
        print("前20个错误分类样本：")
        for idx in misclassified_indices[20:60]:
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            nickname = X_test[idx]
            prob_spam = self.classifier.predict_proba(X_test_vectorized[idx])[0][1]

            print(f"昵称: {nickname}")
            print(f"真实标签: {true_label}")
            print(f"预测标签: {pred_label}")
            print(f"Spam概率: {prob_spam:.2%}")
            print("-" * 30)

        joblib.dump((self.vectorizer, self.classifier), output)
    
    def predict(self, messages):
        """
        预测昵称是否为spam
        
        :param messages: 待预测的昵称列表
        :return: 预测结果（spam/ham）
        """
        processed_messages = self.preprocess_messages(messages)
        processed_vectorized = self.vectorizer.transform(processed_messages)
        predictions = self.classifier.predict(processed_vectorized)
        return predictions
    
    def predict_proba(self, messages):
        """
        获取spam的概率
        
        :param messages: 待预测的昵称列表
        :return: spam概率
        """
        processed_messages = self.preprocess_messages(messages)
        processed_vectorized = self.vectorizer.transform(processed_messages)
        spam_probabilities = self.classifier.predict_proba(processed_vectorized)[:, 1]
        return spam_probabilities
    
    def get_vectorized(self, X_raw):
        X_processed = self.preprocess_messages(X_raw)
        return self.vectorizer.transform(X_processed)

def normalize_value(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, (int, float)):
        return bool(value)
    elif isinstance(value, str):
        lower_value = value.lower()
        if lower_value in ('true', 't', '1', 'yes', 'y'):
            return True
        else:
            return False
    else:
        return False

def main():
    # 读取CSV数据
    df_main = pd.read_csv('data/messages.csv')
    # Fill missing values
    df_main.fillna({'nickname': '__MISSING__'}, inplace=True)

    # Load all exported CSVs
    export_files = glob('data/export/*.csv')
    export_dfs = []

    for file in export_files:
        df_export = pd.read_csv(file)
        df_export.fillna({'nickname': '__MISSING__'}, inplace=True)

        # Normalize is_spam
        df_export['is_spam'] = df_export['is_spam'].apply(normalize_value)
        export_dfs.append(df_export)
        # Normalize col name
        df_export['content'] = df_export['message_text']

    df = pd.concat([df_main] + export_dfs, ignore_index=True)

    # Concat input string fields
    df['input'] = df['nickname'] + " [SEP] " + df['content']
    
    # 准备数据
    messages = df['input'].tolist()
    labels = df['is_spam'].tolist()

    # 初始化分类器
    classifier = TelegramMessageClassifier()
    
    # 训练分类器
    # classifier.train(messages, labels, output='models/nb_20250502.joblib')
    
    classifier.load('models/nb_20250502.joblib')

    # 预测示例
    test_messages = [
        "Yvonne Watson [SEP] 6啊   哪个路子有他野     @wikipedia       牛逼 一小时搞三万 我在外面搬砖都没有那么多 真牛逼",  # spam
        '围棋澜澜 [SEP] 还有一个月过年了，还在找项慕的可以过来瞧瞧我这项慕，问问终究是好的，有时间得来    @wikibooks',  # spam
        'e [SEP] 阿道夫的股票跌了气急败坏了\n毕竟辉达大股东',  # ham
        'C World [SEP] https://github.com/Alex313031/thorium/issues/307',  # ham
        'aaa llll-散修ある [SEP] 【自动驾驶VS辅助驾驶【离谱动画《柴来了》】-哔哩哔哩】 https://b23.tv/XL8TgQd',  # ham
        'K_M101918 [SEP] 有爸爸羞辱调教我吗？湿了想要了😑',  # spam
        'grisha_in919 [SEP] 有没睡的搭子交流聊涩涩吗 想湿了🫥',  # spam
        'ISSAM Ucjgoho [SEP] 日 供 应 上 万 笔 转 账 能 量 认 准 机 器 人 ID ： @wikipedia_zh %2F4,L/34zY(9g',  # spam
        'dhjdjjde [SEP] 执行力强的来 一天5o o o. @yuchen37408',  # spam
        'ME [SEP] 主流 山寨 現貨 👈👈  看行情交流进🫰🫰',   # spam
        'Lucky Cat [SEP] 简单操作跟我过好日子   @Gooblaj2',  # spam
        '太有钱了 发了1288u👉 @kofunt',  # spam
        '嘀滴wzyz22 Agbomon [SEP] 机会就在眼前-主页了解',  # spam
        '嘀滴ynyz22 Delbole [SEP] 几天赚了几万 一个月可以奔驰c 抓紧时间.做的主夜',  # spam
        '嘀滴mqyz22 Chimboya [SEP] 没项目做？我们提供轻松的赚钱方式,竹页简介',  # spam
        '好 pppp222060999999 [SEP] 鹰峰同学',  # ham
        '好 pppp222060999999 [SEP] 大家好',  # ham
        '好 pppp222060999999 [SEP] 风口新项目纯绿色招人，动动手一天保底9k看我签!铭',  # spam
        '派小星 [SEP] 好想要！有喜欢小熟女的吗？勃起的有吗🤣',  # spam
        'Erika [SEP] 什么⁉  想要撸没好片，来这里吧  支持发图搜索大片  @seseyibot  （全网最齐全）',  # spam
        'HOSSAIN MD SAZZAD [SEP] I want to Learn Biotechnology',  # ham
        '诗诗y [SEP] 失眠，有聊的吗？叫我公主给看内搭🕺',  # spam
        'M [SEP] 我操他妈的，飞机上骗子怎么这么多！直到我遇见他 @WWW_USDT699 才知道一个小时2W 是多么简单，现在深圳都已经买了一套房了',  # spam
        "叶 枫 [SEP] github挂个梯子吧",  # ham
        "钓鱼 [SEP] 你见过这样的人吗？ @www_usdt699 是如何在一天内完成一辆奔驰价值交易的？这简直是太震撼了！",  # spam
    ]
    predictions = classifier.predict(test_messages)
    probabilities = classifier.predict_proba(test_messages)
    
    for nickname, pred, prob in zip(test_messages, predictions, probabilities):
        print(f"昵称: {nickname}, 预测: {pred}, Spam概率: {prob:.2%}")

    return

    messages_to_explain = [
        "叶 枫 [SEP] github挂个梯子吧",
    ]
    # 获取要解释的样本的向量化特征
    X_to_explain = classifier.get_vectorized(messages_to_explain).toarray()
    
    # 创建一个可调用的模型包装函数
    def model_predict(x):
        return classifier.classifier.predict_proba(x)[:, 1]  # 返回正类（spam）的概率
    
    # 获取背景数据
    processed_messages = classifier.preprocess_messages(messages)
    X_train, _, _, _ = train_test_split(
        processed_messages, labels, test_size=0.2, random_state=seed
    )
    X_background = classifier.vectorizer.transform(X_train).toarray()
    
    # 使用KernelExplainer，它适用于任何模型
    explainer = shap.KernelExplainer(
        model_predict, 
        shap.sample(X_background, 100),  # 使用样本作为背景数据以提高效率
        feature_names=classifier.vectorizer.get_feature_names_out()
    )
    
    # 计算SHAP值
    shap_values = explainer.shap_values(X_to_explain)
    
    # 显示waterfall图
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[0], 
        base_values=explainer.expected_value,
        data=X_to_explain[0],
        feature_names=classifier.vectorizer.get_feature_names_out()
    ), max_display=20)
    plt.tight_layout()
    # plt.savefig('shap_waterfall.png')
    plt.close()
    
    # 显示决策图
    plt.figure(figsize=(12, 8))
    shap.decision_plot(
        explainer.expected_value, 
        shap_values[0], 
        X_to_explain[0],
        feature_names=classifier.vectorizer.get_feature_names_out()
    )
    plt.tight_layout()
    # plt.savefig('shap_decision_plot.png')
    plt.close()
    
    # 打印预测结果
    pred = classifier.predict(messages_to_explain)
    prob = classifier.predict_proba(messages_to_explain)
    print(f"消息: {messages_to_explain[0]}")
    print(f"预测: {'Spam' if pred[0] else 'Ham'}, 概率: {prob[0]:.2%}")
    print(f"SHAP解释已保存为图片文件")

if __name__ == "__main__":
    main()
