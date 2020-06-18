import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


class tutorial_2_linear_regression():
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.fig_index = 1
        self.categorical_cols, self.numerical_cols = None, None
        self.feature_cols = None
        self.result = None

    def run(self):
        self.load_data()
        self.plot_data()
        self.make_feature_cols()
        self.linear_regression()

    def load_data(self):
        self.X_train = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
        self.X_test = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
        self.y_train = self.X_train.pop('survived')
        self.y_test = self.X_test.pop('survived')

        # columns type
        self.categorical_cols = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
        self.numerical_cols = ['age', 'fare']

    def plt_set_fig(self):
        plt.figure(self.fig_index)
        self.fig_index += 1

    def plot_data(self, show_plots=False):
        self.plt_set_fig()
        self.X_train.age.hist(bins=self.X_train.age.nunique())
        self.plt_set_fig()
        self.y_train.value_counts().plot(kind='bar')
        self.plt_set_fig()
        self.X_train.sex.value_counts().plot(kind='bar')
        self.plt_set_fig()
        self.X_train['class'].value_counts().plot(kind='bar')
        self.plt_set_fig()
        pd.concat([self.X_train, self.y_train], axis=1).groupby('sex').survived.mean().plot(kind='bar').set_xlabel('% survived')
        if show_plots: [plt.show(block=i) for i in range(1, self.fig_index)]

    def make_feature_cols(self):
        feature_cols = []
        for feature_name in self.categorical_cols:
            vocabulary = self.X_train[feature_name].unique()
            feature_cols.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
        for feature_name in self.numerical_cols:
            feature_cols.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
        self.feature_cols = feature_cols.copy()

    def make_input_func(self, data_df, label_df, num_epochs=1_000, shuffle=True, batch_size=32):
        def input_func():
            ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
            if shuffle: ds = ds.shuffle(1_000)
            ds = ds.batch(batch_size).repeat(num_epochs)
            return ds
        return input_func

    def linear_regression(self):
        train_input_func = self.make_input_func(data_df=self.X_train, label_df=self.y_train)
        test_input_func = self.make_input_func(data_df=self.X_test, label_df=self.y_test, num_epochs=1, shuffle=False)

        # model
        linear_est = tf.estimator.LinearClassifier(feature_columns=self.feature_cols)
        linear_est.train(train_input_func)
        result = linear_est.evaluate(test_input_func)
        self.result = result.copy()
        print(self.result)


if __name__ == '__main__':
    print(tf.version)
    t = tutorial_2_linear_regression()
    t.run()
