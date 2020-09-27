# Import Statements
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

# -*- coding: utf-8 -*-
'''
Created on Sun Apr 28 16:19:16 2019

@author: carlosbrown

Useful data science helper classes
'''


class DStats:
    '''Helpful methods for data science with statistical theme.'''

    def __init__(self, *args):
        self.data = []
        for item in args:
            self.data.append(item)

    def draw_perm_reps(self, func, size=1):
            """Generate multiple permutation replicates."""

            # Initialize array of replicates: perm_replicates
            perm_replicates = np.empty(size)

            for i in range(size):
                # Generate permutation sample
                perm_sample_1, perm_sample_2 = self.permutation_sample(
                        self.data[0], self.data[1])

                # Compute the test statistic
                perm_replicates[i] = func(perm_sample_1, perm_sample_2)

            return perm_replicates

    def permutation_sample(self, data1, data2):
            """Generate a permutation sample from two data sets."""

            # Concatenate the data sets: data
            data = np.concatenate((data1, data2))

            # Permute the concatenated array: permuted_data
            permuted_data = np.random.permutation(data)

            # Split the permuted array into two: perm_sample_1, perm_sample_2
            perm_sample_1 = permuted_data[:len(data1)]
            perm_sample_2 = permuted_data[len(data1):]

            return perm_sample_1, perm_sample_2

    def diff_of_means(self, data1, data2):
        return np.mean(data1) - np.mean(data2)

    def draw_bs_reps(self, data, func, size=1):
            """Draw bootstrap replicates."""

            # Initialize array of replicates: bs_replicates
            bs_replicates = np.empty(shape=size)

            # Generate replicates
            for i in range(size):
                bs_replicates[i] = self.bootstrap_replicate_1d(data, func)

            return bs_replicates

    def bootstrap_replicate_1d(self, data, func):
        """Draw bootstrap replicate"""

        boot_sample = np.random.choice(data, size=data.shape[0], replace=True)

        return func(boot_sample)

    def conf_int_boot(self, data, conf=95):
        '''Create bootstrap confidence interval of mean'''

        x1 = (100-conf) / 2
        x2 = 100 - x1
        boot_samples = self.draw_bs_reps(data, np.mean, size=3000)
        return tuple(np.percentile(boot_samples, [x1, x2]))

    def conf_int_param(self, data, dist='t', df=None, conf=95):
        '''Compute parametric confidence interval for mean, t or normal.'''

        if type(data) != np.ndarray:
            data = np.array(data)

        # Compute distribution parameters for confidence interval
        mu = np.mean(data)
        std = np.std(data)
        n = data.shape[0]

        if dist == 't':
            df = n - 1
            SE = std / np.sqrt(df)
            return stats.t.interval(alpha=conf/100, df=df, loc=mu, scale=SE)
        else:
            SE = std / np.sqrt(n)
            return stats.norm.interval(alpha=conf/100, loc=mu, scale=SE)

    def pearsonr_ci(self, x, y, alpha=0.05):
        '''Calculate Pearson correlation along with the confidence interval
        using scipy and np
        Parameters
        ----------
        x, y : iterable object such as a list or np.array
          Input for correlation calculation
        alpha : float
          Significance level. 0.05 by default
        Returns
        -------
        r : float
          Pearson's correlation coefficient
        pval : float
          The corresponding p value
        lo, hi : float
          The lower and upper bound of confidence intervals
        '''

        r, p = stats.pearsonr(x, y)
        r_z = np.arctanh(r)
        se = 1 / np.sqrt(x.size - 3)
        z = stats.norm.ppf(1 - alpha / 2)
        lo_z, hi_z = r_z - z*se, r_z + z*se
        lo, hi = np.tanh((lo_z, hi_z))
        return r, p, lo, hi

    def ecdf(self, data):
        """Compute ECDF for a one-dimensional array of measurements."""
        # Number of data points: n
        if type(data) == np.ndarray:
            n = data.shape[0]
        else:
            n = len(data)

        # x-data for the ECDF: x
        x = np.sort(data)

        # y-data for the ECDF: y
        y = np.arange(1, n+1) / n

        return x, y

    def statsmodel_ols(self, df, target, features, suppress_out=False):
        '''Builds an OLS model from pandas DataFrame, prints summary.'''

        # Create string of model notation for StatsModel
        model = target.strip() + ' ~ ' + ' + '.join(features)

        try:
            m = ols(model, df).fit()
            if not suppress_out:
                print(m.summary())
                self.vif(df, features)
                plt.figure()
                sns.set_style('darkgrid')
                sns.scatterplot(np.arange((m.resid.shape[0])), m.resid). \
                    set_title('Model Residuals')
                plt.figure()
                self.qqplot(m.resid)
                plt.figure()
                plt.title('Residual Distribution with normal')
                sns.distplot(m.resid, kde=False, fit=stats.norm)
        except Exception as E:
            print(E.args)
            return None
        return m

    def vif(self, df, features):
        print('\n**Variance Inflation Factors**\n')
        for index, var in enumerate(features):
            print('{} has VIF of {:.2f}'.format(var, variance_inflation_factor(
                    df[features].values, index)))

    def qqplot(self, data):
        '''Plot Q-Q plot of data against normal distribution.'''
        if type(data) != np.ndarray:
            try:
                data = np.array(data)
            except Exception as E:
                print(E.args)
                return

        sample_mean = np.mean(data)
        sample_std = np.std(data)

        # Generate normal distribution for plotting
        normal = np.random.normal(loc=sample_mean, scale=sample_std, size=100)
        percentiles = np.arange(0, 105, 5)

        # Generate Quantiles of sample and normal distribution
        quantile_data = np.percentile(data, percentiles)
        quantile_normal = np.percentile(normal, percentiles)

        # Construct Q-Q plot
        sns.scatterplot(quantile_normal, quantile_data).set_title('Q-Q Plot')
        x = np.linspace(np.min((quantile_data.min(), quantile_normal.min())),
                        np.max((quantile_data.max(), quantile_normal.max())))
        sns.lineplot(x, x, color="k")


class DLearn:
    '''Helpful methods for doing data science from a machine learning view.'''

    def __init__(self, data):
        pass

    def evalclusters(self, data, ks=range(1, 6)):
        '''Function to fit multiple KMeans models for evaluating inertia
        on one dataset.
        '''
        inertias = []
        for k in ks:
            model = KMeans(n_clusters=k)
            model.fit(data)
            inertias.append(model.inertia_)
        # Plot ks vs inertias
        plt.plot(ks, inertias, '-o')
        plt.xlabel('number of clusters, k')
        plt.ylabel('inertia')
        plt.title('Elbow Plot - KMeans Clustering')
        plt.xticks(ks)
        plt.show()

    def select_features(self, df, pred, target, linear=True):
        '''Method to select relevant features of a linear/non-linear model
        Avoids data leakage by running feature selection on the train/test
        split
        '''
        X_train, _, y_train, _ = train_test_split(df[pred], df[target],
                                                  test_size=0.2)
        if linear:
            model = Lasso()
        else:
            model = RandomForestRegressor()
        X_train = X_train.values.reshape(-1, 1)
        y_train = y_train.values.reshape(-1, 1)
        model.fit(X_train, y_train)
        rel_features = []

        for index, coef in enumerate(model.coef_):
            if coef != 0:
                rel_features.append(pred[index])
        return rel_features

    def cv_optimize(self, clf, parameters, Xtrain, ytrain, n_folds=5):
        '''Tune hyperparameters of untrained machine learning model using
        k-fold cross validation.
        '''
        gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds)
        gs.fit(Xtrain, ytrain)
        print("BEST PARAMS", gs.best_params_)
        return gs.best_estimator_

    def do_classify(self, clf, parameters, indf, featurenames, targetname,
                    target1val, standardize=False, train_size=0.8):
        '''run classification
        '''
        subdf = indf[featurenames]
        if standardize:
            subdfstd = (subdf - subdf.mean()) / subdf.std()
        else:
            subdfstd = subdf

        X = subdfstd.values
        y = (indf[targetname].values == target1val) * 1

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
                                                        train_size=train_size)
        clf = self.cv_optimize(clf, parameters, Xtrain, ytrain)
        clf.fit(Xtrain, ytrain)

        training_accuracy = clf.score(Xtrain, ytrain)
        test_accuracy = clf.score(Xtest, ytest)

        print("Accuracy on training data: {:0.2f}".format(training_accuracy))
        print("Accuracy on test data:     {:0.2f}".format(test_accuracy))
        return clf, Xtrain, ytrain, Xtest, ytest

    def getrocdata(self, clf, X_test, y_test):
        '''Calculate ROC curve and AUC of classifier.
        '''

        y_pred_proba = clf.predict_proba(X_test)
        fpr, tpr, thresh = roc_curve(y_test, y_pred_proba[:, 1])
        auc_ret = auc(fpr, tpr)
        print('Area under curve:', auc)
        return fpr, tpr, auc_ret

    def plot_roc_curve(self, fpr, tpr):
        '''Plots roc curve given false positive rate (fpr), and true positive
        rate (tpr).
        '''

        plt.plot(fpr, tpr, label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, classes,
                              normalize=False, title=None,
                              cmap=plt.cm.Blues):
        '''Prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`. Taken from
        sklearn website and modified.
        '''
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        ax.grid('off')
        return ax

class extendPandas:
    '''Helper functions to extend pandas manipulations for common data wrangling tasks'''
    def __init__(self, *args):
        pass
    
    def rolling_average(self, df_roll, periods, column, id_col, date_col='date'):
        '''Perform rolling window calculation on dataframe with non-consecutive dates in the index'''
        df = df_roll.copy()
        try:
            df_pivot = df.pivot(index=id_col, columns=date_col, values=column)
        except ValueError as e:
            print(e)
        df_pivot_roll = df_pivot.rolling(window=periods, axis=1).mean()
        df_pivot_roll = df_pivot_roll.reset_index(drop=False)
        var_name = column + '_rolling'
        df_melt = df_pivot_roll.melt(id_vars=id_col, var_name=var_name)
        df_melt = df_melt.reset_index(drop=False)

        df = df.merge(df_melt, on=[date_col, id_col])

        return df

if __name__ == '__main__':
    # Example usage
    boston = load_boston()
    bos = pd.DataFrame(boston.data)
    bos.columns = boston['feature_names']
    bos['PRICE'] = boston.target

    features = list(boston['feature_names'])
    data = DStats()
    model = data.statsmodel_ols(bos, 'PRICE', features, suppress_out=False)
