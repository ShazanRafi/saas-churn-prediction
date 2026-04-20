from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class FeatureConstructor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.low_watch_threshold = X['ViewingHoursPerWeek'].quantile(0.3)
        return self

    def transform(self, X):
        X = X.copy()

        if 'CustomerID' in X.columns:
            X = X.drop(columns=['CustomerID'])

        X['engagement_score'] = (X['ViewingHoursPerWeek'] + X['AverageViewingDuration'] + X['ContentDownloadsPerMonth'])
        X['cost_per_hour'] = X['MonthlyCharges'] / (X['ViewingHoursPerWeek'] + 1)
        X['watch_intensity'] = X['ViewingHoursPerWeek'] / (X['AccountAge'] + 1)
        X['support_to_usage'] = X['SupportTicketsPerMonth'] / (X['ViewingHoursPerWeek'] + 1)
        X['log_total_charges'] = np.log1p(X['TotalCharges'])
        X['watchlist_to_watch_ratio'] = X['WatchlistSize'] / (X['ViewingHoursPerWeek'] + 1)
        X['session_depth'] = X['AverageViewingDuration'] / (X['ViewingHoursPerWeek'] + 1)

        is_premium = (X['SubscriptionType'] == 'Premium')
        low_watch = (X['ViewingHoursPerWeek'] < self.low_watch_threshold)
        X['premium_underuse'] = (is_premium & low_watch).astype(int)

        X['frustration_index'] = X['SupportTicketsPerMonth'] * (6 - X['UserRating'])
        X['loyalty_tier'] = pd.cut(X['AccountAge'], bins=[0,6,12,24,60,120], labels=['very_new','new','growing','loyal','veteran'])
        X['is_new_user'] = (X['AccountAge'] <= 3).astype(int)

        return X


class OutlierHandler(BaseEstimator, TransformerMixin):
    
    def __init__(self, column, z_thresh=2.0, iqr_k=1.5, skew_thresh=0.5):
        self.column = column
        self.z_thresh = z_thresh
        self.iqr_k = iqr_k
        self.skew_thresh = skew_thresh

    def fit(self, X, y=None):
        col = X[self.column]
        skew = col.skew()
        self.method_ = 'zscore' if (-self.skew_thresh < skew < self.skew_thresh) else 'iqr'
        if self.method_ == 'zscore':
            self.mean_ = col.mean()
            self.std_ = col.std()
            self.lower_ = self.mean_ - self.z_thresh * self.std_
            self.upper_ = self.mean_ + self.z_thresh * self.std_
        else:
            q1 = col.quantile(0.25)
            q3 = col.quantile(0.75)
            iqr = q3 - q1
            self.lower_ = q1 - self.iqr_k * iqr
            self.upper_ = q3 + self.iqr_k * iqr

        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].clip(self.lower_, self.upper_)
        return X

    def count_outliers(self, X):
        col = X[self.column]
        mask = (col < self.lower_) | (col > self.upper_)
        return int(mask.sum())






    

    
