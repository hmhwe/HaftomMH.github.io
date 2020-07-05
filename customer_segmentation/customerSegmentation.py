import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA


input_path1     = 'attribution.csv'
input_path2     = 'conversions.csv'
clust_method   = 'kmeans'
apply_PCA      = False
clust_num      = 4

class RFM(object):
    def __init__(self, clust_method, clust_num, apply_PCA, save_format = '.xlsx', affinity='euclidean', linkage='ward', random_state=0, fuzzy_param=2.0,error=1e-6,max_Iter=100, variance = 0.95, summary = True, download = True):
        self.clust_method = clust_method
        self.clust_num = clust_num
        self.affinity = affinity
        self.linkage = linkage
        self.random_state = random_state
        self.apply_PCA = apply_PCA
        self.variance = variance
        self.error = error
        self.max_Iter = max_Iter
        
        
    def read_file(self, path1, path2):
            dframe1 = pd.read_csv(path1)
            dframe2 = pd.read_csv(path2)  
            return dframe1, dframe2
    
    
    def clean_data(self, conversions, attribution):
       
       Data = pd.merge(conversions, attribution, left_on='Conv_ID', right_on='Conv_ID', how='left')
       
       #Handle missing values
       Data = Data.dropna()
       Data.reset_index(drop = True)
       
       Data.Conv_Date = pd.to_datetime(Data.Conv_Date)
       
       #extract month and day
       Data['month_year'] = Data['Conv_Date'].dt.to_period('M')
       Data['day_month_year'] = Data['Conv_Date'].dt.to_period('D')


       return Data


    def recency(self, filtered_df):
        score_df = pd.DataFrame(filtered_df['User_ID'].unique())
        score_df.columns = ['User_ID']
        
        max_revenue = filtered_df.groupby('User_ID').Conv_Date.max().reset_index()
        max_revenue.columns = ['User_ID','MaxRevenueDate']
        max_revenue['Recency'] = (max_revenue['MaxRevenueDate'].max() - max_revenue['MaxRevenueDate']).dt.days
        score_df = pd.merge(score_df, max_revenue[['User_ID','Recency']], on='User_ID')
        
        return score_df
    
    
    def frequency(self, score_df, filtered_df):
        tx_frequency = filtered_df.groupby('User_ID').Conv_Date.count().reset_index()
        tx_frequency.columns = ['User_ID','Frequency']

        #add this data to our main dataframe
        score_df = pd.merge(score_df, tx_frequency, on='User_ID')

        
        return score_df
    
    def monetary(self, score_df, filtered_df):
        tx_revenue = filtered_df.groupby('User_ID').Revenue.sum().reset_index()
        tx_revenue.columns = ['User_ID','Monetary']

        score_df = pd.merge(score_df, tx_revenue, on='User_ID')

        
        return score_df
    
    def overall_score(self, score_df):
       score_df['OverallScore'] = score_df['RecencyCluster'] + score_df['FrequencyCluster'] + score_df['MonetaryCluster']
       score_df['Segment'] = 'Low'
       score_df.loc[score_df['OverallScore'] > 1,'Segment'] = 'Mid' 
       score_df.loc[score_df['OverallScore'] > 4,'Segment'] = 'High'        
       return score_df




    #for ordering cluster numbers
    def order_cluster(self, cluster_name, field_name, df, ascending):
        df_n = df.groupby(cluster_name)[field_name].mean().reset_index()
        df_n = df_n.sort_values(by=field_name, ascending = ascending).reset_index(drop = True)
        df_n['index'] = df_n.index
        ordered_df = pd.merge(df, df_n[[cluster_name,'index']], on = cluster_name)
        ordered_df = ordered_df.drop([cluster_name],axis=1)
        ordered_df = ordered_df.rename(columns={"index":cluster_name})
        return ordered_df


    def agglomerative(self, score_df, col_name):
        aggromerative = AgglomerativeClustering(n_clusters = self.clust_num, affinity=self.affinity, linkage=self.linkage)
        aggromerative.fit(score_df.col_name)
        res_clusters = aggromerative.predict(score_df.col_name)
        return res_clusters
    
    def kmeans(self, score_df, col_name):
        kmeans = KMeans(n_clusters = self.clust_num, random_state = self.random_state)
        kmeans.fit(score_df[[col_name]])
        res_cluster = kmeans.predict(score_df[[col_name]])
        return res_cluster
        
     
    def kmedoids(self, score_df, col_name):
        kmedoids = KMedoids(n_clusters = self.clust_num, random_state = self.random_state)
        kmedoids.fit(score_df[[col_name]])
        res_cluster = kmedoids.predict(score_df[[col_name]])
        return res_cluster
        
         
    def app_PCA(self, filt_df):
        pca = PCA(self.variance)
        principalComponents = pca.fit_transform(filt_df)
        reduced_df = pd.DataFrame(principalComponents)
        self.n_components = pca.n_components_
        return reduced_df
        
    def run_RFM(self, file1, file2):
        attribution, contributions = self.read_file(file1, file2)
        filt_df = self.clean_data(attribution, contributions)
        score_df = self.recency(filt_df)
        score_df = self.frequency(score_df, filt_df)
        score_df = self.monetary(score_df, filt_df)
        
        clusters_rec = self.kmeans(score_df, 'Recency')
        score_df['RecencyCluster'] = clusters_rec
        score_df = self.order_cluster('RecencyCluster', 'Recency',score_df, False)

        
        clusters_fre = self.kmeans(score_df, 'Frequency')
        score_df['FrequencyCluster'] = clusters_fre
        score_df = self.order_cluster('FrequencyCluster', 'Frequency', score_df, True)


        clusters_mon = self.kmeans(score_df, 'Monetary')
        score_df['MonetaryCluster'] = clusters_mon
        score_df = self.order_cluster('MonetaryCluster', 'Monetary', score_df, True)
        
        overall_score = self.overall_score(score_df)

        #return rfm score
        return overall_score
    

score = RFM(clust_method = clust_method, clust_num = clust_num, apply_PCA = apply_PCA)
overall_score = score.run_RFM(input_path1, input_path2)

#print(overall_score.groupby('OverallScore')['Recency','Frequency','Monetary'].mean())

print(overall_score.head())

    