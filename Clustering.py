import pandas as pd
import os
from sklearn.cluster import AgglomerativeClustering, KMeans
import skfuzzy as fuzz
import numpy as np
from sklearn_extra.cluster import KMedoids
from prettytable import PrettyTable
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


input_path     = 'Data'
clust_method   = 'fcm'
apply_PCA      = True
clust_num      = 5

class clustering(object):
    def __init__(self, clust_method, clust_num, apply_PCA, save_format = '.xlsx', affinity='euclidean', linkage='ward', random_state=0, fuzzy_param=2.0,error=1e-6,max_Iter=100, variance = 0.95, summary = True, download = True):
        self.clust_method = clust_method
        self.clust_num = clust_num
        self.affinity = affinity
        self.linkage = linkage
        self.download = download
        self.summary = summary
        self.save_format = save_format
        self.random_state = random_state
        self.apply_PCA = apply_PCA
        self.variance = variance
        self.fuzzy_param = fuzzy_param
        self.error = error
        self.max_Iter = max_Iter
        
    def read_file(self, path):
        filename, file_extension = os.path.splitext(path)
        if file_extension == '.xlsx':
            df = pd.read_excel(path, index_col=None)
        if file_extension == '.csv':
            df = pd.read_csv(path)  
        return df
    
    def clean_data(self, dataframe):
        self.nm_df = dataframe.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
        self.nm_df = self.nm_df.reset_index(drop=True)
        filt_df = self.nm_df._get_numeric_data()
        drop_column = filt_df.columns
        drop_column = drop_column[:5]
        filt_df  = filt_df.drop(columns = drop_column)    
        return filt_df


    def standardise_data(self, dataframe):
        stand_data = StandardScaler().fit_transform(dataframe)
        return stand_data

    def agglomerative(self, filtered_df):
        cluster = AgglomerativeClustering(n_clusters=self.clust_num, affinity=self.affinity, linkage=self.linkage)
        res_clusters = cluster.fit_predict(filtered_df)
        return res_clusters
    
    def kmeans(self, filtered_df):
        cluster = KMeans(n_clusters=self.clust_num, random_state=self.random_state)#.fit(X)
        res_cluster = cluster.fit_predict(filtered_df)
        return res_cluster
        
    def fuzzyCmeans(self,filtered_df):

        center, membership, u0, d, jm, p, fpc = fuzz.cluster.cmeans(filtered_df,c=self.clust_num, m=self.fuzzy_param, error=self.error, maxiter=self.max_Iter)
        res_cluster = np.argmax(membership, axis=0)

        return res_cluster
    
    def kmedoids(self, filtered_df):
        cluster = KMedoids(n_clusters=self.clust_num, random_state=self.random_state)#.fit(X)
        res_cluster = cluster.fit_predict(filtered_df)
        return res_cluster
        
    def download_data(self, clustered_df, file_format):
        if file_format == '.xlsx':
            clustered_df.to_excel(self.clust_method+'_output.xlsx')
        if file_format == '.csv':
            clustered_df.to_csv(self.clust_method+'_output.csv')
            
    def gen_summary(self):
        if os.path.exists(self.clust_method+'_output.txt'):
            os.remove(self.clust_method+'_output.txt')
        f = open(self.clust_method+'_output.txt', 'a')
        print('The file is saved in {} format'.format(self.save_format), file=f)
        print('The clustering algorithm used is', self.clust_method, file=f)
        print('The input file have {} observations and {} features'.format(self.input_df.shape[0], self.input_df.shape[1]),file=f)
        print('Removed {} observations and {} features after preprocessing'.format(self.input_df.shape[0]-self.nm_df.shape[0], self.input_df.shape[1]-self.nm_df.shape[1]), file=f)
        if self.apply_PCA:
            print("%d components capture %f amount of variance in the data." %(self.n_components,self.variance), file=f)
        t = PrettyTable()
        t.field_names = ['Index', 'Number of observations']
        for i in range(self.clust_num):
            t.add_row([i, (self.clustered_df.output_class == i).sum()])
        f.write(str(t))
        f.close()
        
    def app_PCA(self, filt_df):
        pca = PCA(self.variance)
        principalComponents = pca.fit_transform(filt_df)
        reduced_df = pd.DataFrame(principalComponents)
        self.n_components = pca.n_components_
        return reduced_df
        
    def run_clustering(self, input_path):
        self.input_df = self.read_file(input_path)
        filt_df = self.clean_data(self.input_df)
        filt_df = self.standardise_data(filt_df)
        if self.apply_PCA:
            filt_df = self.app_PCA(filt_df)
        if self.clust_method == 'agglomerative':
            clusters = self.agglomerative(filt_df)
        elif self.clust_method == 'kmeans':
            clusters = self.kmeans(filt_df)
        elif self.clust_method == 'fcm':
            clusters = self.fuzzyCmeans(filt_df.T) 
        elif self.clust_method == 'kmedoids':
            clusters = self.kmedoids(filt_df)
        clustered_df = pd.DataFrame(clusters, columns=['output_class'])
        self.clustered_df = pd.concat([self.nm_df[self.nm_df.columns[0:2]], clustered_df], axis = 1, sort=True)
        if self.download:
            self.download_data(self.clustered_df, self.save_format)
        if self.summary:
            self.gen_summary()
        return clusters

cluster = clustering(clust_method = clust_method, clust_num = clust_num, apply_PCA = apply_PCA)
output_clusters = cluster.run_clustering(input_path)
    