#Load libraries

import pandas as pd
import numpy as np
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import count, to_utc_timestamp, unix_timestamp, lit, datediff, col,from_unixtime
import matplotlib.pyplot as plt


spark = SparkSession \
    .builder \
    .appName("Python Spark BI analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
    

input_path1     = 'attribution.csv'
input_path2     = 'conversions.csv'
p_flag      =  False


class analysis(object):
    def __init__(self, p_flag):
        self.p_flag = p_flag
        
       
    def read_file(self, path1, path2):
        attribution = spark.read.format('com.databricks.spark.csv').\
               options(header='true', \
               inferschema='true').\
        load(path1, header=True);        
        
        conversions = spark.read.format('com.databricks.spark.csv').\
               options(header='true', \
               inferschema='true').\
        load(path2, header=True);
        
        return attribution, conversions
        
    
    def clean_data(self, conversions, attribution):
        
        #Handle missing values
        attribution = attribution.dropna(how='any')
        conversions = conversions.dropna(how='any')
        Data = attribution.join(conversions, on=['Conv_ID'], how='inner')
        
        # Change conv_date to datetime format
        timeFmt = "yyyy-MM-dd"
        Data = Data.withColumn('Conv_Date', to_utc_timestamp(unix_timestamp(col('Conv_Date'),timeFmt).cast('timestamp')
                 , 'UTC'))
           
        return Data
    
    
    def describeData(self, filt_df):
        filt_df.printSchema()
        filt_df.describe().show()
        
        #group by channel
        group_data = filt_df.groupBy("Channel")
        group_data.agg({'User_ID':'count'}).show(5)
        
        
        group_data = filt_df.groupBy("Channel")
        group_data.agg({'Revenue':'sum'}).show(5)
        
        return True
    
    def plot(self, filt_df):
        
        #For demo, we can plot 2 graphs by setting  p_flag = True
        
        #Total revenue from every channel
        ax = filt_df.groupby(['Channel'])['Revenue'].sum().plot(kind = 'bar', title = "Total_Revenue via channels", figsize = (15, 10), legend = True, fontsize = 12)
        ax.set_xlabel("Channels", fontsize = 12)
        ax.set_ylabel("Revenue", fontsize = 12)
        plt.show()                            
                  
        #Monthly total transactions
        ax = filt_df.groupby(['month_year'])['Conv_ID'].count().plot(kind = 'line', title = "Monthly total transaction", figsize = (15, 10), legend = True, fontsize = 12)
        ax.set_xlabel("date", fontsize = 12)
        ax.set_ylabel("transactions", fontsize = 12)
        plt.show()
        
        
    def run_analysis(self, file1, file2):
        attribution, conversions = self.read_file(file1, file2)
        attribution.show(5)
        attribution.printSchema()
        conversions.show(5)
        conversions.printSchema()        
        
        filt_df = self.clean_data(attribution, conversions)
        
        #Look into the data
        self.describeData(filt_df)        
        pframe = filt_df.toPandas()
        pframe['month_year'] = pframe['Conv_Date'].dt.to_period('M')
        
        if p_flag:
            self.plot(pframe)
               
           
        return pframe


score = analysis(p_flag = p_flag)
Data = score.run_analysis(input_path1, input_path2)

print(Data.head())

    