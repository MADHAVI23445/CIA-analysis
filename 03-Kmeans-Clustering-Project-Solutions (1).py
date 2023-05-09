#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'><img src='../Pierian_Data_Logo.png'/></a>
# ___
# <center><em>Copyright by Pierian Data Inc.</em></center>
# <center><em>For more information, visit us at <a href='http://www.pieriandata.com'>www.pieriandata.com</a></em></center>

# # CIA Country Analysis and Clustering
# 
# 
# Source: All these data sets are made up of data from the US government. 
# https://www.cia.gov/library/publications/the-world-factbook/docs/faqs.html
# 
# ## Goal: 
# 
# ### Gain insights into similarity between countries and regions of the world by experimenting with different cluster amounts. What do these clusters represent? *Note: There is no 100% right answer, make sure to watch the video for thoughts.*
# 
# ----
# 
# ## Imports and Data
# 
# **TASK: Run the following cells to import libraries and read in data.**

# In[701]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[702]:


df = pd.read_csv('../DATA/CIA_Country_Facts.csv')


# ## Exploratory Data Analysis
# 
# **TASK: Explore the rows and columns of the data as well as the data types of the columns.**

# In[703]:


# CODE HERE


# In[704]:


df.head()


# In[705]:


df.info()


# In[706]:


df.describe().transpose()


# # Exploratory Data Analysis
# 
# Let's create some visualizations. Please feel free to expand on these with your own analysis and charts!

# **TASK: Create a histogram of the Population column.**

# In[707]:


# CODE HERE


# In[708]:


sns.histplot(data=df,x='Population')


# **TASK: You should notice the histogram is skewed due to a few large countries, reset the X axis to only show countries with less than 0.5 billion people**

# In[709]:


#CODE HERE


# In[710]:


sns.histplot(data=df[df['Population']<500000000],x='Population')


# **TASK: Now let's explore GDP and Regions. Create a bar chart showing the mean GDP per Capita per region (recall the black bar represents std).**

# In[711]:


# CODE HERE


# In[712]:


plt.figure(figsize=(10,6),dpi=200)
sns.barplot(data=df,y='GDP ($ per capita)',x='Region',estimator=np.mean)
plt.xticks(rotation=90);


# **TASK: Create a scatterplot showing the relationship between Phones per 1000 people and the GDP per Capita. Color these points by Region.**

# In[713]:


#CODE HERE


# In[714]:


plt.figure(figsize=(10,6),dpi=200)
sns.scatterplot(data=df,x='GDP ($ per capita)',y='Phones (per 1000)',hue='Region')
plt.legend(loc=(1.05,0.5))


# **TASK: Create a scatterplot showing the relationship between GDP per Capita and Literacy (color the points by Region). What conclusions do you draw from this plot?**

# In[715]:


#CODE HERE


# In[716]:


plt.figure(figsize=(10,6),dpi=200)
sns.scatterplot(data=df,x='GDP ($ per capita)',y='Literacy (%)',hue='Region')


# **TASK: Create a Heatmap of the Correlation between columns in the DataFrame.**

# In[717]:


#CODE HERE


# In[718]:


sns.heatmap(df.corr())


# **TASK: Seaborn can auto perform hierarchal clustering through the clustermap() function. Create a clustermap of the correlations between each column with this function.**

# In[719]:


# CODE HERE


# In[720]:


sns.clustermap(df.corr())


# -----

# ## Data Preparation and Model Discovery
# 
# Let's now prepare our data for Kmeans Clustering!
# 
# ### Missing Data
# 
# **TASK: Report the number of missing elements per column.**

# In[721]:


#CODE HERE


# In[722]:


df.isnull().sum()


# **TASK: What countries have NaN for Agriculture? What is the main aspect of these countries?**

# In[723]:


df[df['Agriculture'].isnull()]['Country']


# **TASK: You should have noticed most of these countries are tiny islands, with the exception of Greenland and Western Sahara. Go ahead and fill any of these countries missing NaN values with 0, since they are so small or essentially non-existant. There should be 15 countries in total you do this for. For a hint on how to do this, recall you can do the following:**
# 
#     df[df['feature'].isnull()]
#     

# In[724]:


# REMOVAL OF TINY ISLANDS
df[df['Agriculture'].isnull()] = df[df['Agriculture'].isnull()].fillna(0)


# **TASK: Now check to see what is still missing by counting number of missing elements again per feature:**

# In[725]:


#CODE HERE


# In[726]:


df.isnull().sum()


# **TASK: Notice climate is missing for a few countries, but not the Region! Let's use this to our advantage. Fill in the missing Climate values based on the mean climate value for its region.**
# 
# Hints on how to do this: https://stackoverflow.com/questions/19966018/pandas-filling-missing-values-by-mean-in-each-group
# 

# In[727]:


# CODE HERE


# In[728]:


# https://stackoverflow.com/questions/19966018/pandas-filling-missing-values-by-mean-in-each-group
df['Climate'] = df['Climate'].fillna(df.groupby('Region')['Climate'].transform('mean'))


# **TASK: Check again on many elements are missing:**

# In[729]:


#CODE HERE


# In[730]:


df.isnull().sum()


# **TASK: It looks like Literacy percentage is missing. Use the same tactic as we did with Climate missing values and fill in any missing Literacy % values with the mean Literacy % of the Region.**

# In[731]:


#CODE HERE


# In[732]:


df[df['Literacy (%)'].isnull()]


# In[733]:


# https://stackoverflow.com/questions/19966018/pandas-filling-missing-values-by-mean-in-each-group
df['Literacy (%)'] = df['Literacy (%)'].fillna(df.groupby('Region')['Literacy (%)'].transform('mean'))


# **TASK: Check again on the remaining missing values:**

# In[734]:


df.isnull().sum()


# **TASK: Optional: We are now missing values for only a few countries. Go ahead and drop these countries OR feel free to fill in these last few remaining values with any preferred methodology. For simplicity, we will drop these.**

# In[735]:


# CODE HERE


# In[736]:


df = df.dropna()


# ## Data Feature Preparation

# **TASK: It is now time to prepare the data for clustering. The Country column is still a unique identifier string, so it won't be useful for clustering, since its unique for each point. Go ahead and drop this Country column.**

# In[737]:


#CODE HERE


# In[738]:


X = df.drop("Country",axis=1)


# **TASK: Now let's create the X array of features, the Region column is still categorical strings, use Pandas to create dummy variables from this column to create a finalzed X matrix of continuous features along with the dummy variables for the Regions.**

# In[739]:


#COde here


# In[740]:


X = pd.get_dummies(X)


# In[741]:


X.head()


# ### Scaling

# **TASK: Due to some measurements being in terms of percentages and other metrics being total counts (population), we should scale this data first. Use Sklearn to scale the X feature matrics.**

# In[742]:


#CODE HERE


# In[743]:


from sklearn.preprocessing import StandardScaler


# In[744]:


scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)


# In[745]:


scaled_X


# ### Creating and Fitting Kmeans Model
# 
# 
# 

# **TASK: Use a for loop to create and fit multiple KMeans models, testing from K=2-30 clusters. Keep track of the Sum of Squared Distances for each K value, then plot this out to create an "elbow" plot of K versus SSD. Optional: You may also want to create a bar plot showing the SSD difference from the previous cluster.**

# In[746]:


#CODE HERE


# In[747]:


from sklearn.cluster import KMeans


# In[748]:


ssd = []

for k in range(2,30):
    
    model = KMeans(n_clusters=k)
    
    
    model.fit(scaled_X)
    
    #Sum of squared distances of samples to their closest cluster center.
    ssd.append(model.inertia_)


# In[749]:


plt.plot(range(2,30),ssd,'o--')
plt.xlabel("K Value")
plt.ylabel(" Sum of Squared Distances")


# In[750]:


pd.Series(ssd).diff().plot(kind='bar')


# -----

# # Model Interpretation
# 
# 
# **TASK: What K value do you think is a good choice? Are there multiple reasonable choices? What features are helping define these cluster choices. As this is unsupervised learning, there is no 100% correct answer here. Please feel free to jump to the solutions for a full discussion on this!.**

# In[751]:


# Nothing to really code here, but choose a K value and see what features 
# are most correlated to belonging to a particular cluster!

# Remember, there is no 100% correct answer here!


# -----
# 
# 
# #### Example Interpretation: Choosing K=3
# 
# **One could say that there is a significant drop off in SSD difference at K=3 (although we can see it continues to drop off past this). What would an analysis look like for K=3? Let's explore which features are important in the decision of 3 clusters!**

# In[753]:


model = KMeans(n_clusters=3)
model.fit(scaled_X)


# In[754]:


model.labels_


# In[756]:


X['K=3 Clusters'] = model.labels_


# In[757]:


X.corr()['K=3 Clusters'].sort_values()


# ------------
# -------------
# 
# # BONUS CHALLGENGE:
# ## Geographical Model Interpretation

# The best way to interpret this model is through visualizing the clusters of countries on a map! **NOTE: THIS IS A BONUS SECTION.  YOU MAY WANT TO JUMP TO THE SOLUTIONS LECTURE FOR A FULL GUIDE, SINCE WE WILL COVER TOPICS NOT PREVIOUSLY DISCUSSED AND BE HAVING A NUANCED DISCUSSION ON PERFORMANCE!**
# 
# ----
# ----
# 
# **IF YOU GET STUCK, PLEASE CHECK OUT THE SOLUTIONS LECTURE. AS THIS IS OPTIONAL AND COVERS MANY TOPICS NOT SHOWN IN ANY PREVIOUS LECTURE**
# 
# ----
# ----

# **TASK: Create cluster labels for a chosen K value. Based on the solutions, we believe either K=3 or K=15 are reasonable choices. But feel free to choose differently and explore.**

# In[765]:


model = KMeans(n_clusters=15)
    
model.fit(scaled_X)
    


# In[766]:


model = KMeans(n_clusters=3)
    
model.fit(scaled_X)


# **TASK: Let's put you in the real world! Your boss just asked you to plot out these clusters on a country level choropleth map, can you figure out how to do this? We won't step by step guide you at all on this, just show you an example result. You'll need to do the following:**
# 
# 1. Figure out how to install plotly library: https://plotly.com/python/getting-started/
# 
# 2. Figure out how to create a geographical choropleth map using plotly: https://plotly.com/python/choropleth-maps/#using-builtin-country-and-state-geometries
# 
# 3. You will need ISO Codes for this. Either use the wikipedia page, or use our provided file for this: **"../DATA/country_iso_codes.csv"**
# 
# 4. Combine the cluster labels, ISO Codes, and Country Names to create a world map plot with plotly given what you learned in Step 1 and Step 2.
# 
# 
# **Note: This is meant to be a more realistic project, where you have a clear objective of what you need to create and accomplish and the necessary online documentation. It's up to you to piece everything together to figure it out! If you get stuck, no worries! Check out the solution lecture.**
# 
# 

# In[767]:


iso_codes = pd.read_csv("../DATA/country_iso_codes.csv")


# In[768]:


iso_codes


# In[769]:


iso_mapping = iso_codes.set_index('Country')['ISO Code'].to_dict()


# In[770]:


iso_mapping


# In[771]:


df['ISO Code'] = df['Country'].map(iso_mapping)


# In[772]:


df['Cluster'] = model.labels_


# In[773]:


import plotly.express as px

fig = px.choropleth(df, locations="ISO Code",
                    color="Cluster", # lifeExp is a column of gapminder
                    hover_name="Country", # column to add to hover information
                    color_continuous_scale='Turbo'
                    )
fig.show()


# ---
