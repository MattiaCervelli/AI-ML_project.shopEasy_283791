# AI-ML_project

# AI-ML_project
### Project 2 - ShopEasy Customer Segmentation
# Chiara Baldoni, Mattia Cervelli, Paola Pinna 

### Libraries used:
This project utilizes several libraries, each serving a specific purpose in the data analysis pipeline:

- **Pandas & NumPy**: Essential for data manipulation and numerical operations.
- **Matplotlib & Seaborn**: Provide a wide range of plotting functions to visualize data and insights.
- **Scikit-learn's Preprocessing**: LabelEncoder and StandardScaler help prepare the dataset for machine learning models.
- **PCA (Principal Component Analysis)**: Reduces dimensionality of the dataset, preserving essential characteristics.
- **KMeans & DBSCAN**: Clustering algorithms used to segment customers into distinct groups.
- **Silhouette Score**: Assesses the quality of clusters formed by the clustering algorithms.
- **Hierarchical Clustering (scipy.cluster.hierarchy)**: An alternative clustering method that builds a hierarchy of clusters.
- **AgglomerativeClustering**: A type of hierarchical clustering that merges data samples into clusters based on their similarity.

These tools are integral to uncovering customer segments and driving personalized marketing strategies.


## INTRODUCTION 
Our project aims to analyze and understand the customer data of ShopEasy, a leading e-commerce site. By applying various machine learning techniques and statistical analysis, the project seeks to segment the customer base, revealing distinct patterns and behaviors in purchasing and interaction with the platform. We began by refining the dataset, removing irrelevant columns, and addressing missing values. Our project involved detailed univariate and bivariate analyses to understand variable distributions and relationships. Feature engineering was crucial, involving encoding categorical data and scaling numerical features. We conducted correlation analysis uncovering both expected and surprising correlations. For detailed customer grouping we implemented K-Means Clustering and Hierarchal Clustering methods, while PCA aided in reducing dimensionality. The project outcomes were significant, achieving meaningful customer segmentation and providing critical insights. This strategic insight is crucial for maintaining competitive advantage and enhancing customer satisfaction in the dynamic e-commerce market. 

## METHODS
# Data preprocessing and Exploration 
1. Initial Dataset Exploration and Cleaning: 
We started by examining the dataset to understand its structure. Non-essential columns like 'personId' where removed to focus on more relevant features, streamlining the data for analysis.
2. Handling missing values: 
Special attention was given to the 'leastAmountPaid' column where we found 313 missing values. This column was particularly focused upon due to its critical role in understanding customer spending. Where 'itemCount' was 0, indicating no purchase, missing values in leastAmountPaid were filled with zeros, this decision was based on the rationale that no purchase should equate to zero payment. In cases where 'itemCount' was 1, suggesting a single item purchase, we filled missing 'leastAmountPaid' values with corresponding 'singleitemCosts', this approach was chosen under the assumption that the cost of a single item would be a reliable proxy for the total amount paid in such cases. For all other scenarios we used median values to fill missing data. The median was preferred over the mean due to its robustness against outliers, ensuring a more representative value for typical customer spending. 
We also dropped the row with missing maxSpendLimit data as it was only one row.


# Feature Engineering
# Encoding Categorical Features
Feature engineering played a critical role in preparing the dataset for effective analysis and modeling.
In preparation for clustering, we need to convert categorical features into a numerical format. This step is crucial as clustering algorithms require numerical inputs to calculate distances or similarities between data points.
We identified categorical variables like 'location' and 'AccountType' that required trasformation to be effectively used in machine learning algorithms. 

1. One hot encoding: 
- The location feature is a nominal categorical variable with no inherent order (e.g., city names). 
- We use One-Hot Encoding for this feature to prevent the introduction of artificial ordinality. In One-Hot Encoding, each unique value in the location column is transformed into a separate binary (0 or 1) column, ensuring that no unintended order is implied.

2. *Label Encoding for 'accountType':*
   - Contrary to location, accountType is an ordinal categorical feature with a meaningful order (Regular, Premium, Student). 
   - We use Label Encoding for accountType to preserve this inherent order. Label Encoding assigns a unique integer to each category according to its order. This approach is suitable for ordinal data where the relative ordering of categories carries meaningful information.

3. *Dropping the Original 'accountType' column:*
   - After encoding accountType, the original column is redundant and can be dropped from our dataset.

Through these encoding steps, our dataset is now transformed into a completely numerical format, making it ready for clustering and other machine learning algorithms that require numerical inputs.


# Exploratory Data Analysis
Univariate Analysis:
This analysis was conducted to gain a fundamental understanding of each variable independently. This step was crucial for identifying patterns, anomalies, and typical ranges within individual features. For numeric variables, the KDE plots reveal the distribution shape, which can highlight skewness, central tendency, and dispersion. For categorical variables, the bar plots illustrate the frequency of different categories, offering insights into the most and least common categories. This foundational analysis is crucial for data preprocessing, feature selection, and informing subsequent multivariate analyses.

Numeric Features Analysis
For the numeric columns in the ShopEasy dataset, the univariate analysis is conducted using Kernel Density Estimate (KDE) plots. We first selected the numeric columns (int or float) in the DataFrame, then we created a grid of plots where each numeric variable will have its own subplot in the grid. The number of rows and columns in this grid is determined based on the total number of numeric columns. The figure size is set to ensure that each subplot is large enough to be easily visible. For each numeric variable, a KDE plot is created These plots display the distribution of data across a continuous interval or time span. They are especially useful for identifying the shape of the distribution (normal, skewed, bimodal).
Categorical Features Analysis
For categorical features, bar plots are used to understand the distribution of each category within the variables. We first identified the categorical columns ('location' and 'accountType') that need analysis. Similar to the numeric features, we created a grid of subplots for categorical features. Then we created a bar plot for each categorical variable.
These plots show the count of observations in each category of the variable, which helps in understanding the frequency distribution of different categories.
The categories are ordered by their frequency to make the plots more informative.


Bivariate Analysis: 
This type of analysis in our ShopEasy dataset aims to identify correlations, patterns, or associations between pairs of features.
Bivariate analysis in our ShopEasy dataset is a crucial step to uncover relationships between variables. It helps in understanding how variables interact with each other, which can be vital for making informed decisions, especially in business contexts like customer segmentation or targeted marketing. The combination of correlation heatmaps and scatter plots provides a comprehensive view of these relationships, both for expected and unexpected correlations.
-	Correlation Heatmap:
The correlation heatmap is used to visualize the strength and direction of relationships between pairs of numerical variables. The correlation matrix is calculated using shopeasy_df_encoded.corr(), which computes Pearson correlation coefficients for all pairs of numerical variables. 
The heatmap visually represents these correlations with varying colors. Colors typically vary from dark to light, indicating the strength of the correlation from strong to weak. Positive correlations (closer to +1) suggest that as one variable increases, the other variable also tends to increase. Negative correlations (closer to -1) indicate that as one variable increases, the other tends to decrease. Values close to 0 imply weak or no linear relationship between the variables.

-	Scatter Plots for High Correlation Pairs
To visually investigate the relationships between pairs of variables that have high correlations. Pairs of variables with significant correlations are selected based on the heatmap results. sns.scatterplot is used to create scatter plots for these pairs. Each point on a scatter plot represents an observation in the dataset, plotted against two of the variables.
The pattern of the points in the plot can indicate the nature of the relationship. A linear pattern suggests a strong linear relationship, while a more diffuse pattern indicates a weaker relationship.
Outliers or anomalies can also be identified, which are points that fall far away from the general pattern.

### Rationale for Excluding Certain Features
After the analysis of the pairs of variables with high correlation we chose to exclude certain features based on specific observations.

#### *SingleItemCosts & MultipleItemCosts as subsets of ItemCosts*
- *Redundancy Issue*: singleItemCosts and multipleItemCosts are subsets of the total itemCosts. Including these specific subsets along with the total costs can lead to redundancy in the data. 
- *Clustering Implication*: For clustering, the goal is often to capture distinct patterns or groups in the data. Since itemCosts already encapsulates the total expenditure, including its subsets (single and multiple item costs) might not contribute additional distinct information for cluster formation.
- *Data Simplification*: Removing these subsets simplifies the dataset while retaining the core information encapsulated by itemCosts, making the clustering more efficient and focused.

#### *Choosing EmergencyCount over EmergencyUseFrequency*
- *Quantitative vs. Qualitative Data*: emergencyCount provides a quantitative measure of how many times emergency funds have been used, offering concrete data. emergencyUseFrequency, while useful, offers a more qualitative insight and might be less precise for clustering.

#### *Choosing ItemBuyFrequency over MultipleItemBuyFrequency*
- *Broader Scope*: itemBuyFrequency captures the overall purchasing frequency, encompassing all types of purchases. This offers a more holistic view of a customer's shopping habits compared to multipleItemBuyFrequency, which is more specific.
- *Representative Data*: itemBuyFrequency is more representative of the user's general activity on the platform. Clustering based on overall activity can yield more universally applicable insights than focusing solely on installment-based purchases.

## Interactions between categorical features
We then focus on pairs of features that are likely to provide the most valuable insights into customer behavior and ShopEasy's business strategy. By examining the interplay between these pairs, we can derive actionable insights that may enhance personalized marketing efforts, inventory management, credit policies, and customer loyalty programs.


1. *AccountTotal and FrequencyIndex:*
   - This pair is instrumental in revealing the dynamics between a user's total expenditure on the platform and their shopping frequency. We hypothesize that a positive correlation might exist — users who shop more frequently could be contributing a significant portion of ShopEasy's revenue.
   - Insight from this analysis could inform the design of loyalty programs, ensuring they are optimally structured to reward and incentivize the most valuable and consistent shoppers.

2. *ItemCosts and ItemBuyFrequency:*
   - By comparing the aggregate cost of items purchased with the frequency of transactions, we can identify patterns such as whether users making frequent smaller purchases contribute more to the revenue than those making infrequent larger purchases.
   - This analysis is crucial for inventory stocking and management, as well as for crafting personalized product recommendations that resonate with users' purchasing rhythms.

3. *EmergencyFunds and EmergencyUseFrequency:*
   - Understanding how users utilize their emergency funds in correlation with the frequency of such usage offers insights into consumer behavior under different purchasing conditions.
   - This data can reveal the impact of offering a financial cushion on customer loyalty and purchasing frequency, potentially validating the emergency fund as a feature that not only provides peace of mind to the users but also benefits ShopEasy's sales metrics.



-	Scatter Plots for Surprising Correlation Pairs
To explore and understand unexpected or non-intuitive correlations between variables.
Similar to the high correlation pairs, scatter plots are created for pairs of variables that show surprising or interesting correlations.
These correlations might not be strong but could reveal unique or non-obvious patterns in the data.
These plots can unveil insights that might not be evident from traditional correlation analysis, such as nonlinear relationships or clusters within the data.

## Selection of Features for Pairplot Visualization

In order to deepen our understanding of ShopEasy's customer behaviors and tailor segmentation strategies, we select a set of features for pairplot visualization. The pairplot will enable us to observe the distribution of individual features and the relationships between pairs of features. The following eight features are chosen based on their strong correlations and strategic relevance to ShopEasy's goals:

1. **itemBuyFrequency**: Integral for assessing how often users engage in purchasing, and its correlation with other features provides insights into buying patterns.

2. **itemCount**: To visualize the volume of items purchased, which is closely linked to purchase frequency and can indicate customer loyalty and market basket size.

3. **singleItemCosts**: Offers a view into the expenditure on individual items, showing the influence of single purchases on total spending.

4. **multipleItemCosts**: Since installment-based purchases are a key component of customer behavior, understanding this relationship is vital for financial service offerings.

5. **emergencyFunds**: Observing how users allocate funds for emergencies and how this correlates with their actual use can inform financial product development.

6. **emergencyCount**: Important for visualizing the usage pattern of emergency funds, which can reflect financial management behaviors among users.
7. **leastAmountPaid**: By examining this feature alongside others, we can infer aspects of user spending habits and the scale of transactions.

8. **paymentCompletionRate**: Given its unexpected inverse relationship with overall spending, it's valuable to examine how payment behaviors vary across the customer base.

These features are pivotal in painting a comprehensive picture of customer interactions with ShopEasy's platform. The pairplot is expected to uncover nuanced relationships and user segments, facilitating the creation of more targeted and effective marketing campaigns, and enhancing customer experience personalization.

### Rationale for Selected Features in Pairplot

The selected features for the pairplot aim to provide a comprehensive understanding of customer behavior on the ShopEasy platform. Here's the reasoning behind each feature's inclusion:

- **accountTotal**: Reflects total spending, crucial for understanding overall user expenditure.
- **frequencyIndex**: Indicates shopping frequency, offering insights into how often users engage with the platform.
- **itemCosts**: Captures the total cost of purchased items, essential for analyzing spending behavior.
- **emergencyFunds**: Reveals the user's tendency to save for emergencies, indicating financial planning habits.
- **itemBuyFrequency**: Shows the overall purchase frequency, key for understanding customer activity levels.
- **singleItemBuyFrequency**: Provides insight into the frequency of single-item purchases, highlighting a specific shopping pattern.
- **emergencyCount**: The number of times emergency funds are used, reflecting user behavior in urgent situations.
- **itemCount**: Total number of items purchased, important for analyzing purchasing volume.
- **maxSpendLimit**: Indicates the spending capacity set by ShopEasy, shedding light on user trust and creditworthiness.
- **monthlyPaid**: Total monthly payments, useful for understanding consistent spending behavior.
- **paymentCompletionRate**: Shows the rate at which users complete their payments, an indicator of financial reliability.
- **accountLifespan**: The duration of the user's relationship with ShopEasy, offering insights into loyalty and long-term engagement.
These features collectively provide diverse yet relevant insights into customer spending patterns, purchase frequency, financial habits, and overall engagement, making them ideal for segmentation analysis through a pairplot.

#  Scaling Data
Scaling ensures that all features contribute equally to the result and prevents features with larger ranges from dominating the analysis, leading to more accurate and interpretable results. Many machine learning algorithms, particularly those that rely on distance calculations like K-Means, perform better or converge faster when features are on a similar scale.
All features are selected for scaling. In your code, this is done by creating a list columns_to_scale which includes all columns.
This comprehensive approach ensures that all numerical features are normalized, making the dataset uniform for effective clustering.
StandardScaler is employed for scaling. This method standardizes features by removing the mean and scaling to unit variance.
The scaler is fitted to the data. This command computes the mean and standard deviation to be used for later scaling and then performs the standardization.
By fitting and transforming, you scale the data and transform it into a format suitable for the clustering algorithms.
A new DataFrame is then created to hold the scaled data.

## CLUSTERING 

# Dimensionality reduction with PCA
The PCA (Principal Component Analysis) step in the provided code is an essential part of dimensionality reduction and feature extraction.
The PCA algorithm is initialized and then fitted to the scaled dataset (shopeasy_df_scaled). This step involves calculating the principal components that represent the maximum variance in the dataset.
The explained_variance_ratio_ attribute of the PCA model provides the variance explained by each of the principal components.
This information is crucial to understand how much information (variance) can be attributed to each principal component.
The cumulative sum of the explained variances (cum_sum) is calculated. This gives insight into how many components are needed to explain a certain percentage of the variance in the dataset.
This step helps in deciding the number of principal components to retain for effective data representation with reduced dimensions.
PCA is again initialized, this time specifying n_components=2, to reduce the dataset to two dimensions.
The transformed dataset (shopeasy_pca) is obtained by applying the fit_transform method. This results in a 2-dimensional representation of the dataset, preserving as much variance as possible.
The components of the PCA model represent the contribution of each feature to the principal components.
This information is organized into a DataFrame for easier interpretation. The DataFrame is transposed to align principal components with the features.
The most influential features for each principal component are identified using nlargest method. This helps in understanding which features contribute most to the variation in the dataset in the context of the reduced dimensions.
These key features can provide insights into the underlying structure of the data and are critical for interpreting the results of PCA.
A heatmap is plotted to visualize the relationship between the original features and the principal components. The heatmap shows how each feature contributes to each principal component, helping to further interpret the PCA results.
The PCA step effectively reduces the dimensionality of the data, making it more manageable and interpretable while retaining most of the variance. This is particularly useful for visualization and clustering in high-dimensional datasets like yours.



## K-Means:
We first found the optimal number of clusters. Determining the optimal number of clusters is a crucial step for effective clustering. This decision significantly impacts the quality of the segmentation you perform on your dataset.

1. Elbow Method:
Process: The Elbow Method involves running the KMeans clustering algorithm on the dataset for a range of values of k (number of clusters) and then calculating the sum of squared distances from each point to its assigned center (inertia).
Implementation: In the code, KMeans is run for k values ranging from 1 to 9. For each k, the inertia is calculated and stored in inertia_vector.
Interpretation: A plot of k values against their corresponding inertia is created. The "elbow" of the plot, where the rate of decrease sharply changes, indicates the appropriate number of clusters. This point represents a balance between minimizing within-cluster variance and maximizing between-cluster variance.

2. Silhouette Score:
Process: The Silhouette Score is a measure of how similar an object is to its own cluster compared to other clusters. The value ranges from -1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.
Implementation: The Silhouette Score is computed for k values from 2 to 9 using the silhouette_score function. This is because silhouette is not defined for a single cluster.
Interpretation: The k value that gives the lowest Silhouette Score is typically considered the best choice for the number of clusters. This score helps in assessing the separation distance between the resulting clusters.

Decision on the Number of Clusters
The decision on the number of clusters was based on a combination of the Elbow Method and Silhouette Analysis.
The Elbow Method provided a visual cue for the potential number of clusters, while the Silhouette Score and Silhouette Analysis offered a more nuanced view of how well the data was segmented into clusters.
The five clusters configuration has the lowest silhouette score.
Therefore, clustering will proceed using five clusters.

Implementation of K-means clustering
KMeans clustering is used to segment the customer base into distinct groups. This segmentation helps in understanding various customer behaviors and preferences, which can inform targeted marketing strategies, personalized service offerings, and overall business strategy. The careful selection of k and the preprocessing steps like scaling ensure that the clustering is both meaningful and reliable.
The KMeans algorithm from scikit-learn’s cluster module is used.
It is initialized with parameters like the number of clusters (n_clusters), the method for initialization (init), and the number of initializations (n_init). In your code, init is set to 'k-means++', which is an advanced method for centroid initialization that can speed up convergence.
The algorithm is applied to the scaled data, and it iteratively performs the assignment and update steps to form clusters.
After the algorithm converges, each data point in your dataset is assigned to one of the k clusters.
The result is stored in a variable (e.g., y_kmeans), which contains the cluster index for each data point.



## Hierarchal clustering
Hierarchical Clustering is a method of cluster analysis which seeks to build a hierarchy of clusters. It is distinct from KMeans in its approach: instead of specifying the number of clusters at the start, it generates a dendrogram, a tree-like diagram that records the sequences of merges or splits.
Hierarchical clustering serves as a valuable complement to KMeans, offering an alternative clustering approach that can validate or provide additional insights into customer segmentation. Its implementation and the resulting dendrogram provide a detailed view of how customers can be grouped based on their shopping behaviors and attributes.


1.	Creating the Linkage Matrix:
The code uses scipy.cluster.hierarchy.linkage to perform hierarchical clustering.
The linkage matrix (linkage_matrix) is created using the 'ward' method. The 'ward' method minimizes the variance within each cluster, effectively seeking to form compact, spherical clusters similar to K-Means.
The data input for the linkage matrix is the PCA reduced data (shopeasy_reduced), which helps in managing computational complexity and efficiency.
2.	Dendrogram Visualization:
A dendrogram is generated using scipy.cluster.hierarchy.dendrogram.
This visualization helps in understanding how the hierarchical clustering process groups the data points into clusters step by step.
By visually inspecting the dendrogram, one can interpret the number of clusters that could meaningfully segment the data. The decision is made by identifying the longest vertical distance that doesn’t cross any extended horizontal lines.
From visual inspection of the dendrogram plot it appears that the customer data may be organised around three clusters.

3.	Applying Agglomerative Clustering:
The AgglomerativeClustering function from scikit-learn is then used with the determined number of clusters.
This method starts with each data point as a single cluster and merges the closest pairs of clusters until the specified number of clusters is reached.
4.	Cluster Assignment:
Each data point in the dataset is assigned to one of the clusters formed by the hierarchical clustering process. The result contains the cluster indices for each data point.

## Conclusion

Some clues about the cluster nature can neverthless be inferred observing the relationship between the first two PCA dimensione and customer data features.

The first PCA dimension, which accounts for slightly more than 26% of total customer data info variance, seems correlated to the way and the frequence the customer shops with. 
It has strong correlations (>=0.35) with:
- itemCosts: Total costs of items purchased by the user;
- itemCount: Total number of individual items purchased by the user;
- monthlyPaid: Total amount paid by the user every month;
- singleItemBuyFrequency: How often the user makes single purchases without opting for installments.

The second PCA dimension, which accounts for slightly more than 20% of total customer data info variance, seems correlated to the customer fund management and its recourse to the mergency fund. 
It has strong correlations (>=0.4) with:
- accountTotal: Total amount spent by the user on ShopEasy since their registration;
- emergencyFunds: Amount that the user decided to keep as a backup in their ShopEasy wallet for faster checkout or emergency purchases;
- emergencyCount: Number of times the user has used their emergency funds.

In this view, observing the 2D cluster plot identifies by the hierarchical search algorithm, we can hypotesize three kinds of customers:
- the first group (in red in the plot) is low-spending (low itemCosts) with limited need to resort to an emergncy fund (low emergencyFunds);
- the second group (in blue in the plot) is high-spending (high itemCosts) has substantial funds in the ShopEasy wallet (high emergencyFunds);
- the second group (in green) is also higher-spending (high itemCosts) but does not use ShopEasy wallet (low emergencyFunds). This kind of customer is likely not informed of the feature and, in general, of trhe full range of the ShopEasy services. This kind of customer could be addrerssed with a fidelization campaign.

In this view, observing the 2D cluster plot identifies by the kmean search algorithm, we can hypotesize five kinds of customers:
- the first group (in purple in the plot) represents moderate spenders with balanced item costs and emergency funds. These customers might be using ShopEasy for regular purchases and occasionally rely on emergency funds, suggesting a calculated approach to budgeting and spending.
- the second group (in pink in the plot) consists of customers with low to moderate item costs and a lower tendency to utilize emergency funds. They may be occasional shoppers or new to the platform, not yet fully engaged with the ShopEasy wallet or other financial services offered.
- the third group (in plue in the plot) is characterized by high spenders who also maintain a substantial balance in their emergency funds. This indicates a deep engagement with the ShopEasy platform and a higher level of trust and reliance on its financial features.
- the fourth group (in red in the plot) includes customers with high item costs but minimal use of emergency funds. This segment likely represents confident consumers who are comfortable with their spending but may not be taking full advantage of the financial planning tools available on ShopEasy.
- the fifth group (in green in the plot) shows diverse spending behaviors but uniformly low use of emergency funds. This suggests a segment of customers who are either unaware of or choosing not to use ShopEasy's financial management features, indicating potential for growth through targeted educational and loyalty campaigns.

The 2D plots of the ShopEasy customer data clustering resulting from our analysis show there are no low-density zones between the clusters; on the contrary, there is some overlap between the plot cluster areas.
This indicates that, in general, the clusters indentified do not possess a strong distinctive character.
