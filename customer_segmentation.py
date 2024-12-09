import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime

#Loading dataset
file_path = "/mnt/c/Users/cromw/Downloads/dataset.csv"

try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print("The specified file was not found. Please check the file path.")
    exit()

# Data exploration and cleaning
print("Initial Data Overview:")
print(data.head())
print(data.describe())
print(data.info())

# Check for Missing Values
print("\nMissing Values Count:")
missing_values = data.isnull().sum()
print(missing_values)

# Drop all rows missing profile_id
data = data.dropna(subset=['profile_id']) 

data['profile_id'] = data['profile_id'].astype(int)
data['amount'] = data['amount'].astype(float)

data = data[data['amount'] > 0]

data['date_created'] = pd.to_datetime(data['date_created'], format='mixed', errors='coerce')
data = data.dropna(subset=['date_created'])

# Compute RFM
snapshot_date = data['date_created'].max() + pd.Timedelta(days=1)

rfm = data.groupby('profile_id').agg({
    'date_created': lambda x: (snapshot_date - x.max()).days,  # Recency
    'amount': 'sum'                                           # Monetary Value
}).rename(columns={
    'date_created': 'Recency',
    'amount': 'MonetaryValue'
})

# Add Frequency as a separate column
rfm['Frequency'] = data.groupby('profile_id').size()

print(rfm.head())

# Scale RFM Values
recency_bins = [rfm['Recency'].min()-1, 20, 50, 150, 250, rfm['Recency'].max()]
frequency_bins = [rfm['Frequency'].min() - 1, 2, 3, 10, 100, rfm['Frequency'].max()]
monetary_bins = [rfm['MonetaryValue'].min() - 1, 300, 600, 2000, 5000, rfm['MonetaryValue'].max()]

rfm['R_Score'] = pd.cut(rfm['Recency'], bins=recency_bins, labels=range(1, 6), include_lowest=True).astype(int)
rfm['F_Score'] = pd.cut(rfm['Frequency'], bins=frequency_bins, labels=range(1, 6), include_lowest=True).astype(int)
rfm['M_Score'] = pd.cut(rfm['MonetaryValue'], bins=monetary_bins, labels=range(1, 6), include_lowest=True).astype(int)

rfm['R_Score'] = 5 - rfm['R_Score']  # Reverse Recency score for logical mapping

print(rfm[['R_Score', 'F_Score', 'M_Score']].head())

# Perform K-Means Clustering
X = rfm[['R_Score', 'F_Score', 'M_Score']]

inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)


plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Curve')
plt.grid(True)
plt.savefig('elbow_curve.png')  # Save Elbow Curve
plt.show()

# Apply K-Means with Optimal Clusters
optimal_clusters = 4  # Adjust based on the elbow curve
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(X)

# Analyze and Visualize Clusters
cluster_summary = rfm.groupby('Cluster').agg({
    'R_Score': 'mean',
    'F_Score': 'mean',
    'M_Score': 'mean',
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': 'mean'
}).reset_index()

print("\nCluster Summary:")
print(cluster_summary)

# Add Cluster Labels 
cluster_labels = {
    0: "Champions",
    1: "Loyal Customers",
    2: "At-Risk Customers",
    3: "Recent Customers"
}
rfm['Segment'] = rfm['Cluster'].map(cluster_labels)

# Save Cluster Summary as CSV
rfm.to_csv('customer_segments.csv', index=False)

# Visualize Customer Segments as Pie Chart
plt.figure(figsize=(8, 8))
rfm['Cluster'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Customer Segments')
plt.savefig('customer_segments_pie_chart.png')  # Save Pie Chart
plt.show()