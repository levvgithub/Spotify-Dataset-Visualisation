import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Loading the dataset
file_path = "data.csv"
spotify_data = pd.read_csv(file_path, encoding='latin1')

# Converting duration from milliseconds to minutes
spotify_data['duration_min'] = spotify_data['duration_ms'] / 60000  # 1 minute = 60000 ms
# Create a new column for the decade
spotify_data['decade'] = (spotify_data['year'] // 10) * 10

# Keeping only relevant columns
columns_to_keep = ['valence', 'year', 'artists', 'danceability',
                   'duration_min', 'energy', 'loudness', 'name', 'popularity', 'tempo','decade']
spotify_data = spotify_data[columns_to_keep]

# Dropping rows with missing or invalid data
spotify_data = spotify_data.dropna()
spotify_data = spotify_data.drop_duplicates()

# Removing square brackets and quotes from artist names
spotify_data['artists'] = spotify_data['artists'].str.strip("[]").str.replace("'", "")

# Normalizing numeric features:
scaler = MinMaxScaler()
spotify_data[['normalized_danceability', 'normalized_energy', 'normalized_loudness', 'normalized_tempo', 'normalized_valence']] = scaler.fit_transform(spotify_data[['danceability', 'energy', 'loudness', 'tempo', 'valence']])

# Checking datatype and convert if needed
spotify_data['year'] = spotify_data['year'].astype(int)

# Removing incorrect outliers
spotify_data = spotify_data[(spotify_data['popularity'] >= 0) & (spotify_data['popularity'] <= 100)]

# Saving the cleaned DataFrame to a CSV file
# index=False avoids saving the row numbers
spotify_data.to_csv('data_cleaned.csv', index=False)

# Set a general style for plots
sns.set(style="whitegrid")

# 1. Trend of Popularity Over Time
# Plotting popularity trend over the years
plt.figure(figsize=(10, 6))
sns.lineplot(data=spotify_data, x="year", y="popularity", marker="o")
plt.title("Trend of Popularity Over Time")
plt.xlabel("Year")
plt.ylabel("Popularity")
plt.show()

# 2. Correlation Heatmap
# Generating and displaying a heatmap of feature correlations
spotify_data_no_decade = spotify_data.drop(columns=["decade", 'normalized_danceability', 'normalized_energy', 'normalized_loudness', 'normalized_tempo', 'normalized_valence'])
plt.figure(figsize=(12, 8))
correlation_matrix = spotify_data_no_decade.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.show()

# 3. Median Danceability and Tempo by Decade
# Calculating the median danceability and tempo by decade
median_stats = spotify_data.groupby('decade').agg(
    median_danceability=('normalized_danceability', 'median'),
    median_tempo=('normalized_tempo', 'median')
).reset_index()

# Plotting with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# First axis for danceability
ax1.bar(median_stats['decade'], median_stats['median_danceability'],
        width=5,
        color='skyblue', alpha=0.7, label='Normalized Danceability')
ax1.set_xlabel('Decade')
ax1.set_ylabel('Normalized Danceability', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')

# Ensuring all x-axis labels are shown
plt.xticks(median_stats['decade'], rotation=0)

# Second axis for tempo
ax2 = ax1.twinx()
ax2.plot(median_stats['decade'], median_stats['median_tempo'],
         color='salmon', marker='o', label='Normalized Tempo')
ax2.set_ylabel('Normalized Tempo', color='salmon')
ax2.tick_params(axis='y', labelcolor='salmon')

# Matching the y-limits of both axes
ax2.set_ylim(ax1.get_ylim())

# Plotting the chart
fig.suptitle("Median Danceability and Tempo by Decade (Normalized)", fontsize=14)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()


# 4. Analyze Top-10 Tracks by Decade

# Filter for the top-10 most popular tracks for
# Initializing an empty list to store the top 10 tracks
top_10_per_decade_list = []

# Grouping the data by 'decade' and extract the top 10 tracks by 'popularity'
for decade, group in spotify_data.groupby('decade'):
    top_10 = group.nlargest(10, 'popularity')  # Get the top 10 tracks for this decade
    top_10_per_decade_list.append(top_10)      # Append the result to the list

# Concatenating all the top 10 results into a single DataFrame
top_10_per_decade = pd.concat(top_10_per_decade_list, ignore_index=True)

# Calculating the median danceability and duration (in minutes) for the top 10 tracks in each decade
median_top_10 = top_10_per_decade.groupby('decade').agg(
    median_danceability=('normalized_danceability', 'median'),
    median_duration_min=('duration_min', 'median')
).reset_index()

# Plotting with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# First axis for danceability
ax1.bar(median_top_10['decade'],
        median_top_10['median_danceability'],
        width=5,
        color='skyblue',
        alpha=0.7,
        label='Normalized Danceability (Top-10)')
ax1.set_xlabel('Decade')
ax1.set_ylabel('Normalized Danceability', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')

# Setting the range and ticks for the first axis
ax1.set_ylim(0, 1)  # Set range from 0 to 1

# Ensuring all x-axis labels are shown
plt.xticks(median_top_10['decade'], rotation=0)  # Keep labels horizontal and all visible

# Second axis for duration in minutes
ax2 = ax1.twinx()
ax2.plot(median_top_10['decade'], median_top_10['median_duration_min'],
         color='salmon', marker='o', label='Median Duration (min, Top-10)')
ax2.set_ylabel('Median Duration (min)', color='salmon')
ax2.tick_params(axis='y', labelcolor='salmon')

# Setting the range and ticks for the second axis
ax2.set_ylim(2.5, 5)  # Set range from 2.5 to 5
ax2.set_yticks([2.5, 3, 3.5, 4, 4.5, 5])  # Define ticks at 0.5 intervals

# Plotting the chart
fig.suptitle("Median Danceability and Duration (Top-10 Popular Tracks) by Decade", fontsize=14)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()
