import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load course and interaction data
courses_df = pd.read_csv('courses.csv')
interactions_df = pd.read_csv('interactions.csv')
print("Courses Data Info:")
courses_df.info()
print("\nInteractions Data Info:")
interactions_df.info()

# Visualize the distribution of course ratings
plt.figure(figsize=(8, 6))
sns.histplot(courses_df['rating'], kde=True)
plt.title('Distribution of Course Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Analyze interactions per user
interactions_per_user = interactions_df['user_id'].value_counts()
print("\nInteractions per user stats:")
print(interactions_per_user.describe())