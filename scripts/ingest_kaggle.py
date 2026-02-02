"""
Quick Kaggle ingestion helper.

- Requires you to have Kaggle API configured: place kaggle.json in ~/.kaggle with API credentials.
- Example dataset argument (replace with your target courses dataset):
    kaggle datasets download --dataset stackoverflow/stacksample

This script shows how you'd programmatically download & extract metadata for courses.
For demo, it reads our existing CSVs and prints counts.
"""

import os
import pandas as pd


def main():
	root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	data_dir = os.path.join(root, 'data')
	courses = pd.read_csv(os.path.join(data_dir, 'courses.csv'))
	print('Courses:', len(courses))
	if os.path.exists(os.path.join(data_dir, 'interactions.csv')):
		print('Interactions:', len(pd.read_csv(os.path.join(data_dir, 'interactions.csv'))))


if __name__ == '__main__':
	main()


