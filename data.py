import os
from pathlib import Path


def get_data_from_files(data_path = Path('./data')):
    files = os.listdir(data_path)
    print(f"List of files with data: {files}")


    data_files = []
    reference_files = []
    for file in files:
        with open(data_path / file, 'r', encoding='utf-8') as f:
            reference_files.append(f.readline().rstrip('\n'))
            data_files.append(f.read())
    return data_files, reference_files

def split_articles(articles, split_function: callable):
    articles_splitted = []
    for index, article in articles.iterrows():
        splitted_data = split_function(article["body_cleanned"])
        for split in splitted_data:
            articles_splitted.append((split, index))

    return articles_splitted