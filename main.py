import sys
import subprocess
import os
import string
import math

print("Loading... Please wait.")

q = {}
q[1] = "1. How is artificial intelligence used in video games?"
q[2] = "2. What are the types of supervised learning?"
q[3] = "3. When was Python 3.0 released?"
q[4] = "4. What is backpropagation?"
q[5] = "5. How do neurons connect in a neural network?"
q[6] = "6. What is generalization in machine learning?"
q[7] = "7. What scripting language is used for natural language processing?"
q[8] = "8. Where is Python's name derived from?"

try:
    import nltk
except ImportError as e:
    print("\"nltk\" is not available. Install it with pip")

try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  print("\n Required nltk libraries not found. \nDownloading...")
  nltk.download('punkt')
  nltk.download('stopwords')

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Calculate IDF values across files
    files = load_files("corpus")
    file_words = {filename: tokenize(files[filename]) for filename in files}
    file_idfs = compute_idfs(file_words)
    print("\nThis is a snippet generating script. You can use one of the example questions or your own keywords to generate a snippet.")
    # Prompt user for query
    while True:
      print("\nWould you like to use an example question?\n")
      query = input("Yes/No: ")
      if query.lower() == "yes" or query.lower() == "y":
        print("\nExample Questions: \n")
        for k,v in q.items():
          print(v + "\n")
        query = input("Select number: ")
        try:
          if int(query) in range(1,9):
            query = q[int(query)]
          else:
            print("\nYou did not select a number from 1 to 8\n")
            continue
        except ValueError as e:
          # Probably entered a sentence/words
          print("\nWhat you have entered is not a number: {0}\n".format(query))
          continue
      elif query.lower() == "no" or query.lower() == "n":
        query = input("Please enter a question about Artificial intelligence, Machine learning, Python, Neural networks, Natural language processing or Probabilities: ")
      else:
         print("\nWhat you have entered is not applicable here. Please provide a yes or no answer\n")
         continue

      query = set(tokenize(query))
      # Determine top file matches according to TF-IDF
      filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

      # Extract sentences from top files
      sentences = dict()
      for filename in filenames:
          for passage in files[filename].split("\n"):
              for sentence in nltk.sent_tokenize(passage):
                  tokens = tokenize(sentence)
                  if tokens:
                      sentences[sentence] = tokens

      # Compute IDF values across sentences
      idfs = compute_idfs(sentences)

      # Determine top sentence matches
      matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
      print("\n[------------------------------------]\n")
      print("\n             ANSWER\n")
      for match in matches:
          print(match)
      print("\n[------------------------------------]")


def load_files(directory):
    """
    Given a directory name, returns a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's content as a string.
    """

    os.chdir(directory)  # Change directory

    corpus = os.listdir()  # Create a list of all files in directory

    # Create a dictionary of .txt filenames and corresponding content strings
    files = dict()
    for i in corpus:
        files.setdefault(i)
        with open(i, "r", encoding='utf-8') as f:
            files[i] = f.read()

    return files


def tokenize(document):
    """
    Given a `document` (string), returns a list of all of the
    words in that document, in order.

    Converst all words to lowercase and removes any
    punctuation or English stopwords.
    """

    words = []  # Initialise list of words

    # Split up sentences into tokens
    tokens = nltk.word_tokenize(document)

    for token in tokens:
        token = token.lower()  # Make all lowercase
        if token in nltk.corpus.stopwords.words(
                "english"):  # Remove English stopwords
            continue
        if token in string.punctuation:  # Remove punctuation
            continue

        words.append(token)  # Append token to list of words

    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps the names of .txt files 
    to lists of their words, returns a `idfs` dictionary that maps words to their 
    inverse document frequency (IDF) values.
    """

    idfs = dict()  # Initialise dictionary of words and corresponding IDFs

    for txt_file in documents:
        list_unique = []  # Create a list of unique words for each .txt file

        # Count the number of times each word appears in a .txt file
        for word in documents[txt_file]:
            if word not in list_unique:
                list_unique.append(word)
                idfs.setdefault(word, 0)
                idfs[word] += 1

    # Calculate IDF values for each word in `idfs` dictionary
    for word in idfs.keys():
        idfs[word] = math.log(len(documents) / idfs[word])

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping the names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), returns a list of the filenames of the the `n` top
    files that match the query, ranked according to term frequency-inverse document frequency (TF-IDF).
    """

    # Create a TF-IDF dictionary
    tf_idf = dict()

    for txt_file in files:
        tf_idf.setdefault(txt_file, 0)

        # Update TF-IDF value for each .txt file
        for word in query:
            if word in files[txt_file]:
                tf_idf[txt_file] += files[txt_file].count(word) * idfs[word]

    # Sort dictionary by TF-IDF values in descending order
    tf_idf = dict(sorted(tf_idf.items(), key=lambda x: x[1], reverse=True))

    # Return a list of the top `n` .txt filenames maching the query
    top_n = list(tf_idf.keys())[:n]

    return top_n


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), returns a list of the `n` top sentences that match
    the query, ranked according to their SUMMED inverse inverse document frequency (IDF) 
    and then by their query term density (QTD). 
    """

    # Create a dictionary of all sentences and their IDF (summed) and QTD values (list[IDF, QTD])
    ranking = dict()
    for sentence in sentences:
        ranking.setdefault(sentence, [0, 0])
        for word in query:
            if word in sentences[sentence]:
                ranking[sentence][0] += idfs[word]  # IDF
                ranking[sentence][1] += list(
                    sentences[sentence]).count(word) / len(
                        sentences[sentence])  # QTD

    # Sort dictionary values by IDF (descending) and then QTD (descending)
    ranking = dict(
        sorted(sorted(ranking.items(), key=lambda x: x[0], reverse=True),
               key=lambda x: x[1],
               reverse=True))

    # Return a list of the top `n` sentences maching the query
    top_n = list(ranking.keys())[:n]

    return top_n


if __name__ == "__main__":
    main()