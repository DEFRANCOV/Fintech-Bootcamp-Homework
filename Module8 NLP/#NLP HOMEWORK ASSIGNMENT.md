# NLP HOMEWORK ASSIGNMENT

## Sentiment Analysis

- We first import Bitcoin and Ethereum articles from the newsapi.  
- We create a function create_sentiment_df, to create a dataframe that reads in the articles’ content.   Within the function, we apply SentimentIntensityAnalyzer to the articles.  The output are four scores:  Compound, Positive, Negative and Neutral.   These four scores will form four respective columns in the dataframe.  
- We use the create_sentiment_df to create dataframes for Bitcoin and Ethereum.
- In general, Ethereum has a higher positive score, and a higher compound score than Bitcoin.

## Natural Language Processing

### Tokenizer
- We import word tokenize, stop words and WordLemmatizer function from nltk library.   We also import re.
- We combine the set of stopwords from words (English) with a self-made said, and combine them through a union command into sw_complete.
- We initiate WordNetLemmatizer.
- We create a function tokenizer that cleans punctuation and spaces and then tokenizes this polished data set.
- Next, we iterate through each word and if appropriate convert the word to a root word with the lemmatize.
- Finally, we iterate again through the lemmatized dataset, convert them to lowercase and remove the words that are within sw_complete.
- We apply tokenize function to the “text” column of the Bitcoin and Ethereum dataframes and create a “tokens” column in each dataframe.

### Ngrams and Frequency
- We import Counter from collections and import ngrams from nltk.
- We create a function called biglist, not BigFoot.   
- Within biglist, we initiate an empty list called big_list.   
- We then iterate through all rows, but only the “tokens” column of the argument dataframe.  Each value is actually a list itself, so we need to create a new iteration to go through each word within the list.  In essence the second iteration is creating one big list from all the lists (i.e., values) of the tokens column.  
- We use biglist to create a big list of all the tokens in each column of Bitcoin and Ethereum dataframes.
- We then apply the Counter function to these big lists, with the parameter of 2 to denote bigrams.  
- We finally create a token_count function to count the words in the big lists and through the parameter of 10 return the 10 most common words. 
 
### Wordclouds
- We convert the big lists into strings.
- Through the combination of WordCloud, matplotlib (pyplot and rcParams), we create a graphical display of the words within the lists.   The larger the word, the more the word is used.
- Scaling the data does not make any difference in outcomes.

## Named Entity Recognition
- From displacy we import spacy and load the “en_core_web_sm” function.
- We have already created large strings in Wordclouds above.
- We employ the en_core_web_sm to the strings.
- We then render the processed strings, and we can visually see the named entities which are only a subset of all the words.     




