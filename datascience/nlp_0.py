from collections import Counter 
#text = "Hello there crush! your tear your stones apart apart crush your flag till you cry red is blue nanna-chan are reminded of your nanna-chan! Shinzou wa sasageyo"
def read_book(title_path):
    """
    Read a book amd return it as a string
    """
    with open(title_path, "r", encoding="utf8") as current_file:
        text = current_file.read()
        text = text.replace("\n", "").replace("\r", "")
    return text
#%%
def count_words(text):
    """
    Returns a dict of unique words labelled with respective frequencies
    """
    text = text.lower()
    skips = [".", ",", "!", "?", "'", '"']
    for ch in skips:
        text = text.replace(ch, "")
    word_counts = {}
    word_counts = Counter(text.split(" "))
    return word_counts
#%%
def word_stats(word_counts):
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)
#%%
text = read_book("/home/eihpi/Python-workspace/datascience/Case-studies:Translation/English/shakespeare/Romeo and Juliet.txt")
word_counts = count_words(text)
(num_unique, counts) = word_stats(word_counts)
print(num_unique, sum(counts))
text = read_book("/home/eihpi/Python-workspace/datascience/Case-studies:Translation/German/shakespeare/Romeo und Julia.txt")
word_counts = count_words(text)
(num_unique, counts) = word_stats(word_counts)
print(num_unique, sum(counts))
   
    #%%








