import operator
import re
import gensim as gs

class TrieNode():
    def __init__(self):
        self.children = {}
        self.flag = False

class Trie():
    def __init__(self):
        self.root = TrieNode()
        self.wordList = []

    def createTrie(self,keys):
        for it in keys:
            self.addWord(it)


    def addWord(self, key):
        node = self.root
        for it in list(key):
            if not node.children.get(it):
                node.children[it] = TrieNode()

            node = node.children[it]
        node.flag = True

    def searchWord(self, word):
        node = self.root
        found = True
        for a in list(word):
            if not node.children.get(a):
                return False
                break
            node = node.children[a]
        return node and node.flag and found


def get_words_max_frequence(document,removed_words, symbols):
    t = Trie()
    t.createTrie(removed_word)
    # document = document.lower()
    # words = re.findall(r"[\w]+",document)
    words = gs.utils.simple_preprocess(document)
    print (words)

    list_word = {}
    for a in words:
        if not (t.searchWord(a)):
            if (a in list_word):
                frequence = list_word.get(a)
                frequence += 1
                list_word.update({a: frequence})
            else:
                list_word.update({a: 1})

    print(list_word)
    max_frequence = max(list_word.items(), key=operator.itemgetter(1))[1]

    for key, value in list_word.items():
        if value == max_frequence:
            print(key)

if __name__  == '__main__':

    removed_word =["a", "is", "in", "by", "to", "at", "of", "and", "the"]
    symbols = [".", ",", "!", "'"]

    document = """There is a plot, Harry Potter. A plot to make most terrible things things
     happen at Hogwarts School of Witchcraft and Wizardry this year. Harry Potter's
     summer has included the worst birthday ever, doomy warnings from a house-elf 
     called Dobby, and rescue from the Dursleys by his friend Ron Weasley in a 
     magical flying car!"""

    get_words_max_frequence(document,removed_word,symbols)




