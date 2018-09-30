#NLP LAB 1 
import re

#reading the file.
def readFile(files):
    MainClassification=[]
    data = {
        0:[],
        1:[]}
    for file in files:
        for line in open(file):
            # getting rid of all regular expressions not needed and substitues it with my own something
            lineCleaner = re.sub(r"[,/?!-()*&^}:;{=$%]","",line)
            lineCleaner2 = re.sub(r"[.']"," ",lineCleaner.casefold())
            final = MainClassification.append(lineCleaner2)
            review = line.split('\t')
            features= review[0].split()
            label = int(review[1])
            if label == 0:
                data[0].append(features)
            else:
                data[1].append(features)
            
    print("The total words in the negative class is: " , len(data[0])) 
    print("The total words in the positive class is: " , len(data[1])) 
    return data
readFile(['amazon_cells_labelled.txt',"imdb_labelled.txt","yelp_labelled.txt"])




















##import re
##
###reading the file.
##def readFile(file):
##    MainClassification=[]
##    words= {}
##    for line in open(file):
##        # getting rid of all regular expressions not needed and substitues it with my own something
##        lineCleaner = re.sub(r"[,/?!-()\t\n*&^}:;{=$%]","",line)
##        lineCleaner2 = re.sub(r"[.']"," ",lineCleaner)
##        final = MainClassification.append(lineCleaner2)
##        value = int(lineCleaner2[-1][-1])
##        if value in words:
##            words[value].append(lineCleaner2[:-1])
##            print("YAYY",words)
##        else:
##            words[value] = [lineCleaner2[:-1]]
##            print("I am printing:", words)
##    
##    #return MainClassification
##
##readFile('amazon_cells_labelled.txt')
