#NLP LAB 1
import re

#reading the file.
def readFile(file):
    MainClassification=[]
    blah= {}
    for line in open(file):
        # getting rid of all regular expressions not needed and substitues it with my own something
        lineCleaner = re.sub(r"[,/?!-()\t\n*&^}:;{=$%]","",line)
        lineCleaner2 = re.sub(r"[.']"," ",lineCleaner)
        final = MainClassification.append(lineCleaner2)
        value = int(lineCleaner2[-1][-1])
        if value in blah:
            blah[value].append(lineCleaner2[:-1])
            print(blah)
        else:
            blah[value] = [lineCleaner2[:-1]]
            print(blah)
    
    #return MainClassification

readFile('amazon_cells_labelled.txt')
    
        
def split():
    positiveClass={}
    negativeClass ={}
    MainClassification, lineCleaner2 = readFile('amazon_cells_labelled.txt')
    value = int(lineCleaner2[-1][-1])
    if value in MainClassification:
        MainClassification[value].append(lineCleaner2[:-1])
    else:
        MainClassification[value] = [lineCleaner2[:-1]]
    return value

def PostiveClass(MainClaasification):
     blah ={}
     for line in MainClassification[1]:
         value = int(lineCleaner2[-1][-1])
         if value in blah:
             blah[value].append(lineCleaner2[:-1])
             print(blah)
         else:
             blah[value] = [lineCleaner2[:-1]]
             print(blah)
     return blah
