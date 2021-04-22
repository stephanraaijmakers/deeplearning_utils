import sys
import re

with open(sys.argv[1],"r",encoding="ISO-8859-1") as fp:
    lines=fp.readlines()
    print("word,pos,ner")
    for line in lines[1:]:
        line=line.rstrip()
        if line=="":
            continue
        line=re.sub("[\"\,]+","",line)
        m=re.match("([^\s]+)\s+([^\s]+)\s+(.+)$",line)
        if m:
            print("%s,%s,%s"%(m.group(1),m.group(2),m.group(3)))
    
            
        
