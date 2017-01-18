import os

for filename in os.listdir("."):
   if filename.endswith(".new"):
     os.rename(filename, filename[:14]+'_new.fts')
 
