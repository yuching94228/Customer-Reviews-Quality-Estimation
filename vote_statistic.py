# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:12:07 2016
2017/02/13
@author: yuching

"""
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

dict_id={}
count=0
count2=0
with open('C:/Users/yuching/Desktop/py_code/data/rating.txt', 'r') as fid:
    for line in fid:
        temp=line.split('::::')
        voter_score=(str(temp[7])[0:len(temp[7]) - 1]).split(":::")
        if(voter_score[len(voter_score)-1]=="</endperson>"):
            for rac in voter_score:
                if(str(rac) <> "</endperson>"):
                    voter = rac.split(':')
                    if(is_number(str(voter[0]))==True and is_number(str(temp[0]))==True and voter[0]<>temp[0] and voter[1] is not None ):
                        id=(voter[0] + ":" + temp[0])

                        if(id in dict_id):
                            dict_id[id][0]=str(int(dict_id[id][0])+int(voter[1]))
                            dict_id[id][1]=str(int(dict_id[id][1])+1)
                        else:
                            # print id,voter[1]
                            dict_id.setdefault(id,[voter[1],1])
            count=count+1
            print count
fid.close()

count=0
filew = open('C:/Users/yuching/Desktop/py_code/data/output.txt', 'w')
filew.write("VOTER,POSTER,TOTALSCORE,TOTALVOTE,AVGSCORE\n")
for item in dict_id:
    count=count+1
    vot_pos=str(item).split(':')
    print count
    filew.write("%s,%s,%s,%s,%s\n" % (str(vot_pos[0]),str(vot_pos[1]),str(dict_id[item][0]),str(dict_id[item][1]),str(round((float(dict_id[item][0])/float(dict_id[item][1])),3))))
filew.close()



