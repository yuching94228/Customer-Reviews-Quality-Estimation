# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:12:07 2016
2017/02/13
@author: yuching

"""

in_content_root = '/Users/yuching/Desktop/test_data/FINAL_DATA/CONTENT.csv'
in_user_root = '/Users/yuching/Desktop/test_data/FINAL_DATA/USER.csv'
in_network_root = '/Users/yuching/Desktop/test_data/FINAL_DATA/NETWORK.csv'

out_1_root = '/Users/yuching/Desktop/test_data/FINAL_DATA/3_numerical/1_content/content.arff'
out_2_root = '/Users/yuching/Desktop/test_data/FINAL_DATA/3_numerical/2_user/user.arff'
out_3_root = '/Users/yuching/Desktop/test_data/FINAL_DATA/3_numerical/3_network/network.arff'
out_4_root = '/Users/yuching/Desktop/test_data/FINAL_DATA/3_numerical/4_content+user/content+user.arff'
out_5_root = '/Users/yuching/Desktop/test_data/FINAL_DATA/3_numerical/5_content+network/content+network.arff'
out_6_root = '/Users/yuching/Desktop/test_data/FINAL_DATA/3_numerical/6_user+network/user+network.arff'
out_7_root = '/Users/yuching/Desktop/test_data/FINAL_DATA/3_numerical/7_content+user+network/content+user+network.arff'

# user to dictionary
user_dict = {}
with open(in_user_root, 'r') as fuser:
    for line in fuser:
        temp = line.split(',')
        user_dict.setdefault(temp[0],[temp[1],temp[2],temp[3][0:len(temp[3])-1]])
fuser.close()

# network to dictionary
network_dict = {}
with open(in_network_root, 'r') as fnetwork:
    for line in fnetwork:
        temp = line.split(',')
        network_dict.setdefault(temp[0],[temp[1],temp[2],temp[3],temp[4],temp[5],temp[6],temp[7][0:len(temp[7])-1]])
fnetwork.close()

fw1 = open(out_1_root, 'w')
fw1.write("@relation content\n\n"
          "@attribute Overall {0,1,2,3,4,5}\n"
          "@attribute Days numeric\n"
          "@attribute Sentence numeric\n"
          "@attribute Words numeric\n"
          "@attribute GIW_pos numeric\n"
          "@attribute GIW_neg numeric\n"
          "@attribute GIW_add numeric\n"
          "@attribute SWN_pos numeric\n"
          "@attribute SWN_neg numeric\n"
          "@attribute SWN_add numeric\n"
          "@attribute pos_ADJ numeric\n"
          "@attribute pos_ADP numeric\n"
          "@attribute pos_ADV numeric\n"
          "@attribute pos_CONJ numeric\n"
          "@attribute pos_DET numeric\n"
          "@attribute pos_NOUN numeric\n"
          "@attribute pos_NUM numeric\n"
          "@attribute pos_PRT numeric\n"
          "@attribute pos_PRON numeric\n"
          "@attribute pos_VERB numeric\n"
          "@attribute pos_punctuation numeric\n"
          "@attribute Helpful numeric\n\n"
          "@data\n")
fw2 = open(out_2_root, 'w')
fw2.write("@relation user\n\n"
          "@attribute registering numeric\n"
          "@attribute reviews numeric\n"
          "@attribute trustors numeric\n"
          "@attribute Helpful numeric\n\n"
          "@data\n")
fw3 = open(out_3_root, 'w')
fw3.write("@relation network\n\n"
          "@attribute deg_cen numeric\n"
          "@attribute in_deg_cen numeric\n"
          "@attribute out_deg_cen numeric\n"
          "@attribute clo_cen numeric\n"
          "@attribute PageR numeric\n"
          "@attribute betw_cen numeric\n"
          "@attribute eigen_cent numeric\n"
          "@attribute Helpful numeric\n\n"
          "@data\n")
fw4 = open(out_4_root, 'w')
fw4.write("@relation content+user\n\n"
          "@attribute Overall {0,1,2,3,4,5}\n"
          "@attribute Days numeric\n"
          "@attribute Sentence numeric\n"
          "@attribute Words numeric\n"
          "@attribute GIW_pos numeric\n"
          "@attribute GIW_neg numeric\n"
          "@attribute GIW_add numeric\n"
          "@attribute SWN_pos numeric\n"
          "@attribute SWN_neg numeric\n"
          "@attribute SWN_add numeric\n"
          "@attribute pos_ADJ numeric\n"
          "@attribute pos_ADP numeric\n"
          "@attribute pos_ADV numeric\n"
          "@attribute pos_CONJ numeric\n"
          "@attribute pos_DET numeric\n"
          "@attribute pos_NOUN numeric\n"
          "@attribute pos_NUM numeric\n"
          "@attribute pos_PRT numeric\n"
          "@attribute pos_PRON numeric\n"
          "@attribute pos_VERB numeric\n"
          "@attribute pos_punctuation numeric\n"
          "@attribute registering numeric\n"
          "@attribute reviews numeric\n"
          "@attribute trustors numeric\n"
          "@attribute Helpful numeric\n\n"
          "@data\n")
fw5 = open(out_5_root, 'w')
fw5.write("@relation content+network\n\n"
          "@attribute Overall {0,1,2,3,4,5}\n"
          "@attribute Days numeric\n"
          "@attribute Sentence numeric\n"
          "@attribute Words numeric\n"
          "@attribute GIW_pos numeric\n"
          "@attribute GIW_neg numeric\n"
          "@attribute GIW_add numeric\n"
          "@attribute SWN_pos numeric\n"
          "@attribute SWN_neg numeric\n"
          "@attribute SWN_add numeric\n"
          "@attribute pos_ADJ numeric\n"
          "@attribute pos_ADP numeric\n"
          "@attribute pos_ADV numeric\n"
          "@attribute pos_CONJ numeric\n"
          "@attribute pos_DET numeric\n"
          "@attribute pos_NOUN numeric\n"
          "@attribute pos_NUM numeric\n"
          "@attribute pos_PRT numeric\n"
          "@attribute pos_PRON numeric\n"
          "@attribute pos_VERB numeric\n"
          "@attribute pos_punctuation numeric\n"
          "@attribute deg_cen numeric\n"
          "@attribute in_deg_cen numeric\n"
          "@attribute out_deg_cen numeric\n"
          "@attribute clo_cen numeric\n"
          "@attribute PageR numeric\n"
          "@attribute betw_cen numeric\n"
          "@attribute eigen_cent numeric\n"
          "@attribute Helpful numeric\n\n"
          "@data\n")
fw6 = open(out_6_root, 'w')
fw6.write("@relation user+network\n\n"
          "@attribute registering numeric\n"
          "@attribute reviews numeric\n"
          "@attribute trustors numeric\n"
          "@attribute deg_cen numeric\n"
          "@attribute in_deg_cen numeric\n"
          "@attribute out_deg_cen numeric\n"
          "@attribute clo_cen numeric\n"
          "@attribute PageR numeric\n"
          "@attribute betw_cen numeric\n"
          "@attribute eigen_cent numeric\n"
          "@attribute Helpful numeric\n\n"
          "@data\n")
fw7 = open(out_7_root, 'w')
fw7.write("@relation content+user+network\n\n"
          "@attribute Overall {0,1,2,3,4,5}\n"
          "@attribute Days numeric\n"
          "@attribute Sentence numeric\n"
          "@attribute Words numeric\n"
          "@attribute GIW_pos numeric\n"
          "@attribute GIW_neg numeric\n"
          "@attribute GIW_add numeric\n"
          "@attribute SWN_pos numeric\n"
          "@attribute SWN_neg numeric\n"
          "@attribute SWN_add numeric\n"
          "@attribute pos_ADJ numeric\n"
          "@attribute pos_ADP numeric\n"
          "@attribute pos_ADV numeric\n"
          "@attribute pos_CONJ numeric\n"
          "@attribute pos_DET numeric\n"
          "@attribute pos_NOUN numeric\n"
          "@attribute pos_NUM numeric\n"
          "@attribute pos_PRT numeric\n"
          "@attribute pos_PRON numeric\n"
          "@attribute pos_VERB numeric\n"
          "@attribute pos_punctuation numeric\n"
          "@attribute registering numeric\n"
          "@attribute reviews numeric\n"
          "@attribute trustors numeric\n"
          "@attribute deg_cen numeric\n"
          "@attribute in_deg_cen numeric\n"
          "@attribute out_deg_cen numeric\n"
          "@attribute clo_cen numeric\n"
          "@attribute PageR numeric\n"
          "@attribute betw_cen numeric\n"
          "@attribute eigen_cent numeric\n"
          "@attribute Helpful numeric\n\n"
          "@data\n")


data_row = 0
with open(in_content_root, 'r') as f:
    for line in f:
        temp = line.split(',')
        user = user_dict[temp[0]]
        network = network_dict[temp[0]]
        for i in range(1, 22, 1):
            fw1.write("%s%s" % (temp[i], ','))
            fw4.write("%s%s" % (temp[i], ','))
            fw5.write("%s%s" % (temp[i], ','))
            fw7.write("%s%s" % (temp[i], ','))
        for i in range(0, 3, 1):
            fw2.write("%s%s" % (user[i], ','))
            fw4.write("%s%s" % (user[i], ','))
            fw6.write("%s%s" % (user[i], ','))
            fw7.write("%s%s" % (user[i], ','))
        for i in range(0, 7, 1):
            fw3.write("%s%s" % (network[i], ','))
            fw5.write("%s%s" % (network[i], ','))
            fw6.write("%s%s" % (network[i], ','))
            fw7.write("%s%s" % (network[i], ','))
        fw1.write("%s%s" % (temp[23][0:len(temp[23])-1], '\n'))
        fw2.write("%s%s" % (temp[23][0:len(temp[23])-1], '\n'))
        fw3.write("%s%s" % (temp[23][0:len(temp[23])-1], '\n'))
        fw4.write("%s%s" % (temp[23][0:len(temp[23])-1], '\n'))
        fw5.write("%s%s" % (temp[23][0:len(temp[23])-1], '\n'))
        fw6.write("%s%s" % (temp[23][0:len(temp[23])-1], '\n'))
        fw7.write("%s%s" % (temp[23][0:len(temp[23])-1], '\n'))

        # if(temp[22]=="4" or temp[22]=="5"):
        #     fw1.write('H\n')
        #     fw2.write('H\n')
        #     fw3.write('H\n')
        #     fw4.write('H\n')
        #     fw5.write('H\n')
        #     fw6.write('H\n')
        #     fw7.write('H\n')
        # else:
        #     fw1.write('U\n')
        #     fw2.write('U\n')
        #     fw3.write('U\n')
        #     fw4.write('U\n')
        #     fw5.write('U\n')
        #     fw6.write('U\n')
        #     fw7.write('U\n')

        data_row+=1
        print data_row
f.close()

fw1.close()
fw2.close()
fw3.close()
fw4.close()
fw5.close()
fw6.close()
fw7.close()

