import re, json
import matplotlib.pyplot as plt 
import numpy as np

######################################################################
##						MUSES										##
######################################################################

''' 
	This script is for providing figures from results
'''


original_result_acc = {'XGB': 64.32180334240186,
'SVMRBF': 62.88379323746599,
'DT': 42.79051690633502,
'NN1': 61.67897396035757,
'NN2': 59.46366109599689,
'LR':  53.75048581422463,
'SVMLinear': 61.01826661484648,
'ET': 57.675864749319864}

original_result_bl_acc = {'XGB': 38.73885598812973,
'SVMRBF': 40.11915227094146,
'DT': 23.414982991329484,
'NN1': 40.366002445578786,
'NN2': 38.85517823491521,
'LR':  33.00442665818106,
'SVMLinear': 38.85042015536712,
'ET': 28.04807911674886}

original_acc = [64.32180334240186,
62.88379323746599,
42.79051690633502,
 61.67897396035757,
59.46366109599689,
53.75048581422463,
61.01826661484648,
57.675864749319864]

original_bl_acc = [38.73885598812973,
40.11915227094146,
23.414982991329484,
40.366002445578786,
38.85517823491521,
33.00442665818106,
38.85042015536712,
28.04807911674886
]

smote_acc = [64.43839875631558 , 61.67897396035757,34.628837932374665,60.90167120093276,58.725223474543334,46.24951418577536,57.559269335406135,61.09599689078896]
smote_bl_acc = [41.33869398310072,38.88805012510409,26.293540020646557, 40.81212024703827,39.527754140423795
,34.528158744513185
,41.17219539857017
,43.754419161340344]
dim = [150, 225, 300, 375, 450]
XGB_acc = [34.20132141469102,34.00699572483482,34.784298484259615,34.4733773804897,35.2506801399145]
SVMRBF_acc = [30.31480761756704,29.14885347842985,27.905169063350176,27.12786630392538, 26.816945200155462]
DT_acc = [20.054411193159734,18.53867081228138,19.471434123591138,20.054411193159734,21.41469102215313]
NN1_acc = [21.453556160124368,21.41469102215313,23.39681305868636,22.619510299261563,22.89156626506024]
NN2_acc = [19.97668091721726,20.481927710843372,20.79284881461329,21.958802953750485,23.39681305868636]
LR_acc = [5.67431014380101,8.083948698017878,9.249902837155071,7.69529731830548,5.985231247570929]
SVMLinear_acc = [20.870579090555772,21.25923047026817,22.813835989117763 , 21.764477263894285,21.647881849980568]
ET_acc = [42.51846094053634, 42.20753983676642,41.85775359502526, 41.080450835600466, 40.61406917994559]

XGB_bl_acc=[8.735053198599054,8.648903466712428, 8.901607682420472,8.809935421839986, 8.982409011953253]
SVMRBF_bl_acc=[ 7.986288866134393,7.79748356361446,7.658166166152982,7.51326297871565,7.596712185978148]
DT_bl_acc = [6.74178565449809,6.868891231982209,6.258118782985192,7.266944642025386,7.425089602189376]
NN1_bl_acc = [6.588074614508901,6.416004761432564, 6.786591376003794,6.861194434995134,6.92999843399712]
NN2_bl_acc = [6.411145869286225,6.3189188869520905,6.52289780286405,6.526405311497535,6.700187669560733]
LR_bl_acc = [8.170297269447175,5.266652045604051, 9.497006561729343,6.664204406564342, 5.752705437000709]
SVMLinear_bl_acc = [7.424983336015391,6.6355292754085555,7.097241720415225,6.778463135054051,6.848463511849179]
ET_bl_acc = [10.17743495077356,10.090629395218004,9.966156821378341,9.756285161744021,9.6123417721519]

XGB_acc_true = [item/original_result_acc['XGB'] for item in XGB_acc]
SVMRBF_acc_true = [item/original_result_acc['SVMRBF'] for item in SVMRBF_acc]
DT_acc_true = [item/original_result_acc['DT'] for item in DT_acc]
NN1_acc_true = [item/original_result_acc['NN1'] for item in NN1_acc]
NN2_acc_true = [item/original_result_acc['NN2'] for item in NN2_acc]
LR_acc_true = [item/original_result_acc['LR'] for item in LR_acc]
SVMLinear_acc_true = [item/original_result_acc['SVMLinear'] for item in SVMLinear_acc]
ET_acc_true = [item/original_result_acc['ET'] for item in ET_acc]

XGB_bl_acc_true = [item/original_result_bl_acc['XGB'] for item in XGB_bl_acc]
SVMRBF_bl_acc_true = [item/original_result_bl_acc['SVMRBF'] for item in SVMRBF_bl_acc]
DT_bl_acc_true = [item/original_result_bl_acc['DT'] for item in DT_bl_acc]
NN1_bl_acc_true = [item/original_result_bl_acc['NN1'] for item in NN1_bl_acc]
NN2_bl_acc_true = [item/original_result_bl_acc['NN2'] for item in NN2_bl_acc]
LR_bl_acc_true = [item/original_result_bl_acc['LR'] for item in LR_bl_acc]
SVMLinear_bl_acc_true = [item/original_result_bl_acc['SVMLinear'] for item in SVMLinear_bl_acc]
ET_bl_acc_true = [item/original_result_bl_acc['ET'] for item in ET_bl_acc]
# mean_test_acc_l2 = [mean_test_acc[i] for i in range(len(mean_test_acc)) if i %2 == 0]
# mean_test_acc_l1 = [mean_test_acc[i] for i in range(len(mean_test_acc)) if i %2 == 1]
# mean_test_bl_acc_l2 = [mean_test_bl_acc[i] for i in range(len(mean_test_acc)) if i %2 == 0]
# mean_test_bl_acc_l1 = [mean_test_bl_acc[i] for i in range(len(mean_test_acc)) if i %2 == 1]


##-------------------------------------------------------------------------
## Logistic Regression
##-------------------------------------------------------------------------
# lr_acc_cv = [0.60792786, 0.57063694, 0.61640926, 0.58925549, 0.60120069,
#        0.54376791, 0.59652455, 0.41356961, 0.54472485, 0.28516231,
#        0.49086579, 0.2788386 , 0.3860658 , 0.17926758, 0.14085774,
#        0.27247268]
# lr_bl_acc_cv = [0.37361398, 0.38687049, 0.32793345, 0.3463648 , 0.2916399 ,
#        0.23283052, 0.2865951 , 0.09588156, 0.24074878, 0.0625    ,
#        0.18267074, 0.0625    , 0.13255623, 0.0625    , 0.0625    ,
#        0.0625    ]

# lr_lambda_range = [0.0001, 0.0001, 0.001, 0.001, 0.01, 0.01, 0.1, 0.1, 1,
#                    1, 10, 10, 100, 100, 1000, 1000]
# lr_lambda_range_true = [str(lr_lambda_range[i]) for i in range(len(lr_lambda_range)) if i % 2 == 0]
# lr_l1_acc = [lr_acc_cv[i] for i in range(len(lr_acc_cv)) if i % 2 != 0]
# lr_l2_acc = [lr_acc_cv[i] for i in range(len(lr_acc_cv)) if i % 2 == 0]
# lr_l1_bl_acc = [lr_bl_acc_cv[i] for i in range(len(lr_bl_acc_cv)) if i % 2 != 0]
# lr_l2_bl_acc = [lr_bl_acc_cv[i] for i in range(len(lr_bl_acc_cv)) if i % 2 == 0]

# fig,ax = plt.subplots(figsize=(10,10))
# ax.locator_params(nbins=6, axis='y')
# ax.plot(lr_lambda_range_true, lr_l1_acc, label='L1-acc')
# ax.plot(lr_lambda_range_true, lr_l2_acc, label='L2-acc')
# ax.plot(lr_lambda_range_true, lr_l1_bl_acc, label='L1-bl-acc')
# ax.plot(lr_lambda_range_true, lr_l2_bl_acc, label='L2-bl-acc')
# ax.set_xlabel("lambda", fontsize=25)
# ax.set_ylabel("Accuracy", fontsize=25)
# ax.legend(fontsize=20)
# ax.tick_params(axis='both', labelsize=20)
# plt.savefig("/home/kei/newfig/lr_train.png", format='png', dpi=300)
# plt.close()

##-------------------------------------------------------------------------
## Decision Tree
##-------------------------------------------------------------------------
# dt_acc_cv = [0.49157165, 0.51305511, 0.5098935 , 0.49598724, 0.4767143 ,
#        0.46662553, 0.45924503, 0.45568082, 0.45307026, 0.45161526]
# dt_bl_acc_cv = [0.23589125, 0.2528208 , 0.27315315, 0.27580106, 0.28127787,
#        0.28708938, 0.28294568, 0.27850732, 0.28820857, 0.28438463]

# dt_range = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

# fig,ax = plt.subplots(figsize=(10,10))
# ax.locator_params(nbins=6, axis='y')
# ax.plot(dt_range, dt_acc_cv, label='DT-acc')
# ax.plot(dt_range, dt_bl_acc_cv, label='DT-bl-acc')
# ax.set_xlabel("max_depth", fontsize=25)
# ax.set_ylabel("Accuracy", fontsize=25)
# ax.legend(fontsize=20)
# ax.tick_params(axis='both', labelsize=20)
# plt.savefig("/home/kei/newfig/dt_train.png", format='png', dpi=300)
# plt.close()

##-------------------------------------------------------------------------
## SVM
##-------------------------------------------------------------------------
# svm_acc_cv = [0.28516231, 0.57173235, 0.28516231, 0.65194684, 0.46210481,
#        0.66564978, 0.59587605, 0.65134313, 0.68070891, 0.63241957]
# svm_bl_acc_cv = [0.0625    , 0.27326264, 0.0625    , 0.36075077, 0.10734573,
#        0.4295786 , 0.29157954, 0.45932032, 0.41441575, 0.45076798]

# svm_linear_acc_cv = [svm_acc_cv[i] for i in range(len(svm_acc_cv)) if i % 2 != 0]
# svm_linear_bl_acc_cv = [svm_bl_acc_cv[i] for i in range(len(svm_bl_acc_cv)) if i % 2 != 0]

# svm_rbf_acc_cv = [0.28516231, 0.28516231, 0.46210481, 0.59587605, 0.68070891,
#        0.69908039, 0.69717207, 0.69747299]
# svm_rbf_bl_acc_cv = [0.0625    , 0.0625    , 0.10734573, 0.29157954, 0.41441575,
#        0.51469212, 0.51690645, 0.51691278]

# svm_c_linear = [0.0001, 0.001, 0.01, 0.1, 1]
# svm_c_linear = [str(svm_c_linear[i]) for i in range(len(svm_c_linear))]
# svm_c_rbf = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
# svm_c_rbf = [str(svm_c_rbf[i]) for i in range(len(svm_c_rbf))]

# fig,ax = plt.subplots(figsize=(10,10))
# ax.locator_params(nbins=6, axis='y')
# ax.plot(svm_c_linear, svm_linear_acc_cv, label='SVM_linear-acc')
# ax.plot(svm_c_linear, svm_linear_bl_acc_cv, label='SVM_linear-bl-acc')
# ax.set_xlabel("C", fontsize=25)
# ax.set_ylabel("Accuracy", fontsize=25)
# ax.set_title("SVM Linear", fontsize=25)
# ax.legend(fontsize=20)
# ax.tick_params(axis='both', labelsize=20)
# plt.savefig("/home/kei/newfig/svm_linear_train.png", format='png', dpi=300)
# plt.close()

# fig,ax = plt.subplots(figsize=(10,10))
# ax.locator_params(nbins=6, axis='y')
# ax.plot(svm_c_rbf, svm_rbf_acc_cv, label='SVM_RBF-acc')
# ax.plot(svm_c_rbf, svm_rbf_bl_acc_cv, label='SVM_RBF-bl-acc')
# ax.set_xlabel("C", fontsize=25)
# ax.set_ylabel("Accuracy", fontsize=25)
# ax.set_title("SVM RBF", fontsize=25)
# ax.legend(fontsize=20)
# ax.tick_params(axis='both', labelsize=20)
# plt.savefig("/home/kei/newfig/svm_rbf_train.png", format='png', dpi=300)
# plt.close()

##-------------------------------------------------------------------------
## ExtraTrees
##-------------------------------------------------------------------------
# et_acc_cv = [0.45542846, 0.45728823, 0.45532927, 0.45708653, 0.45693586,
#        0.48098047, 0.48253545, 0.48163141, 0.48278637, 0.48318819,
#        0.51164963, 0.51275417, 0.51225089, 0.51350593, 0.51305494,
#        0.53805398, 0.53679879, 0.538606  , 0.53785257, 0.53775219,
#        0.56204483, 0.56104433, 0.56119523, 0.56229871, 0.56109532,
#        0.58051716, 0.58192561, 0.58187428, 0.58272814, 0.58247763,
#        0.59572756, 0.59708333, 0.59783639, 0.59763608, 0.5984899 ,
#        0.60340834, 0.60852514, 0.60797528, 0.60902928, 0.60892873]
# et_bl_acc_cv = [0.12496188, 0.12740019, 0.12522667, 0.12748523, 0.12674664,
#        0.17608089, 0.17795651, 0.17629229, 0.17717304, 0.17768114,
#        0.21868652, 0.22087241, 0.21993424, 0.22178371, 0.22047504,
#        0.25158917, 0.24982921, 0.25180979, 0.25157866, 0.2515648 ,
#        0.27688213, 0.27459109, 0.27568721, 0.27589943, 0.27563959,
#        0.29587119, 0.29631734, 0.29706574, 0.29775735, 0.29743037,
#        0.31547528, 0.31485884, 0.31531296, 0.31651541, 0.31608395,
#        0.32341837, 0.32964574, 0.3286945 , 0.33034689, 0.33136286]

# et_100 	= [et_acc_cv[i] for i in range(len(et_acc_cv)) if i % 5 == 0]
# et_500 	= [et_acc_cv[i] for i in range(len(et_acc_cv)) if i % 5 == 1]
# et_1000 = [et_acc_cv[i] for i in range(len(et_acc_cv)) if i % 5 == 2]
# et_2000 = [et_acc_cv[i] for i in range(len(et_acc_cv)) if i % 5 == 3]
# et_3000	= [et_acc_cv[i] for i in range(len(et_acc_cv)) if i % 5 == 4]


# et_100_bl 	= [et_bl_acc_cv[i] for i in range(len(et_bl_acc_cv)) if i % 5 == 0]
# et_500_bl 	= [et_bl_acc_cv[i] for i in range(len(et_bl_acc_cv)) if i % 5 == 1]
# et_1000_bl	= [et_bl_acc_cv[i] for i in range(len(et_bl_acc_cv)) if i % 5 == 2]
# et_2000_bl 	= [et_bl_acc_cv[i] for i in range(len(et_bl_acc_cv)) if i % 5 == 3]
# et_3000_bl 	= [et_bl_acc_cv[i] for i in range(len(et_bl_acc_cv)) if i % 5 == 4]


# et_max_depth = [3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 9, 9, 9,
#                    9, 9, 11, 11, 11, 11, 11, 13, 13, 13, 13, 13, 15, 15,
#                    15, 15, 15, 17, 17, 17, 17, 17]
# et_max_depth_true = [et_max_depth[i] for i in range(len(et_max_depth)) if i % 5 == 0]

# fig,ax = plt.subplots(figsize=(10,10))
# ax.locator_params(nbins=6, axis='y')
# ax.plot(et_max_depth_true, et_100, label='ET-100trees-acc')
# ax.plot(et_max_depth_true, et_500, label='ET-500trees-acc')
# ax.plot(et_max_depth_true, et_1000, label='ET-1000trees-acc')
# ax.plot(et_max_depth_true, et_2000, label='ET-2000trees-acc')
# ax.plot(et_max_depth_true, et_3000, label='ET-3000trees-acc')
# ax.set_xlabel("max_depth", fontsize=25)
# ax.set_ylabel("Accuracy", fontsize=25)
# ax.set_title("Accuracy", fontsize=25)
# ax.legend(fontsize=20)
# ax.tick_params(axis='both', labelsize=20)
# plt.savefig("/home/kei/newfig/et_train_acc.png", format='png', dpi=300)
# plt.close()

# fig,ax = plt.subplots(figsize=(10,10))
# ax.locator_params(nbins=7, axis='y')
# ax.plot(et_max_depth_true, et_100_bl, label='ET-100trees-bl-acc')
# ax.plot(et_max_depth_true, et_500_bl, label='ET-500trees-bl-acc')
# ax.plot(et_max_depth_true, et_1000_bl, label='ET-1000trees-bl-acc')
# ax.plot(et_max_depth_true, et_2000_bl, label='ET-2000trees-bl-acc')
# ax.plot(et_max_depth_true, et_3000_bl, label='ET-3000trees-bl-acc')
# ax.set_xlabel("max_depth", fontsize=25)
# ax.set_ylabel("Accuracy", fontsize=25)
# ax.set_title("Balanced accuracy", fontsize=25)
# ax.legend(fontsize=20)
# ax.tick_params(axis='both', labelsize=20)
# plt.savefig("/home/kei/newfig/et_train_bl_acc.png", format='png', dpi=300)
# plt.close()

##-------------------------------------------------------------------------
## Neural networks
##-------------------------------------------------------------------------
# nn1_acc_cv = [0.6717736 , 0.67181673, 0.67829644, 0.68537494, 0.63779186,
#        0.46787689, 0.67453047, 0.67508347, 0.67483243, 0.65154286,
#        0.53719691, 0.28516231]
# nn1_bl_acc_cv = [0.48634013, 0.48306666, 0.48688489, 0.45898206, 0.34143307,
#        0.11185041, 0.46485005, 0.4640804 , 0.44730041, 0.36756734,
#        0.21233244, 0.0625    ]
# nn2_acc_cv =  [0.66800709, 0.66569862, 0.66669964, 0.66590158, 0.66835799,
#        0.66554315, 0.67503396, 0.67719245, 0.62559441, 0.62434052,
#        0.28516231, 0.28516231, 0.66348828, 0.65997714, 0.66815577,
#        0.66268604, 0.67388049, 0.67282606, 0.6303133 , 0.63242088,
#        0.46335984, 0.4666253 , 0.28516231, 0.28516231]
# nn2_bl_acc_cv = [0.48691995, 0.49234217, 0.49478887, 0.48101444, 0.48845962,
#        0.49233204, 0.46050009, 0.4579801 , 0.3193496 , 0.3227834 ,
#        0.0625    , 0.0625    , 0.44227908, 0.44768601, 0.4408238 ,
#        0.44973265, 0.42002913, 0.42325582, 0.3328772 , 0.33184701,
#        0.10713268, 0.10808993, 0.0625    , 0.0625    ]

# nn_lambda_range = [0.001, 0.001, 0.01, 0.01, 0.1, 0.1, 1, 1, 10, 10, 100,
#                    100, 0.001, 0.001, 0.01, 0.01, 0.1, 0.1, 1, 1, 10, 10,
#                    100, 100]
# nn_lambda_range_true = [nn_lambda_range[i] for i in range(len(nn_lambda_range)) if i % 2 == 0 and i < len(nn_lambda_range)/2]

# nn1_relu_acc = [nn1_acc_cv[i] for i in range(len(nn1_acc_cv)) if i < len(nn1_acc_cv)/2]
# nn1_relu_bl_acc = [nn1_bl_acc_cv[i] for i in range(len(nn1_bl_acc_cv)) if i < len(nn1_bl_acc_cv)/2]
# nn1_log_acc = [nn1_acc_cv[i] for i in range(len(nn1_acc_cv)) if i >= len(nn1_acc_cv)/2]
# nn1_log_bl_acc = [nn1_bl_acc_cv[i] for i in range(len(nn1_bl_acc_cv)) if i >= len(nn1_bl_acc_cv)/2]

# nn2_relu_600_100_acc = [nn2_acc_cv[i] for i in range(len(nn2_acc_cv)) if i < len(nn2_acc_cv)/2 and i % 2 == 0]
# nn2_relu_600_100_bl_acc = [nn2_bl_acc_cv[i] for i in range(len(nn2_bl_acc_cv)) if i < len(nn2_bl_acc_cv)/2 and i % 2 == 0]
# nn2_log_600_100_acc = [nn2_acc_cv[i] for i in range(len(nn2_acc_cv)) if i >= len(nn2_acc_cv)/2 and i % 2 == 0]
# nn2_log_600_100_bl_acc = [nn2_bl_acc_cv[i] for i in range(len(nn2_bl_acc_cv)) if i >= len(nn2_bl_acc_cv)/2 and i % 2 == 0]
# nn2_relu_600_200_acc = [nn2_acc_cv[i] for i in range(len(nn2_acc_cv)) if i < len(nn2_acc_cv)/2 and i % 2 == 0]
# nn2_relu_600_200_bl_acc = [nn2_bl_acc_cv[i] for i in range(len(nn2_bl_acc_cv)) if i < len(nn2_bl_acc_cv)/2 and i % 2 != 0]
# nn2_log_600_200_acc = [nn2_acc_cv[i] for i in range(len(nn2_acc_cv)) if i < len(nn2_acc_cv)/2 and i % 2 == 0]
# nn2_log_600_200_bl_acc = [nn2_bl_acc_cv[i] for i in range(len(nn2_bl_acc_cv)) if i >= len(nn2_bl_acc_cv)/2 and i % 2 != 0]

# fig,ax = plt.subplots(figsize=(10,10))
# ax.locator_params(nbins=6, axis='y')
# ax.plot(nn_lambda_range_true, nn1_relu_acc, label='ReLU-acc')
# ax.plot(nn_lambda_range_true, nn1_log_acc, label='Logistic-acc')
# ax.plot(nn_lambda_range_true, nn1_relu_bl_acc, label='ReLU-acc')
# ax.plot(nn_lambda_range_true, nn1_log_bl_acc, label='Logistic-acc')
# ax.set_xlabel("lambda", fontsize=25)
# ax.set_ylabel("Accuracy", fontsize=25)
# ax.set_title("Neural Network 1 hidden layer n_nodes = 600", fontsize=25)
# ax.legend(fontsize=20)
# ax.tick_params(axis='both', labelsize=20)
# plt.savefig("/home/kei/newfig/nn1_train.png", format='png', dpi=300)
# plt.close()

# fig,ax = plt.subplots(figsize=(10,10))
# ax.locator_params(nbins=7, axis='y')
# ax.plot(nn_lambda_range_true, nn2_relu_600_100_acc, label='ReLU-600-100-acc')
# ax.plot(nn_lambda_range_true, nn2_log_600_100_acc, label='Logistic-600-100-acc')
# ax.plot(nn_lambda_range_true, nn2_relu_600_100_bl_acc, label='ReLU-600-100-bl-acc')
# ax.plot(nn_lambda_range_true, nn2_log_600_100_bl_acc, label='Logistic-600-100-bl-acc')
# ax.plot(nn_lambda_range_true, nn2_relu_600_200_acc, label='ReLU-600-200-acc')
# ax.plot(nn_lambda_range_true, nn2_log_600_200_acc, label='Logistic-600-200-acc')
# ax.plot(nn_lambda_range_true, nn2_relu_600_200_bl_acc, label='ReLU-600-200-bl-acc')
# ax.plot(nn_lambda_range_true, nn2_log_600_200_bl_acc, label='Logistic-600-200-bl-acc')
# ax.set_xlabel("lambda", fontsize=25)
# ax.set_ylabel("Accuracy", fontsize=25)
# ax.set_title("Neural network 2 hidden layers", fontsize=25)
# ax.legend(fontsize=18)
# ax.tick_params(axis='both', labelsize=20)
# plt.savefig("/home/kei/newfig/nn2_train.png", format='png', dpi=300)
# plt.close()

##--------------------------------------------------------------------------
## PCA
##__________________________________________________________________________

fig,ax = plt.subplots(figsize=(10,10))
ax.locator_params(nbins=5, axis='y')
ax.plot(dim, XGB_acc_true, label='XGB')
ax.plot(dim, SVMRBF_acc_true, label='SVMRBF')
ax.plot(dim, DT_acc_true, label='DT')
ax.plot(dim, NN1_acc_true, label='NN1')
ax.plot(dim, NN2_acc_true, label='NN2')
ax.plot(dim, LR_acc_true, label='LR')
ax.plot(dim, SVMLinear_acc_true, label='SVMLinear')
ax.plot(dim, ET_acc_true, label='ET')
ax.set_xlabel("n_dimensions", fontsize=25)
ax.set_ylabel("Accuracy ratio over non-PCA", fontsize=25)
ax.set_title("Accuracy", fontsize=25)
ax.legend(fontsize=16)
ax.tick_params(axis='both', labelsize=20)
plt.savefig("/home/kei/newfig/pca_accuracy.png", format='png', dpi=300)
plt.close()

fig,ax = plt.subplots(figsize=(10,10))
ax.locator_params(nbins=5, axis='y')
ax.plot(dim, XGB_bl_acc_true, label='XGB')
ax.plot(dim, SVMRBF_bl_acc_true, label='SVMRBF')
ax.plot(dim, DT_bl_acc_true, label='DT')
ax.plot(dim, NN1_bl_acc_true, label='NN1')
ax.plot(dim, NN2_bl_acc_true, label='NN2')
ax.plot(dim, LR_bl_acc_true, label='LR')
ax.plot(dim, SVMLinear_bl_acc_true, label='SVMLinear')
ax.plot(dim, ET_bl_acc_true, label='ET')
ax.set_xlabel("n_dimensions",fontsize=25)
ax.set_ylabel("Balanced accuracy ratio over non-PCA", fontsize=25)
ax.set_title("Balanced accuracy", fontsize=25)
ax.legend(fontsize=16)
ax.tick_params(axis='both', labelsize=20)
plt.savefig("/home/kei/newfig/pca_bl_accuracy.png", format='png', dpi=300)
plt.close()




##-------------------------------------------------------------------
## SMOTE-----------------------------------------------------------
##___________________________________________________________________

# ind = np.arange(len(original_result_acc.keys()))
# width = 0.2

# plt.figure(figsize=(10,10))
# fig,ax=plt.subplots()
# ax.locator_params(nbins=4, axis='y')
# p1 = ax.bar(ind, original_acc, width, color='r')
# p2 = ax.bar(ind + width, smote_acc, width,
#             color='y')
# ax.set_xticklabels((0, 'XGB', 'SVM RBF', 'DT', 'NN1', 'NN2', 'LR', 'SVMLinear', 'ET'), fontsize=11)
# ax.legend((p1[0], p2[0]), ('Non-SMOTE', 'With SMOTE'), fontsize=15)
# ax.tick_params(axis='y', labelsize=14)
# ax.set_ylabel("Accuracy", fontsize=15)

# ax.set_title("Accuracy with SMOTE", fontsize=15)
# plt.savefig("/home/kei/newfig/smote_accuracy.png", format='png', dpi=300)
# plt.close()



# plt.figure(figsize=(10,10))
# fig,ax=plt.subplots()
# ax.locator_params(nbins=4, axis='y')
# p1 = ax.bar(ind, original_bl_acc, width, color='r')
# p2 = ax.bar(ind + width, smote_bl_acc, width,
#             color='y')
# ax.set_xticklabels((0, 'XGB', 'SVM RBF', 'DT', 'NN1', 'NN2', 'LR', 'SVMLinear', 'ET'), fontsize=11)
# ax.legend((p1[0], p2[0]), ('Non-SMOTE', 'With SMOTE'), fontsize=15)
# ax.tick_params(axis='y', labelsize=14)
# ax.set_ylabel("Balanced Accuracy", fontsize=15)

# ax.set_title("Balanced Accuracy with SMOTE", fontsize=15)
# plt.savefig("/home/kei/newfig/smote_bl_accuracy.png", format='png', dpi=300)
# plt.close()

