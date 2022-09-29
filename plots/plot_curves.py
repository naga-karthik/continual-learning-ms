"""
Plots the average zero-shot test dice curves for single-domain, multi-domain, fine-tuning and 
replay experiments for 9 random sequences of domains/centers. 
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.metrics import ACC, BWT, FWT, LA


orig_centers_list = ['bwh', 'karo', 'milan', 'rennes', 'nih', 'montp', 'ucsf', 'amu']

# =============================================================================================
# RESULTS FROM RANDOMIZED ORDERING OF THE DOMAINS
# ==============================================================================================


# ===================================== seed=12345 ====================================================
order_seed_12345 = ['amu', 'rennes', 'nih', 'bwh', 'karo', 'ucsf', 'montp', 'milan']

# this list corresponds to the results from the same seed, with matched Dice scores as in the order_seed above 
test_dice_mat_ST_12345 = [0.40749279, 0.44507822, 0.60181814, 0.48095065, 0.4199203, 0.6058743, 0.35930485, 0.4582386]
mean_test_dice_mat_ST_12345 = np.mean(test_dice_mat_ST_12345)


test_dice_mat_ER_12345 = np.array([[0.40749279, 0.24478376, 0.25934839, 0.22895442, 0.14841747,
        0.55226886, 0.31140625, 0.05161838],
       [0.53726643, 0.46214876, 0.44080925, 0.46426174, 0.2177645 ,
        0.6646024 , 0.55663633, 0.0814062 ],
       [0.53325701, 0.47363192, 0.57894254, 0.49480516, 0.21711293,
        0.67241496, 0.561589  , 0.09679699],
       [0.50331247, 0.35050789, 0.57770842, 0.43416566, 0.20099294,
        0.4676674 , 0.29524064, 0.05530363],
       [0.57103115, 0.50543755, 0.60070753, 0.50061989, 0.44893995,
        0.69147015, 0.55527735, 0.34202436],
       [0.58407593, 0.51702857, 0.63845575, 0.53861076, 0.38111028,
        0.70653427, 0.5968591 , 0.33592597],
       [0.57335991, 0.52235031, 0.63466197, 0.54758346, 0.36877364,
        0.70121807, 0.6114819 , 0.25117251],
       [0.58936274, 0.54158092, 0.65697622, 0.55335999, 0.45012784,
        0.71830654, 0.62067091, 0.46067989]])

test_dice_mat_FT_12345 = np.array([[0.40749279, 0.24478376, 0.25934839, 0.22895442, 0.14841747,
        0.55226886, 0.31140625, 0.05161838],
       [0.52402383, 0.41131139, 0.40473041, 0.43502361, 0.21619976,
        0.62605387, 0.55446225, 0.08257789],
       [0.53003901, 0.43172473, 0.51451433, 0.48079741, 0.20417833,
        0.66196632, 0.48524773, 0.0644462 ],
       [0.52596325, 0.40868482, 0.59569466, 0.45453262, 0.2080812 ,
        0.63442194, 0.4299973 , 0.07738094],
       [0.55063498, 0.3982214 , 0.46897644, 0.45329878, 0.45010415,
        0.67882055, 0.43223673, 0.36957729],
       [0.50966144, 0.38323873, 0.57703364, 0.46616802, 0.22831419,
        0.68883872, 0.48286071, 0.13940896],
       [0.53627455, 0.40307266, 0.43436521, 0.45875335, 0.24142607,
        0.65055412, 0.5097335 , 0.0994444 ],
       [0.41427517, 0.2764256 , 0.37151352, 0.26330245, 0.34069309,
        0.45170465, 0.29742315, 0.41448241]])

er_mean_12345, ft_mean_12345 = [], []
for i, cname in enumerate(order_seed_12345):
        # Replay
        er_mean_12345.append(np.mean(test_dice_mat_ER_12345[i, :]))     # zero-shot
        ft_mean_12345.append(np.mean(test_dice_mat_FT_12345[i, :]))

# so this particular seed results in some ordering of the centers, so we're comparing the results
# with the same seed in multi-domain learning. The test results are according to the original centers_list 
# ordering at the top
test_dice_mat_MT_12345 = np.array([[0.52841103, 0.43672198, 0.38362455, 0.52646005, 0.65321267,
        0.57288122, 0.71050936, 0.5696007 ]])
# creating a dict for the orig_centers_list and the corresp. Dice scores
orig_results_dict = {}
for i, c_name in enumerate(orig_centers_list):
        orig_results_dict[c_name] = test_dice_mat_MT_12345[0][i]


result_matrix_mt_12345 = []
for c_name in order_seed_12345:
        result_matrix_mt_12345.append(orig_results_dict[c_name])
mean_result_matrix_mt_12345 = np.mean(result_matrix_mt_12345)

df_seed_12345 = pd.DataFrame({
        'centers': order_seed_12345,
        'er_mean': er_mean_12345,
        'ft_mean': ft_mean_12345,
        'mean_mt': mean_result_matrix_mt_12345,
        'st': mean_test_dice_mat_ST_12345
})


# ===================================== seed=4242 ====================================================
order_seed_4242=['nih', 'bwh', 'rennes', 'milan', 'karo', 'montp', 'amu', 'ucsf']
test_dice_mat_ST_4242 = [0.49975699, 0.39181086, 0.4505806, 0.34543383, 0.33184218, 0.3458195, 0.33687487, 0.38611567]
mean_test_dice_mat_ST_4242 = np.mean(test_dice_mat_ST_4242)

test_dice_mat_ER_4242 = np.array([[0.49975699, 0.37558928, 0.40805349, 0.02819232, 0.18150236,
        0.34477609, 0.49678996, 0.45878023],
       [0.54471946, 0.44246352, 0.47534394, 0.06850772, 0.2090226 ,
        0.41364011, 0.53286284, 0.49690852],
       [0.5492239 , 0.44571656, 0.51743853, 0.07095644, 0.20098622,
        0.44325548, 0.46812567, 0.57045126],
       [0.5700888 , 0.46202797, 0.51292789, 0.41328454, 0.45869765,
        0.44540942, 0.5403735 , 0.55737728],
       [0.56705797, 0.46846247, 0.51935518, 0.42277789, 0.5011785 ,
        0.44828856, 0.49530041, 0.57759058],
       [0.53813642, 0.46965224, 0.50195509, 0.36310819, 0.46069223,
        0.43918049, 0.47578454, 0.54940081],
       [0.55367476, 0.45795581, 0.51997375, 0.36689276, 0.4826223 ,
        0.45122856, 0.44843578, 0.54849899],
       [0.57390374, 0.49636412, 0.54681277, 0.40885392, 0.49834949,
        0.48521447, 0.47110167, 0.59769058]])

test_dice_mat_FT_4242 = np.array([[0.49975699, 0.37558928, 0.40805349, 0.02819232, 0.18150236,
        0.34477609, 0.49678996, 0.45878023],
       [0.53890568, 0.4239279 , 0.49718016, 0.05674854, 0.20585719,
        0.40312114, 0.4943881 , 0.53133607],
       [0.49780425, 0.39803991, 0.48418808, 0.08431378, 0.21734181,
        0.41524875, 0.50307679, 0.48874047],
       [0.24559557, 0.16726434, 0.24576238, 0.39744335, 0.35790238,
        0.26863137, 0.18078546, 0.23649251],
       [0.41678652, 0.37504122, 0.44977537, 0.3012417 , 0.40414792,
        0.35761929, 0.44157016, 0.40889221],
       [0.41333336, 0.4036395 , 0.48345485, 0.1547427 , 0.28272933,
        0.44879398, 0.44813868, 0.45785022],
       [0.37406269, 0.36077291, 0.4659943 , 0.10825386, 0.21420082,
        0.39694154, 0.44560206, 0.48797822],
       [0.35907283, 0.35873348, 0.47547212, 0.11172553, 0.19181043,
        0.38530207, 0.40016985, 0.50385427]])

test_dice_mat_MT_4242 = np.array([[0.48004219, 0.51444221, 0.44766808, 0.5396663 , 0.59102935,
        0.47844082, 0.59535229, 0.51113957]])
orig_results_dict = {}
for i, c_name in enumerate(orig_centers_list):
        orig_results_dict[c_name] = test_dice_mat_MT_4242[0][i]


er_mean_4242, ft_mean_4242 = [], []
for i, cname in enumerate(order_seed_4242):
        # Replay
        er_mean_4242.append(np.mean(test_dice_mat_ER_4242[i, :]))     # zero-shot
        ft_mean_4242.append(np.mean(test_dice_mat_FT_4242[i, :]))

result_matrix_mt_4242 = []
for c_name in order_seed_4242:
        result_matrix_mt_4242.append(orig_results_dict[c_name])
mean_result_matrix_mt_4242 = np.mean(result_matrix_mt_4242)

df_seed_4242 = pd.DataFrame({
        'centers': order_seed_4242,
        'er_mean': er_mean_4242,
        'ft_mean': ft_mean_4242,
        'mean_mt': mean_result_matrix_mt_4242,
        'st': mean_test_dice_mat_ST_4242
})


# ======================================= seed=728 ====================================================
order_seed_728 = ['rennes', 'ucsf', 'karo', 'amu', 'nih', 'bwh', 'milan', 'montp']

test_dice_mat_ST_728 = [0.49696088, 0.65557146, 0.32869187, 0.29842463, 0.56544751, 0.52238566, 0.44500774, 0.34789175]
mean_test_dice_mat_ST_728 = np.mean(test_dice_mat_ST_728)

test_dice_mat_ER_728 = np.array([[0.49696088, 0.67433977, 0.22874185, 0.60843265, 0.53505522,
        0.49731594, 0.19869635, 0.40150484],
       [0.50854951, 0.69497538, 0.24454352, 0.60733777, 0.54329979,
        0.52258456, 0.24114121, 0.39457619],
       [0.55030429, 0.68344605, 0.4153707 , 0.59366304, 0.54623502,
        0.54300559, 0.41393998, 0.41172409],
       [0.5414381 , 0.70575786, 0.38848388, 0.602503  , 0.56182599,
        0.55932981, 0.38624933, 0.40682709],
       [0.5156492 , 0.69468683, 0.38335994, 0.57166076, 0.54664528,
        0.52499712, 0.37460661, 0.37266874],
       [0.54030079, 0.70051789, 0.3973158 , 0.59334147, 0.57130259,
        0.56076628, 0.36756626, 0.4029105 ],
       [0.55946988, 0.7174409 , 0.46910053, 0.60310221, 0.58635437,
        0.58923817, 0.47558874, 0.40171385],
       [0.41505441, 0.57991338, 0.36473379, 0.52346814, 0.52376902,
        0.4434371 , 0.36539638, 0.29012322]])

test_dice_mat_FT_728 = np.array([[0.49696088, 0.67433977, 0.22874185, 0.60843265, 0.53505522,
        0.49731594, 0.19869635, 0.40150484],
       [0.38395229, 0.69237566, 0.229544  , 0.53815514, 0.50234467,
        0.45620084, 0.12348589, 0.3109265 ],
       [0.45818749, 0.67066032, 0.3954469 , 0.58355737, 0.52950478,
        0.50467014, 0.33689368, 0.34651458],
       [0.33353692, 0.64511514, 0.20984702, 0.56148714, 0.43319145,
        0.36413783, 0.14064161, 0.30913925],
       [0.49677581, 0.72289449, 0.22396429, 0.61320662, 0.59006441,
        0.53113651, 0.1880632 , 0.35392836],
       [0.42540693, 0.64317024, 0.18246143, 0.54943514, 0.5294469 ,
        0.47155064, 0.14565903, 0.28765789],
       [0.33102983, 0.58576059, 0.39059007, 0.41293579, 0.40392739,
        0.36287469, 0.46467245, 0.21289477],
       [0.49499288, 0.69028926, 0.28566736, 0.5868144 , 0.54825455,
        0.51239264, 0.29741824, 0.39392507]])


er_mean_728, ft_mean_728 = [], []
for i, cname in enumerate(order_seed_728):
        # Replay
        er_mean_728.append(np.mean(test_dice_mat_ER_728[i, :]))       # zero-shot
        ft_mean_728.append(np.mean(test_dice_mat_FT_728[i, :]))

test_dice_mat_MT_728 = np.array([[0.33807296, 0.29122791, 0.25476682, 0.32370684, 0.43677655,
        0.18528283, 0.48663735, 0.38923842]])
orig_results_dict = {}
for i, c_name in enumerate(orig_centers_list):
        orig_results_dict[c_name] = test_dice_mat_MT_728[0][i]

result_matrix_mt_728 = []
for c_name in order_seed_728:
        result_matrix_mt_728.append(orig_results_dict[c_name])
mean_result_matrix_mt_728 = np.mean(result_matrix_mt_728)

df_seed_728 = pd.DataFrame({
        'centers': order_seed_728,
        'er_mean': er_mean_728,
        'ft_mean': ft_mean_728,
        'mean_mt': mean_result_matrix_mt_728,
        'st': mean_test_dice_mat_ST_728
})


# ========================================== seed=171 ====================================================
order_seed_171 = ['rennes', 'ucsf', 'milan', 'nih', 'karo', 'amu', 'bwh', 'montp']
test_dice_mat_ST_171 = [0.50161517, 0.59949881, 0.38169658, 0.48619661, 0.390351, 0.50565201, 0.42488089, 0.17096673]
mean_test_dice_mat_ST_171 = np.mean(test_dice_mat_ST_171)

test_dice_mat_ER_171 = np.array([[0.50161517, 0.45412308, 0.0749867 , 0.44946963, 0.17010531,
        0.59654671, 0.37983009, 0.30508432],
       [0.49723163, 0.6420809 , 0.14889707, 0.43073049, 0.19023828,
        0.61221737, 0.36899772, 0.28562319],
       [0.49005279, 0.65056813, 0.41225386, 0.49363011, 0.44568712,
        0.57785332, 0.44026163, 0.3591215 ],
       [0.47455031, 0.6243968 , 0.3949869 , 0.49775821, 0.42588186,
        0.54940563, 0.43145686, 0.31796539],
       [0.53583544, 0.6926229 , 0.43638977, 0.52210104, 0.50121588,
        0.62353146, 0.49016559, 0.3949796 ],
       [0.50747991, 0.68878597, 0.41720936, 0.5152148 , 0.46072873,
        0.61376846, 0.47102356, 0.36415452],
       [0.53796417, 0.70637393, 0.43995211, 0.55725247, 0.48957053,
        0.62747061, 0.49140686, 0.38329363],
       [0.53225392, 0.67374843, 0.4584457 , 0.5501641 , 0.51078445,
        0.59027719, 0.49807149, 0.38687444]])

test_dice_mat_FT_171 = np.array([[0.50161517, 0.45412308, 0.0749867 , 0.44946963, 0.17010531,
        0.59654671, 0.37983009, 0.30508432],
       [0.37043539, 0.50670618, 0.10105902, 0.36842471, 0.14420761,
        0.52523637, 0.23864216, 0.21971834],
       [0.25482106, 0.14416799, 0.41079473, 0.19890657, 0.36628276,
        0.30911756, 0.16029944, 0.18407536],
       [0.49254084, 0.65412033, 0.15500337, 0.5073148 , 0.22135127,
        0.60272849, 0.40372679, 0.25863442],
       [0.51967537, 0.51542765, 0.34559819, 0.46206349, 0.45834318,
        0.63608021, 0.36906543, 0.31999582],
       [0.44089308, 0.37066343, 0.1306282 , 0.40850624, 0.19721922,
        0.63532668, 0.30091774, 0.21697292],
       [0.50057805, 0.57773805, 0.12970531, 0.53866714, 0.22240119,
        0.59964144, 0.43180445, 0.33947355],
       [0.5006212 , 0.35547292, 0.11789677, 0.40255195, 0.19747135,
        0.61709201, 0.38858479, 0.29723474]])


test_dice_mat_MT_171 = np.array([[0.33146235, 0.36943835, 0.2916095 , 0.32748306, 0.44445688,
        0.22040357, 0.46795386, 0.45112917]])
orig_results_dict = {}
for i, c_name in enumerate(orig_centers_list):
        orig_results_dict[c_name] = test_dice_mat_MT_171[0][i]

er_mean_171, ft_mean_171 = [], []
for i, cname in enumerate(order_seed_171):
        # Replay
        er_mean_171.append(np.mean(test_dice_mat_ER_171[i, :]))       # zero-shot
        ft_mean_171.append(np.mean(test_dice_mat_FT_171[i, :]))

result_matrix_mt_171 = []
for c_name in order_seed_171:
        result_matrix_mt_171.append(orig_results_dict[c_name])
mean_result_matrix_mt_171 = np.mean(result_matrix_mt_171)

df_seed_171 = pd.DataFrame({
        'centers': order_seed_171,
        'er_mean': er_mean_171,
        'ft_mean': ft_mean_171,
        'mean_mt': mean_result_matrix_mt_171,
        'st': mean_test_dice_mat_ST_171
})


# ===================================== seed=1001 ====================================================
order_seed_1001 = ['ucsf', 'milan', 'nih', 'bwh', 'rennes', 'amu', 'karo', 'montp']
test_dice_mat_ST_1001 = [0.5886597, 0.48526689, 0.54775387, 0.46772367, 0.49139252, 0.52047902, 0.51254767, 0.4282237]
mean_test_dice_mat_ST_1001 = np.mean(test_dice_mat_ST_1001)

test_dice_mat_ER_1001 = np.array([[0.5886597 , 0.04714346, 0.30000797, 0.28087565, 0.27565837,
        0.57882446, 0.31244665, 0.23222463],
       [0.65973765, 0.44278491, 0.49763161, 0.45297298, 0.47221836,
        0.58262205, 0.46246308, 0.4238165 ],
       [0.61143064, 0.47217214, 0.55252039, 0.50233036, 0.50741416,
        0.66847539, 0.49560255, 0.56386924],
       [0.62544018, 0.41766107, 0.50894558, 0.4599658 , 0.47735253,
        0.62201267, 0.47955376, 0.56677198],
       [0.60005528, 0.4665941 , 0.58579916, 0.54704058, 0.55278963,
        0.69626474, 0.51216871, 0.66087604],
       [0.60408294, 0.52142322, 0.56920832, 0.53005975, 0.53796691,
        0.68930805, 0.51885593, 0.64441967],
       [0.61798513, 0.46774918, 0.56197351, 0.52587283, 0.54698694,
        0.68323439, 0.53277338, 0.61582142],
       [0.62762278, 0.50165474, 0.56083941, 0.52282751, 0.5398525 ,
        0.68282306, 0.53734291, 0.61289632]])

test_dice_mat_FT_1001 = np.array([[0.5886597 , 0.04714346, 0.30000797, 0.28087565, 0.27565837,
        0.57882446, 0.31244665, 0.23222463],
       [0.34661055, 0.48061344, 0.3078483 , 0.16115128, 0.31260914,
        0.26990238, 0.20537677, 0.2932561 ],
       [0.5508585 , 0.11527554, 0.52929032, 0.44845989, 0.46289903,
        0.67751092, 0.34007517, 0.53451085],
       [0.60023546, 0.09994295, 0.4281742 , 0.4115302 , 0.37443784,
        0.53960973, 0.33134848, 0.39234415],
       [0.56896114, 0.15982813, 0.54672742, 0.48967603, 0.51455224,
        0.63664871, 0.34636205, 0.61673737],
       [0.4949179 , 0.12728246, 0.48758626, 0.4525997 , 0.45744085,
        0.66139096, 0.34889787, 0.53822207],
       [0.59301168, 0.42728126, 0.4719153 , 0.43661427, 0.47241855,
        0.61180931, 0.51014405, 0.44868663],
       [0.57007003, 0.26523364, 0.53231084, 0.48677325, 0.49663931,
        0.64728075, 0.383663  , 0.56428611]])

test_dice_mat_MT_1001 = np.array([[0.53303254, 0.53004038, 0.50378287, 0.5672698 , 0.58692443,
        0.64642441, 0.61524934, 0.68181348]])
orig_results_dict = {}
for i, c_name in enumerate(orig_centers_list):
        orig_results_dict[c_name] = test_dice_mat_MT_1001[0][i]

er_mean_1001, ft_mean_1001 = [], []
for i, cname in enumerate(order_seed_1001):
        # Replay
        er_mean_1001.append(np.mean(test_dice_mat_ER_1001[i, :]))     # zero-shot
        ft_mean_1001.append(np.mean(test_dice_mat_FT_1001[i, :]))

result_matrix_mt_1001 = []
for c_name in order_seed_1001:
        result_matrix_mt_1001.append(orig_results_dict[c_name])
mean_result_matrix_mt_1001 = np.mean(result_matrix_mt_1001)

df_seed_1001 = pd.DataFrame({
        'centers': order_seed_1001,
        'er_mean': er_mean_1001,
        'ft_mean': ft_mean_1001,
        'mean_mt': mean_result_matrix_mt_1001,
        'st': mean_test_dice_mat_ST_1001
})

# ===================================== seed=1113 ====================================================
order_seed_1113 = ['bwh', 'karo', 'milan', 'rennes', 'nih', 'montp', 'ucsf', 'amu']

test_dice_mat_ST_1113 = [0.51297706, 0.4361093, 0.41108167, 0.41600651, 0.39946398, 0.33657119, 0.53531277, 0.2709859]
mean_test_dice_mat_ST_1113 = np.mean(test_dice_mat_ST_1113)

test_dice_mat_ER_1113 = np.array([[0.51297706, 0.26685527, 0.08625709, 0.44549602, 0.45582899,
        0.59744847, 0.57401401, 0.53103405],
       [0.4863365 , 0.40741029, 0.23838918, 0.44176999, 0.41368398,
        0.44966707, 0.6149357 , 0.51114464],
       [0.38553387, 0.29135266, 0.28129154, 0.28542322, 0.30561295,
        0.3261632 , 0.22960544, 0.40986434],
       [0.4630312 , 0.40352321, 0.33932751, 0.38618773, 0.34971064,
        0.40217564, 0.40554255, 0.46648961],
       [0.52203095, 0.44071823, 0.37582144, 0.47437057, 0.47247949,
        0.57943749, 0.6204741 , 0.53452069],
       [0.53234816, 0.47852048, 0.44450119, 0.47610742, 0.46001846,
        0.58179975, 0.66331303, 0.52674925],
       [0.54426509, 0.50015897, 0.46781603, 0.49274334, 0.48065177,
        0.65047061, 0.70088935, 0.52483672],
       [0.54149747, 0.47615251, 0.46145031, 0.48379096, 0.48686266,
        0.62322134, 0.68422621, 0.54971272]])

test_dice_mat_FT_1113 = np.array([[0.51297706, 0.26685527, 0.08625709, 0.44549602, 0.45582899,
        0.59744847, 0.57401401, 0.53103405],
       [0.40664658, 0.38521391, 0.27254602, 0.38666195, 0.3541227 ,
        0.36355233, 0.59734094, 0.35272798],
       [0.33178073, 0.37997109, 0.42380202, 0.31118804, 0.28867936,
        0.27914459, 0.37216514, 0.3152006 ],
       [0.48506907, 0.32491726, 0.18477099, 0.45327181, 0.40409985,
        0.56012583, 0.63728404, 0.47637114],
       [0.50238824, 0.28992382, 0.16642278, 0.45516577, 0.44874033,
        0.5586195 , 0.62114543, 0.55651361],
       [0.44703388, 0.30355576, 0.14511847, 0.42962855, 0.36299774,
        0.59997481, 0.61499512, 0.44158155],
       [0.49274552, 0.29664692, 0.12218503, 0.47279924, 0.42387429,
        0.51138282, 0.66806066, 0.49741668],
       [0.40471953, 0.31237099, 0.13947514, 0.41819936, 0.3122533 ,
        0.53688085, 0.63966143, 0.45545173]])

test_dice_mat_MT_1113 = np.array([[0.53569806, 0.44700074, 0.38185489, 0.45782089, 0.46038446,
        0.59401476, 0.59165758, 0.5237965 ]])
orig_results_dict = {}
for i, c_name in enumerate(orig_centers_list):
        orig_results_dict[c_name] = test_dice_mat_MT_1113[0][i]


er_mean_1113, ft_mean_1113 = [], []
for i, cname in enumerate(order_seed_1113):
        er_mean_1113.append(np.mean(test_dice_mat_ER_1113[i, :]))     # zero-shot
        ft_mean_1113.append(np.mean(test_dice_mat_FT_1113[i, :]))

result_matrix_mt_1113 = []
for c_name in order_seed_1113:
        result_matrix_mt_1113.append(orig_results_dict[c_name])
mean_result_matrix_mt_1113 = np.mean(result_matrix_mt_1113)

df_seed_1113 = pd.DataFrame({
        'centers': order_seed_1113,
        'er_mean': er_mean_1113,
        'ft_mean': ft_mean_1113,
        'mean_mt': mean_result_matrix_mt_1113,
        'st': mean_test_dice_mat_ST_1113
})

# ===================================== seed=46 ====================================================
order_seed_46 = ['ucsf', 'karo', 'rennes', 'milan', 'nih', 'bwh', 'amu', 'montp']
test_dice_mat_ST_46 = [0.45105395, 0.46470708, 0.53442299, 0.37549096, 0.56128269, 0.41356057, 0.26929602, 0.41045848]
mean_test_dice_mat_ST_46 = np.mean(test_dice_mat_ST_46)

test_dice_mat_ER_46 = np.array([[0.45105395, 0.15175664, 0.25003836, 0.04752313, 0.25328198,
        0.24665123, 0.10741811, 0.16245584],
       [0.59353226, 0.49473801, 0.53748298, 0.33042967, 0.55226284,
        0.40687653, 0.48554263, 0.45405948],
       [0.53073466, 0.49536508, 0.54593003, 0.3500109 , 0.57735413,
        0.4207623 , 0.3926242 , 0.46569681],
       [0.57528573, 0.50788963, 0.58287573, 0.42164633, 0.57033193,
        0.44608703, 0.44918901, 0.52931976],
       [0.52638191, 0.49961644, 0.53576928, 0.37480536, 0.59754133,
        0.43370569, 0.42704728, 0.49237132],
       [0.58239287, 0.5493536 , 0.62543833, 0.44394371, 0.59530067,
        0.46784276, 0.40998042, 0.56319618],
       [0.58872801, 0.5670172 , 0.6347425 , 0.44935465, 0.61294311,
        0.49473065, 0.43485916, 0.59416854],
       [0.57474101, 0.55129349, 0.62716681, 0.45432413, 0.60506701,
        0.49669707, 0.46183074, 0.55632377]])

test_dice_mat_FT_46 = np.array([[0.45105395, 0.15175664, 0.25003836, 0.04752313, 0.25328198,
        0.24665123, 0.10741811, 0.16245584],
       [0.47327214, 0.48430914, 0.5016104 , 0.33696467, 0.44837713,
        0.3745693 , 0.43051875, 0.43583989],
       [0.49986747, 0.3465755 , 0.54831213, 0.19162196, 0.55787486,
        0.42001981, 0.43299261, 0.50429291],
       [0.43694893, 0.41773123, 0.4189952 , 0.40550962, 0.37122053,
        0.301539  , 0.29081634, 0.34803206],
       [0.52599978, 0.29650253, 0.52274543, 0.14269513, 0.59952366,
        0.42497855, 0.39235705, 0.46983379],
       [0.51870847, 0.27045906, 0.50548041, 0.11440004, 0.5817607 ,
        0.4208051 , 0.35756075, 0.48716283],
       [0.56068999, 0.24816772, 0.48415524, 0.10499809, 0.49773455,
        0.4039976 , 0.39461055, 0.44332492],
       [0.51710314, 0.3519583 , 0.55723894, 0.21891886, 0.5148086 ,
        0.41825882, 0.37338316, 0.51435554]])

test_dice_mat_MT_46 = np.array([[0.44389883, 0.5462774 , 0.42138755, 0.5861932 , 0.59759563,
        0.53399128, 0.53330457, 0.45755887]])
orig_results_dict = {}
for i, c_name in enumerate(orig_centers_list):
        orig_results_dict[c_name] = test_dice_mat_MT_46[0][i]


er_mean_46, ft_mean_46 = [], []
for i, cname in enumerate(order_seed_1113):
        er_mean_46.append(np.mean(test_dice_mat_ER_46[i, :]))     # zero-shot
        ft_mean_46.append(np.mean(test_dice_mat_FT_46[i, :]))

result_matrix_mt_46 = []
for c_name in order_seed_46:
        result_matrix_mt_46.append(orig_results_dict[c_name])
mean_result_matrix_mt_46 = np.mean(result_matrix_mt_46)


df_seed_46 = pd.DataFrame({
        'centers': order_seed_46,
        'er_mean': er_mean_46,
        'ft_mean': ft_mean_46,
        'mean_mt': mean_result_matrix_mt_46,
        'st': mean_test_dice_mat_ST_46
})

# ===================================== seed=5 ====================================================
order_seed_5 = ['amu', 'milan', 'nih', 'karo', 'bwh', 'montp', 'ucsf', 'rennes']
test_dice_mat_ST_5 = [0.38928571, 0.40600419, 0.41878882, 0.43910703, 0.59241498, 0.36094385, 0.40061125, 0.52399081]
mean_test_dice_mat_ST_5 = np.mean(test_dice_mat_ST_5)

test_dice_mat_ER_5 = np.array([[0.38928571, 0.03781973, 0.25100794, 0.17762837, 0.28571066,
        0.29962716, 0.40067193, 0.35802603],
       [0.35564989, 0.40269578, 0.36491385, 0.40302354, 0.46420643,
        0.44386822, 0.39723027, 0.4824664 ],
       [0.3898578 , 0.37471223, 0.4659999 , 0.39454895, 0.60233015,
        0.48975217, 0.60434222, 0.51046032],
       [0.40258959, 0.40534845, 0.47957721, 0.45011869, 0.614236  ,
        0.50581402, 0.6084671 , 0.52669024],
       [0.42845395, 0.36363202, 0.4649877 , 0.41985297, 0.61648685,
        0.51212907, 0.66475314, 0.53937048],
       [0.44195849, 0.41188776, 0.49745184, 0.45917568, 0.62814045,
        0.60527778, 0.56375116, 0.57936603],
       [0.35666889, 0.3537637 , 0.45911932, 0.42667666, 0.62410128,
        0.48544616, 0.69662821, 0.51630366],
       [0.4428829 , 0.41517881, 0.48793393, 0.45270127, 0.63555628,
        0.53648686, 0.66274387, 0.57033223]])

test_dice_mat_FT_5 = np.array([[0.38928571, 0.03781973, 0.25100794, 0.17762837, 0.28571066,
        0.29962716, 0.40067193, 0.35802603],
       [0.33736911, 0.38628158, 0.22867294, 0.30260414, 0.30700591,
        0.22654402, 0.31729046, 0.39176482],
       [0.37022901, 0.09446049, 0.42042351, 0.23460555, 0.58150882,
        0.38659018, 0.72596335, 0.48148403],
       [0.40597701, 0.38043573, 0.41041404, 0.44464865, 0.6021744 ,
        0.38945732, 0.49941158, 0.51203424],
       [0.30858573, 0.08894012, 0.39599264, 0.23991784, 0.59452313,
        0.35366178, 0.66285706, 0.45879859],
       [0.371914  , 0.11353026, 0.45915788, 0.26080772, 0.6029368 ,
        0.45259875, 0.61416131, 0.49313012],
       [0.42296368, 0.15662244, 0.4492282 , 0.29022342, 0.56228745,
        0.4435221 , 0.49843177, 0.50768876],
       [0.5058437 , 0.13531137, 0.45610166, 0.26795   , 0.60389131,
        0.55292326, 0.58784258, 0.56736749]])

test_dice_mat_MT_5 = np.array([[0.60051572, 0.43127665, 0.32883945, 0.55428827, 0.46041718,
        0.50451058, 0.60345948, 0.42079043]])
orig_results_dict = {}
for i, c_name in enumerate(orig_centers_list):
        orig_results_dict[c_name] = test_dice_mat_MT_5[0][i]


er_mean_5, ft_mean_5 = [], []
for i, cname in enumerate(order_seed_5):
        er_mean_5.append(np.mean(test_dice_mat_ER_5[i, :]))     # zero-shot
        ft_mean_5.append(np.mean(test_dice_mat_FT_5[i, :]))

result_matrix_mt_5 = []
for c_name in order_seed_5:
        result_matrix_mt_5.append(orig_results_dict[c_name])
mean_result_matrix_mt_5 = np.mean(result_matrix_mt_5)


df_seed_5 = pd.DataFrame({
        'centers': order_seed_5,
        'er_mean': er_mean_5,
        'ft_mean': ft_mean_5,
        'mean_mt': mean_result_matrix_mt_5,
        'st': mean_test_dice_mat_ST_5
})


# ===================================== seed=9 ====================================================
order_seed_9 = ['amu', 'karo', 'milan', 'rennes', 'bwh', 'montp', 'nih', 'ucsf']
test_dice_mat_ST_9 = [0.0873301, 0.40943128, 0.4516362, 0.44996828, 0.4482508, 0.36161029, 0.52565831, 0.58627665]
mean_test_dice_mat_ST_9 = np.mean(test_dice_mat_ST_9)

test_dice_mat_ER_9 = np.array([[0.0873301 , 0.05042462, 0.04002318, 0.08752342, 0.05310177,
        0.12751321, 0.03692374, 0.03210329],
       [0.53440464, 0.40549421, 0.35911697, 0.46467581, 0.42700589,
        0.50774539, 0.4339059 , 0.48477304],
       [0.55333102, 0.43811163, 0.50289541, 0.46148983, 0.42933965,
        0.51605606, 0.4545168 , 0.56812751],
       [0.56891251, 0.46139961, 0.49134004, 0.49311289, 0.47499219,
        0.56691456, 0.48167056, 0.54319286],
       [0.57943505, 0.45504016, 0.50875658, 0.4991568 , 0.53217101,
        0.56713849, 0.55385751, 0.67841411],
       [0.56969517, 0.44355512, 0.48239768, 0.47403282, 0.50390953,
        0.52107859, 0.55094564, 0.67545867],
       [0.57680494, 0.43864316, 0.50953686, 0.49128279, 0.51432872,
        0.54462862, 0.55616027, 0.64064002],
       [0.60164762, 0.44090429, 0.50470042, 0.49394932, 0.54709041,
        0.54820812, 0.56837332, 0.66762257]])

test_dice_mat_FT_9 = np.array([[0.0873301 , 0.05042462, 0.04002318, 0.08752342, 0.05310177,
        0.12751321, 0.03692374, 0.03210329],
       [0.50536287, 0.40187508, 0.34188351, 0.46422759, 0.40724966,
        0.43501315, 0.39823404, 0.44376528],
       [0.3668398 , 0.27315652, 0.461373  , 0.3360019 , 0.23664017,
        0.31276506, 0.29412073, 0.53549188],
       [0.53358728, 0.36802965, 0.21708977, 0.47236824, 0.43918973,
        0.51962173, 0.44420218, 0.45622146],
       [0.56168324, 0.30712512, 0.16652276, 0.48144999, 0.49244592,
        0.53310686, 0.54987139, 0.67307305],
       [0.52984595, 0.36053118, 0.19552168, 0.45720258, 0.4301607 ,
        0.46055669, 0.43031111, 0.47484154],
       [0.53893733, 0.31423646, 0.1728387 , 0.45516062, 0.46098989,
        0.51229393, 0.52995545, 0.56610143],
       [0.47388875, 0.32828775, 0.23127693, 0.46527115, 0.45649374,
        0.4759011 , 0.46881375, 0.59493381]])

test_dice_mat_MT_9 = np.array([[0.43305981, 0.39907384, 0.49142584, 0.43801579, 0.51518297,
        0.40099794, 0.68146896, 0.52909613]])
orig_results_dict = {}
for i, c_name in enumerate(orig_centers_list):
        orig_results_dict[c_name] = test_dice_mat_MT_9[0][i]


er_mean_9, ft_mean_9 = [], []
for i, cname in enumerate(order_seed_9):
        er_mean_9.append(np.mean(test_dice_mat_ER_9[i, :]))     # zero-shot
        ft_mean_9.append(np.mean(test_dice_mat_FT_9[i, :]))

result_matrix_mt_9 = []
for c_name in order_seed_9:
        result_matrix_mt_9.append(orig_results_dict[c_name])
mean_result_matrix_mt_9 = np.mean(result_matrix_mt_9)


df_seed_9 = pd.DataFrame({
        'centers': order_seed_9,
        'er_mean': er_mean_9,
        'ft_mean': ft_mean_9,
        'mean_mt': mean_result_matrix_mt_9,
        'st': mean_test_dice_mat_ST_9
})

# ================================================================================================================

plt.rcParams['axes.grid'] = True
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle('Zero-shot Mean Test Dice per Domain with Randomized Domain Ordering')   # zero-shot

dfs_list = [df_seed_12345, df_seed_4242, df_seed_728, df_seed_171, df_seed_1001, df_seed_1113, 
            df_seed_46, df_seed_5, df_seed_9]   

for i in range(3):
    for j in range(3):
        idx = j + 3*i
        sns.lineplot(ax=axes[i, j], data=dfs_list[idx], x="centers", y="er_mean", label="replay",)
        sns.lineplot(ax=axes[i, j], data=dfs_list[idx], x="centers", y="ft_mean", label="fine-tuning",)
        axes[i, j].axhline(dfs_list[idx]["mean_mt"].values[0], color="g", linestyle='-.', label="multi-domain",)
        axes[i, j].axhline(dfs_list[idx]["st"].values[0], color="m", linestyle='--', label="single-domain",)
        # sns.scatterplot(ax=axes[0,0], data=df_seed_12345, x="centers", y="mean_mt", label="multi-domain", marker="X", color="g", s=75)
        # sns.scatterplot(ax=axes[0,0], data=df_seed_12345, x="centers", y="st", label="single-domain", marker="^", color="m", s=75)
        axes[i, j].set_xlabel("Centers", fontsize=11)
        axes[i, j].set_ylabel("Avg. Test Dice", fontsize=11)
        axes[i, j].legend(ncol = 2)
                
fig.tight_layout()                
plt.setp(axes, ylim=[0.2, 0.65])
plt.savefig('output_Z-S_Dice_random_1.png')
plt.show()


# ================================================================================================================
# FOR PLOTTING THE COMBINED AVERAGE CURVE ACROSS DIFFERENT RANDOM ORDERINGS
# Note: In this case, the name of the center does not matter since it is different for each seed. 
# ================================================================================================================

combined_er = np.array([er_mean_12345, er_mean_4242, er_mean_728, er_mean_171, er_mean_1001, er_mean_1113,
                        er_mean_46, er_mean_5, er_mean_9])
combined_ft = np.array([ft_mean_12345, ft_mean_4242, ft_mean_728, ft_mean_171, ft_mean_1001, ft_mean_1113, 
                        ft_mean_46, ft_mean_5, ft_mean_9])
combined_mt = np.array([mean_result_matrix_mt_12345, mean_result_matrix_mt_4242, mean_result_matrix_mt_728, 
                        mean_result_matrix_mt_171, mean_result_matrix_mt_1001, mean_result_matrix_mt_1113, 
                        mean_result_matrix_mt_46, mean_result_matrix_mt_5, mean_result_matrix_mt_9])
combined_st = np.array([mean_test_dice_mat_ST_12345, mean_test_dice_mat_ST_4242, mean_test_dice_mat_ST_728,
                        mean_test_dice_mat_ST_171, mean_test_dice_mat_ST_1001, mean_test_dice_mat_ST_1113, 
                        mean_test_dice_mat_ST_46, mean_test_dice_mat_ST_5, mean_test_dice_mat_ST_9])

# averaging across all rows to get an averaged value of performance on the first domain, then 2nd, and so on
avg_er_across_domains, std_er_across_domains = np.mean(combined_er, axis=0), np.std(combined_er, axis=0)
# print(avg_er_across_domains, std_er_across_domains)
avg_ft_across_domains, std_ft_across_domains = np.mean(combined_ft, axis=0), np.std(combined_ft, axis=0)
avg_mt_across_domains, std_mt_across_domains = np.mean(combined_mt), np.std(combined_mt)
avg_st_across_domains, std_st_across_domains = np.mean(combined_st), np.std(combined_st)

df_combined = pd.DataFrame({
        'centers': ['center 1', 'center 2', 'center 3', 'center 4', 'center 5', 'center 6', 'center 7', 'center 8'],
        'er_mean': avg_er_across_domains,
        'er_std': std_er_across_domains,
        'ft_mean': avg_ft_across_domains,
        'ft_std': std_ft_across_domains,
        'mt_mean': avg_mt_across_domains,
        'mt_std': std_mt_across_domains,
        'st_mean': avg_st_across_domains,
        'st_std': std_st_across_domains
})

plt.rcParams['axes.grid'] = True
fig, axes = plt.subplots(1, 1, figsize=(7, 5))
fig.suptitle('Average Zero-shot Test Dice Scores with Randomized Domain Ordering')   # zero-shot

sns.lineplot(ax=axes, data=df_combined, x="centers", y="er_mean", label="replay", )
# plt.errorbar(data=df_combined, x="centers", y="er_mean", yerr="er_std", fmt='-o')
plt.fill_between(
        df_combined["centers"], 
        y1=(df_combined["er_mean"].values - df_combined["er_std"].values), 
        y2=(df_combined["er_mean"].values + df_combined["er_std"].values), 
        alpha=0.25, 
)

sns.lineplot(ax=axes, data=df_combined, x="centers", y="ft_mean", label="fine-tuning",)
plt.fill_between(
        df_combined["centers"], 
        y1=(df_combined["ft_mean"].values - df_combined["ft_std"].values), 
        y2=(df_combined["ft_mean"].values + df_combined["ft_std"].values), 
        color=['orange'], alpha=0.25
)

axes.axhline(df_combined["mt_mean"].values[0], color="g", linestyle='-.', label="multi-domain",)
axes.axhline(df_combined["st_mean"].values[0], color="m", linestyle='--', label="single-domain",)

axes.set_xlabel("Center IDs", fontsize=11)
axes.set_ylabel("Avg. Test Dice", fontsize=11)
axes.legend(ncol = 2)
axes.set_xticklabels(df_combined['centers'].values, rotation = 45, ha="center")

fig.tight_layout()                
plt.setp(axes, ylim=[0.2, 0.65])
plt.savefig('output_Z-S_Avg_1.png')
plt.show()

