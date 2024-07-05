import mne.io

# @TODO order everything
# def getKCs(filt, name_channel = 'C4_1', sf = 200, pathKC = ''): #Return a vector with only KCs labeled (everything else is zero)
    
#     timeseries = []

#     for j in range(0, len(filt)):
#         data = filt[j][0]
#         path_KC = pathKC[j]
#         filetxt = open(path_KC, 'r')

#         for x in filetxt:
#             if x[-3] == 'K' and x[-2] == 'C':
#                 vector, cantKC = extract_td(x)  #vector[0,0] = position of KC // vector[0,1] = duration of KC
#                 start = int(vector[0,0]*sf)
#                 end = int((vector[0,0] + vector[0,1])*sf)
#                 a_KC, leng = align_KC(start, end, data, sf)
#                 timeseries.append(a_KC.tolist())
        
#         filetxt.close()
    
#     return timeseries

# def get_noKCs(filt, name_channel, sf, s_N = [11,12,17,21,28,36,37,40]):
    
#     noKCs = []
    
#     for sN in s_N:
#         if sN == 11:
#             pos_inic_noKC = [1124,1180,1227,1259,1283,1379,1520.5,1526.5,1653]
#             data = filt[0][0]

#         elif sN == 12:
#             #pos_inic_noKC = [1005,1180,1220,1320,1360,1485,1585,1645,1730,1860,1945,2020,2130,2215]
#             pos_inic_noKC = [1005,1180,1255.5,1288,1300.5,1331.5,1507.5,1529,1655,1743,1770,1789,1801,1900.5]
#             data = filt[1][0]

#         elif sN == 17:
#             #pos_inic_noKC = [910,920,995,1050,1140,1175,1235,1275,1330,1375,1445,1525,1565,1660,1730,2033]
#             pos_inic_noKC = [910,970,979.5,988.5,1145,1160.5,1230.5,1251,1320,1336,1436.5,1444,1466,1504,1506,1534]
#             data = filt[2][0]

#         elif sN == 21:
#             #pos_inic_noKC = [575,605,690,740,805,845,920,955,1015,1090,1160,1200,1255,1260,2715,2770,2805,2860,2875,2905]
#             pos_inic_noKC = [635.5,663.5,854.5,740,878.5,962,980,1006,1052.5,1055.5,1085.5,1096.5,1148.5,1200,1204.5,1213,1219,1234,1240.5,1258.5]
#             data = filt[3][0]

#         elif sN == 28:
#             pos_inic_noKC = [293,318,378.5,505,582.5,666.5,793.5]
#             data = filt[4][0]

#         elif sN == 36:
#             #pos_inic_noKC = [435,500,620,750,820.5,1065,1180,1270,1340,1510,1660,1825,1900,1980,2018,2100,2130,2180,2420,2600,3035,3350,3540,3665,3875,4100]
#             pos_inic_noKC = [542,784.5,804,1083,1117.5,1129,1182,1211,1273.5,1295.5,1305.5,1324.5,1344.5,1434,1493.5,1509,1589,1636.5,1664,1740.5,1798,1817,1927,2203,2221,2309]
#             data = filt[5][0]
        
#         elif sN == 37:
#             pos_inic_noKC = [1776.5,2055]
#             data = filt[6][0]

#         elif sN == 40:
#             #pos_inic_noKC = [755,925,945,1040,1090,1160,1200,1255,1295,1370,1495,1840]
#             pos_inic_noKC = [767,774,921,960,1021,1044,1160.5,1187,1329,1364,1744,1844]
#             data = filt[7][0]
#         else:
#             print('Do not use S',sN, ' because there are no KCs enough to make a pipeline')

#         for start in pos_inic_noKC:
#             noKCs.append(data[int(start):int(start+2*sf)]) #tolist()

#     return noKCs

# def clasification(x , list_testing, name_channel = 'C4_1',sf = 200, path_KCbbdd = ''):
    
#     yesKCs = getKCs(x, name_channel = 'C4_1', sf = sf, pathKC = path_KCbbdd)
#     noKCs = get_noKCs(x, name_channel = 'C4_1', sf = sf)

#     y_yesKCs = np.ones(len(yesKCs)) # 1 --> yes KC
#     y_noKCs = np.zeros(len(noKCs)) # 0 --> no KC

#     y = np.concatenate((y_yesKCs,y_noKCs))
#     X = np.concatenate((np.array(yesKCs), np.array(noKCs)))

#     clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True, random_state = 42))
#     cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)
#     scores0 = cross_val_score(clf, X, y, cv=cv)
#     print('Puntajes del modelo a con una validación cruzada de 10 splits y tomando a la mitad de los datos como entrenamiento:\n ', scores0)
#     clf.fit(X,y)
#     return clf

# def extract_td_phase2(x):  #Extract the position and duration of the signal with phase 2 scoring   
#     vector = np.zeros([1,2])  
#     for i in range(len(x)-1):  
#         cant_p2 = 0
#         if x[i] == ',' and x[i+1] != '2':
#             start =  float(x[0:i])
#             vector[0,0] = start           
#             duration = float(x[i+1:len(x)-5])
#             vector[0,1] = duration
#             cant_p2 += 1
    
#     return vector, cant_p2

# def detect_if_S2(path_KC): #Return a list with all stages with phase 2 labeling (together) 
#     starts_S2 =[]
#     filetxt = open(path_KC, 'r')
#     for x in filetxt:
#         if x[len(x)-4:len(x)-1] == '2.0':
#             vector, cantKC = extract_td_phase2(x)  #vector[0,0] = position // vector[0,1] = duration
#             start = int(vector[0,0])
#             starts_S2.append(start)

#     filetxt.close()

#     return starts_S2

# def put_flag(filt, no_filt, sf = 250, name_channel = 'C4', path_KC = ''):
    
#     x = filt[0]
#     x_nofilt = no_filt
#     tamano = x.shape[0]
#     flags = np.zeros(tamano)
#     window = 2
#     if_S2 = detect_if_S2(path_KC)
#     cant = 0
    
#     for start in if_S2:
#         inicio = int(start*sf)
#         final = inicio + int(30*sf)
#         i = inicio

#         while i < final:
#             data = x_nofilt[i:int(i+window*sf)].tolist()
#             maxi = max(data)
#             mini = min(data)
#             pos_maxi = data.index(maxi)
#             pos_mini = data.index(mini)
#             maxi = maxi * (10**6)
#             mini = mini * (10**6)
#             a_pp = (maxi - mini)
#             t_mini_maxi = (pos_maxi - pos_mini)/sf

#             if (pos_mini<pos_maxi) and (a_pp>75) and (maxi>20) and (mini<-30) and (t_mini_maxi<1) and (t_mini_maxi>0.1):
#                 c = (pos_maxi + pos_mini) // 2
#                 flags[i+int(c-sf):i+int(c+sf)] = 25e-6
#                 i = i + int(window*sf*0.25)
#                 cant = cant + 1
#             else:
#                 i = i + int(window*sf*0.25)
        
#     return flags, cant

# def plot_KC_aligned(filt, name_channel, numKC, path_KC, sf = 250):

#     data_out = np.zeros([1,filt.shape[1]])
#     timeseries = getKCs_list(filt,name_channel, sf, path_KC)
#     data_out[0,sf:int(sf+2*sf)] = timeseries[numKC]

#     return data_out

# def separate_candidates(pos_candidates):
#     inicio = 0
#     pos_candidates = pos_candidates[0]
#     candidates_separated = []
#     ult = len(pos_candidates)
  
#     for i in range(0, ult-1):
#         this_value_pos = pos_candidates[i]
#         next_value_pos = pos_candidates[i+1]

#         if next_value_pos == pos_candidates[-1]:
#             candidates_separated.append(pos_candidates[inicio:])

#         if this_value_pos != next_value_pos - 1:
#             candidates_separated.append(pos_candidates[inicio:i+1])
#             inicio = i+1    

#     return candidates_separated #list

# def detector(clf, filtered_signal, flags, sf=250):

#     data_out =np.zeros([1,filtered_signal.shape[1]])
#     pos_cand = np.where(flags)
#     candi = separate_candidates(pos_cand)
#     cant_KC = 0

#     for a in candi:
#         if len(a) == 2*sf:
#             candidate0 = filtered_signal[0,int(a[0]):int(a[0]+2*sf)]
#             cand = np.array(candidate0)
#             candi = signal.resample(cand, 400)
#             candidate = np.reshape(candi, (1,400))
#             p = clf.predict(candidate)

#             if p == 1: #yes KC
#                 cant_KC = cant_KC + 1
#                 data_out[0,int(a[0]):int(a[0]+2*sf)] = 25e-6

#         elif len(a) > 2*sf:
#             i = a[0]
#             while i < a[-1]:
#                 candidate0 = filtered_signal[0,int(i):int(i+2*sf)]
#                 cand = np.array(candidate0)
#                 candi = signal.resample(cand, 400)
#                 candidate = np.reshape(candi, (1,400))
#                 p = clf.predict(candidate)

#                 if p == 1: #yes KC
#                     cant_KC = cant_KC + 1
#                     data_out[0,int(i):int(i+2*sf)] = 25e-6
#                     i = i+2*sf

#                 elif p == 0: # no KC
#                     i = i + sf/8

#     return data_out, cant_KC

def semiautomatic_detection(raw: mne.io.Raw, eeg_channels_selected=['C4_1']):
    """
    """
    print('Not developed yet')