import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DB = pd.read_csv('Mall_Customers.csv')

print('Cantidad de datos por atributo =', str(len(DB['Age'])) + '.')
DB.head()

def Normaliza(DB):
    DB = DB.to_numpy()

    Atributos = DB[0]
    NoAtributos = len(Atributos)
    Instancias = DB.T[0]
    NoInstancias = len(Instancias)

    MaximoDeAtributos = []
    MinimoDeAtributos = []
    for idx, element in enumerate(Atributos):
      CaractMax = max(DB.T[idx])
      CaractMin = min(DB.T[idx])
      MaximoDeAtributos.append(CaractMax)
      MinimoDeAtributos.append(CaractMin)

    DBNorm = []
    MaximoNormalizado = 1
    MinimoNormalizado = 0
    RangoNormalizado = MaximoNormalizado - MinimoNormalizado
    for idx, element in enumerate(Atributos):
      CaractNorm = []
      if str(type(Atributos[idx]))[8 : -2] != 'str':
        RangodeDatos = MaximoDeAtributos[idx] - MinimoDeAtributos[idx]
        for idx2, element2 in enumerate(Instancias):
          if str(DB.T[idx][idx2]) != 'nan':
            D = DB.T[idx][idx2] - MinimoDeAtributos[idx]
            DPct = D / RangodeDatos
            dNorm = RangoNormalizado * DPct
            Normalizado = MinimoNormalizado + dNorm
            CaractNorm.append(Normalizado)
          else:
            CaractNorm.append(DB.T[idx][idx2])
      else:
        for idx2, element2 in enumerate(Instancias):
          CaractNorm.append(DB.T[idx][idx2])
      DBNorm.append(CaractNorm)
    return(DBNorm)

DB_Norm = Normaliza(DB)

X = np.array(DB_Norm[3])
Y = np.array(DB_Norm[2])

plt.scatter(X, Y, color = 'lightblue', label = 'Datos')
plt.title('Ingreso Anual (k$) vs Edad')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Edad')
plt.legend()
plt.show()

def data2point(X, Y):
    puntos = []
    for idx, x in enumerate(X):
        puntos.append((x, Y[idx]))
    return(puntos)

puntos = data2point(X, Y)

import numpy as np
import random as rd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.cluster import KMeans

def LS(LSpuntos):
    LSsumaX = 0
    LSsumaY = 0
    for LSpunto in LSpuntos:
        LSsumaX += LSpunto[0]
        LSsumaY += LSpunto[1]
    return((LSsumaX, LSsumaY))

def SS(SSpuntos):
    SSsuma = 0
    for SSpunto in SSpuntos:
        SSPX = SSpunto[0]**2
        SSPY = SSpunto[1]**2
        SSsuma += SSPX + SSPY
    return(SSsuma)

def centroid(Cpuntos):
    LS_temp = LS(Cpuntos)
    Nc = len(Cpuntos)
    return((LS_temp[0] / Nc, LS_temp[1] / Nc))

def CF_centroid(CFc):
    NCFc = CFc[0]
    CFcX = CFc[1][0] / NCFc
    CFcY = CFc[1][1] / NCFc
    return((CFcX, CFcY))

def Radius(Rpuntos):
    Nr = len(Rpuntos)
    RP1 = SS(Rpuntos) / Nr
    RP2 = centroid(Rpuntos)**2
    R_temp = (RP1 - RP2)**0.5
    return(R_temp)

def Diameter(CFDin):
    DN = CFDin[0]
    DLs = LS([CFDin[1]])
    DLs = max(DLs)
    DSs = SS([CFDin[1]])
    DP1 = 2 * DN * DSs
    DP2 = -2 * (DLs**2)
    DP3 = DN**2
    D_temp = (DP1 + DP2) / DP3
    return(D_temp)

def CF(CFpuntos):
    LS_temp = LS(CFpuntos)
    SS_temp = SS(CFpuntos)
    NCF = len(CFpuntos)
    return([NCF, LS_temp, SS_temp])

def CF_merge(CF1, CF2):
    CF1_1 = CF1[1]
    CF2_1 = CF2[1]
    CFr = [CF1[0] + CF2[0], (CF1_1[0] + CF2_1[0], CF1_1[1] + CF2_1[1]), CF1[2] + CF2[2]]
    return(CFr)

def EC_distance(EDpuntos):
    dist = ((EDpuntos[0][0] - EDpuntos[1][0])**2 + (EDpuntos[0][1] - EDpuntos[1][1])**2)**0.5
    return(dist)

def ForinCentroids(FiCs, height):
    if height == 0:
        FiC_CF = FiCs
    elif height == 1:
        FiC_CF = [0, (0, 0), 0]
        for FiC in FiCs:
            FiC_CF = [FiC_CF[0] + FiC[0], (FiC_CF[1][0] + FiC[1][0], FiC_CF[1][1] + FiC[1][1]), FiC_CF[2] + FiC[2]]
    else:
        FiC_CF = [0, (0, 0), 0]
        idx = height - 1
        for FiC in FiCs:
            FiCt = ForinCentroids(FiC, idx)
            FiC_CF = [FiC_CF[0] + FiCt[0], (FiC_CF[1][0] + FiCt[1][0], FiC_CF[1][1] + FiCt[1][1]), FiC_CF[2] + FiCt[2]]
    return(FiC_CF)

def AllCentroids(ACelements, ACnum):
    cent = []
    for CFi in ACelements:
        CFNLN = ForinCentroids(CFi, ACnum)
        centa = CF_centroid(CFNLN)
        cent.append(centa)
    return(cent)

def AllDists(ADelements, CFa):
    distAll = []
    for ADelement in ADelements:
        dista = EC_distance([CFa, ADelement])
        distAll.append(dista)
    return(distAll)

def Select_TE(TreeElement, TEpunto, TEidx):
    TECent = AllCentroids(TreeElement, TEidx)
    TEDist = AllDists(TECent, TEpunto)
    TEmDist = min(TEDist)
    TEDist = np.array(TEDist)
    TEmPosa = np.where(TEDist == TEmDist)
    return(TEmPosa[0][0])

def CFClosest(Rootin, punto):
    mRootpos = Select_TE(Rootin, punto, 2)
    mNonLNpos = Select_TE(Rootin[mRootpos], punto, 1)
    mLeafNodepos = Select_TE(Rootin[mRootpos][mNonLNpos], punto, 0)
    CFmca = Rootin[mRootpos][mNonLNpos][mLeafNodepos]
    return(CFmca, mRootpos, mNonLNpos, mLeafNodepos)

def farthest(LNin):
    FE = []
    FEpos = []
    for CFsplit1 in LNin:
        for CFsplit2 in LNin:
            FEt = EC_distance([CFsplit1[1], CFsplit2[1]])
            if FEt not in FE:
                FE.append(FEt)
                FEpos.append([CFsplit1, CFsplit2])
    F = max(FE)
    FE = np.array(FE)
    Fpos = np.where(FE == F)
    CFf1 = FEpos[Fpos[0][0]][0]
    CFf2 = FEpos[Fpos[0][0]][1]
    return(CFf1, CFf2)

def AllCF(ARoot):
    AllCFs = []
    for ANLN in ARoot:
        for ALN in ANLN:
            for ACF in ALN:
                AllCFs.append(ACF)
    return(AllCFs)

def ReduceXCFtree(Tin, RRoot, RBin, RLin):
    newT = Tin + 0.1
    CF_Done = AllCF(RRoot)
    newRoot = [[[CF_Done[0]]]]
    for Ridx, RCF in enumerate(CF_Done[1:]):
        CFmc_all = CFClosest(newRoot, RCF[1])
        CFmc = CFmc_all[0]
        Da = Diameter(newRoot[CFmc_all[1]][CFmc_all[2]][CFmc_all[3]])
        CFa = RCF
        if Da < newT:
            newCF = CF_merge(CFa, CFmc)
            newRoot[CFmc_all[1]][CFmc_all[2]][CFmc_all[3]] = newCF
        else:
            if len(newRoot[CFmc_all[1]][CFmc_all[2]]) < RLin:
                newRoot[CFmc_all[1]][CFmc_all[2]].append(CFa)
            else:
                if len(newRoot[CFmc_all[1]]) < RBin:
                    CFf = farthest(newRoot[CFmc_all[1]][CFmc_all[2]])
                    
                    LNold = []
                    for element in newRoot[CFmc_all[1]][CFmc_all[2]]:
                        if element not in CFf:
                            LNold.append(element)
                         
                    LNold.append(CFa)
                    newRoot[CFmc_all[1]][CFmc_all[2]] = [CFf[0]]
                    newRoot[CFmc_all[1]].append([CFf[1]])
                    
                    for element in LNold:
                        tempD1 = EC_distance([CFf[0][1], element[1]])
                        tempD2 = EC_distance([CFf[1][1], element[1]])
                        dest = min([tempD1, tempD2])
                        if dest == 0:
                            newRoot[CFmc_all[1]][CFmc_all[2]].append(element)
                        else:
                            newRoot[CFmc_all[1]][-1].append(element)
                else:
                    NLNtemp = []
                    for element in newRoot[CFmc_all[1]]:
                        newCF = (0, (0, 0), 0)
                        for element2 in element:
                                newCF = [newCF[0] + element2[0], 
                                         (newCF[1][0] + element2[1][0], newCF[1][1] + element2[1][1]), 
                                         newCF[2] + element2[2]]
                        NLNtemp.append(newCF)

                    CFf = farthest(NLNtemp)

                    ele2 = []

                    for idx, elment in enumerate(NLNtemp):
                        if element not in CFf:
                            ele2.append(newRoot[CFmc_all[1]][idx])

                    ele2.append([CFa])

                    LNold = []
                    for element in NLNtemp:
                        if element not in CFf:
                            LNold.append(element)

                    LNold.append(CFa)

                    newRoot[CFmc_all[1]] = [[CFf[0]]]
                    newRoot.append([[CFf[1]]])

                    for idx, element in enumerate(LNold):
                        tempD1 = EC_distance([CFf[0][1], element[1]])
                        tempD2 = EC_distance([CFf[1][1], element[1]])
                        dest = min([tempD1, tempD2])

                        if dest == 0:
                            newRoot[CFmc_all[1]].append(ele2[idx])
                        else:
                            newRoot[-1].append(ele2[idx])
    
    return(newT, newRoot)

def RootOutliers(RootIn):
    OutNum = []
    for OutNLN in RootIn:
        for OutLN in OutNLN:
            for OutCF in OutLN:
                OutNum.append(OutCF[0])
                
    OutSum = 0
    for OutElement in OutNum:
        OutSum += OutElement
    
    OutProm = OutSum / len(OutNum)
    
    ROutRoot = []
    ROutliers = []
    for OutNLN in RootIn:
        ROutNLN = []
        for OutLN in OutNLN:
            ROutLN = []
            for OutCF in OutLN:
                if OutCF[0] >= (OutProm / 2):
                    ROutLN.append(OutCF)
                else:
                    ROutliers.append(OutCF)
            if len(ROutLN) != 0:
                ROutNLN.append(ROutLN)
        if len(ROutLN) != 0:
            ROutRoot.append(ROutNLN)
    
    return(ROutRoot, ROutliers)

def CFtree(puntosTree, T, BTree, LTree):
    DT = T
    NCFt = len(puntosTree)
    puntosSH = rd.sample(puntosTree, NCFt)
    
    fOutliers = []
    Root = [[[CF([puntosSH[0]])]]]
    for idx, punto in enumerate(puntosSH[1:]):
        CFmc_all = CFClosest(Root, punto)
        CFmc = CFmc_all[0]
        Da = Diameter(Root[CFmc_all[1]][CFmc_all[2]][CFmc_all[3]])
        CFa = CF([punto])
        if Da < DT:
            newCF = CF_merge(CFa, CFmc)
            Root[CFmc_all[1]][CFmc_all[2]][CFmc_all[3]] = newCF
        else:
            if len(Root[CFmc_all[1]][CFmc_all[2]]) < LTree:
                Root[CFmc_all[1]][CFmc_all[2]].append(CFa)
            else:
                if len(Root[CFmc_all[1]]) < BTree:
                    CFf = farthest(Root[CFmc_all[1]][CFmc_all[2]])
                    
                    LNold = []
                    for element in Root[CFmc_all[1]][CFmc_all[2]]:
                        if element not in CFf:
                            LNold.append(element)
                         
                    LNold.append(CFa)
                    Root[CFmc_all[1]][CFmc_all[2]] = [CFf[0]]
                    Root[CFmc_all[1]].append([CFf[1]])
                    
                    for element in LNold:
                        tempD1 = EC_distance([CFf[0][1], element[1]])
                        tempD2 = EC_distance([CFf[1][1], element[1]])
                        dest = min([tempD1, tempD2])
                        if dest == 0:
                            Root[CFmc_all[1]][CFmc_all[2]].append(element)
                        else:
                            Root[CFmc_all[1]][-1].append(element)
                else:
                    if len(Root) < BTree:
                        NLNtemp = []
                        for element in Root[CFmc_all[1]]:
                            newCF = (0, (0, 0), 0)
                            for element2 in element:
                                    newCF = [newCF[0] + element2[0], 
                                             (newCF[1][0] + element2[1][0], newCF[1][1] + element2[1][1]), 
                                             newCF[2] + element2[2]]
                            NLNtemp.append(newCF)

                        CFf = farthest(NLNtemp)

                        ele2 = []

                        for idx, elment in enumerate(NLNtemp):
                            if element not in CFf:
                                ele2.append(Root[CFmc_all[1]][idx])

                        ele2.append([CFa])

                        LNold = []
                        for element in NLNtemp:
                            if element not in CFf:
                                LNold.append(element)

                        LNold.append(CFa)

                        Root[CFmc_all[1]] = [[CFf[0]]]
                        Root.append([[CFf[1]]])

                        for idx, element in enumerate(LNold):
                            tempD1 = EC_distance([CFf[0][1], element[1]])
                            tempD2 = EC_distance([CFf[1][1], element[1]])
                            dest = min([tempD1, tempD2])

                            if dest == 0:
                                Root[CFmc_all[1]].append(ele2[idx])
                            else:
                                Root[-1].append(ele2[idx])
                    else:
                        Reduced = ReduceXCFtree(DT, Root, BTree, LTree)
                        DT = Reduced[0]
                        Root = Reduced[1]
                        POL = RootOutliers(Root)
                        Root = POL[0]
                        fOutliers = POL[1]
    
    for PosOutliers in fOutliers:
        CFmc_all = CFClosest(Root, PosOutliers[1])
        CFmc = CFmc_all[0]
        Da = Diameter(Root[CFmc_all[1]][CFmc_all[2]][CFmc_all[3]])
        CFa = PosOutliers
        if Da < DT:
            newCF = CF_merge(CFa, CFmc)
            Root[CFmc_all[1]][CFmc_all[2]][CFmc_all[3]] = newCF
            
    return(Root)

Umbral = 0.15
B = 8
L = 8
puntos = puntos

rootf = CFtree(puntos, Umbral, B, L)

Colors = ['black', 'indigo', 'pink', 'maroon', 'tomato', 'sienna', 'navy', 'darkorange',
          'tan', 'papayawhip', 'oldlace', 'darkgoldenrod', 'gold', 'yellow', 'yellowgreen', 
          'lawngreen', 'lightgreen', 'darkgreen', 'aquamarine', 'teal', 'blue', 'skyblue', 'cyan', 'red']

figure(figsize=(10, 10))
for idx, NLNf in enumerate(rootf):
    XBf = []
    YBf = []
    for LNf in NLNf:
        for CFf in LNf:
            XBf.append(CFf[1][0])
            YBf.append(CFf[1][1])
            
    plt.scatter(XBf, YBf, color = Colors[idx], label = 'LeafNode de Non-LeafNode ' + str(idx + 1))

plt.title('Cluster Features (CF)')
plt.xlabel('CFx')
plt.ylabel('CFy')
plt.legend()
plt.show()

def NormalizaPuntos(NPuntos):
    
    Xs = []
    Ys = []
    for NPunto in NPuntos:
        Xs.append(NPunto[0])
        Ys.append(NPunto[1])
    
    
    MaximoDeX = max(Xs)
    MinimoDeX = min(Xs)
    MaximoDeY = max(Ys)
    MinimoDeY = min(Ys)

    PuntosNorm = []
    MaximoNormalizado = 1
    MinimoNormalizado = 0
    RangoNormalizado = MaximoNormalizado - MinimoNormalizado
    for Npunto in NPuntos:
        RangodeDatosX = MaximoDeX - MinimoDeX
        RangodeDatosY = MaximoDeY - MinimoDeY
            
        DX = Npunto[0] - MinimoDeX
        DY = Npunto[1] - MinimoDeY
        
        DPctX = DX / RangodeDatosX
        DPctY = DY / RangodeDatosY
        
        dNormX = RangoNormalizado * DPctX
        dNormY = RangoNormalizado * DPctY
        
        NormalizadoX = MinimoNormalizado + dNormX
        NormalizadoY = MinimoNormalizado + dNormY
        
        PuntosNorm.append((NormalizadoX, NormalizadoY))
    
    return(PuntosNorm)

def AgrupDatos(Centroides, CentroideXDato, ADPuntos):
    PuntosDCentr = []
    for idx, Centroide in enumerate(Centroides):
        PuntosDColorX = []
        positions = np.where(CentroideXDato == idx)
        for position in positions[0]:
            PuntosDColorX.append(ADPuntos[position])
        PuntosDCentr.append(PuntosDColorX)
    return(PuntosDCentr)

CFpoints = []
for NLNf in rootf:
    for LNf in NLNf:
        for CFf in LNf:
            CFpoints.append(CFf[1])

CFpoints_Norm = NormalizaPuntos(CFpoints)

CFkmeans = KMeans(n_clusters=5, init='k-means++', n_init=1, random_state=0).fit(CFpoints_Norm)
CFCentroideXDato = CFkmeans.labels_
CFCentroides = CFkmeans.cluster_centers_
CFPuntosDCentr = AgrupDatos(CFCentroides, CFCentroideXDato, puntos)

CFXp = []
CFYp = []
for CFelement in CFCentroides:
    CFXp.append(CFelement[0])
    CFYp.append(CFelement[1])

Colors = ['yellow', 'orange', 'lightblue', 'purple', 'lightgreen']

for idx, CFPuntos in enumerate(CFPuntosDCentr):
    CFXColor = []
    CFYColor = []
    for CFelement in CFPuntos:
        CFXColor.append(CFelement[0])
        CFYColor.append(CFelement[1])
    plt.scatter(CFXColor, CFYColor, color = Colors[idx], label = 'Datos de Centroide ' + str(idx + 1))
    
plt.scatter(CFXp, CFYp, color = 'red', label = 'Centroides')
#plt.title('Ingreso Anual (k$) vs Edad')
#plt.xlabel('Ingreso Anual (k$)')
#plt.ylabel('Edad')
plt.legend()
plt.show()

def distACentrsXpunto(Centrs, DaXpuntos):
    dist = []
    for element in Centrs:
        distXData = []
        for DaXpunto in DaXpuntos:
            distData = ((element[0] - DaXpunto[0])**2 + (element[1] - DaXpunto[1])**2)**0.5
            distXData.append(distData)
        dist.append(distXData)
    return(dist)

def Dato2CentrX(D2Xpuntos, dist):
    MinimIdx = []
    for idx, D2Xpunto in enumerate(D2Xpuntos):
        inim = []
        for element in dist:
            inim.append(element[idx])
        minimo = min(inim)
        MinIdx = inim.index(minimo)
        MinimIdx.append(MinIdx)
    return(MinimIdx)

CFdistancias = distACentrsXpunto(CFCentroides, puntos)
CFCentroideXDato = Dato2CentrX(puntos, CFdistancias)
CFCentroideXDato = np.array(CFCentroideXDato)
CFPuntosDCentr = AgrupDatos(CFCentroides, CFCentroideXDato, puntos)

CFXp = []
CFYp = []
for CFelement in CFCentroides:
    CFXp.append(CFelement[0])
    CFYp.append(CFelement[1])

Colors = ['yellow', 'orange', 'lightblue', 'purple', 'lightgreen']

for idx, CFPuntos in enumerate(CFPuntosDCentr):
    CFXColor = []
    CFYColor = []
    for CFelement in CFPuntos:
        CFXColor.append(CFelement[0])
        CFYColor.append(CFelement[1])
    plt.scatter(CFXColor, CFYColor, color = Colors[idx], label = 'Datos de Centroide ' + str(idx + 1))
    
plt.scatter(CFXp, CFYp, color = 'red', label = 'Centroides')
plt.title('Ingreso Anual (k$) vs Edad')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Edad')
plt.legend()
plt.show()

def Centrs_promedio(Datos):

    Centrs = []
    for K in Datos:
        sumaX = 0
        sumaY = 0
        for data in K:
            sumaX += data[0]
            sumaY += data[1]
        X = sumaX / len(K)
        Y = sumaY / len(K)
        Centrs.append((X, Y))
    return(Centrs)

CFCentroides = Centrs_promedio(CFPuntosDCentr)

CFXp = []
CFYp = []
for CFelement in CFCentroides:
    CFXp.append(CFelement[0])
    CFYp.append(CFelement[1])

Colors = ['yellow', 'orange', 'lightblue', 'purple', 'lightgreen']

for idx, CFPuntos in enumerate(CFPuntosDCentr):
    CFXColor = []
    CFYColor = []
    for CFelement in CFPuntos:
        CFXColor.append(CFelement[0])
        CFYColor.append(CFelement[1])
    plt.scatter(CFXColor, CFYColor, color = Colors[idx], label = 'Datos de Centroide ' + str(idx + 1))
    
plt.scatter(CFXp, CFYp, color = 'red', label = 'Centroides')
plt.title('Ingreso Anual (k$) vs Edad')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Edad')
plt.legend()
plt.show()

for idx, CFCentroide in enumerate(CFCentroides):
    print('El centroide', str(idx + 1), 'tiene coordenadas', str(CFCentroide) + '.')