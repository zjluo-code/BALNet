import numpy as np
import random
from astropy.io import fits

Nwave = np.linspace(1300., 1700., num=1165)  
V = (Nwave - 1549)/1549 * 300000 
low = np.where(V >= -29000)[0][0]
up = np.where(V <= 10000)[0][-1] 
li = int(low / 1165 * 387)
ui = int(up / 1165 * 387) + 1 
deltaV = V[1]-V[0]
nn = np.where((V>=-29000)&(V<=10000))[0]

def SaveFits(cata, path, name):
    primary_hdu = fits.PrimaryHDU(cata)
    fits_filename = path + name + '.fits'
    primary_hdu.writeto(fits_filename, overwrite=True)

def RandomNONBAL(RSNR, ASNR, temp1, Flux_DR16, NONi, cut=0):
    ### Based on a subset selected from the SDSS DR16Q with redshift range of 1.5 to 5.7
    ### RSNR: Median SNR (1400-1600 \AA) for observed BALQSO 
    ### ASNR: Median SNR (1400-1600 \AA) for the subset
    ### temp1: Index for non-BALQSO in the subset
    ### Flux_DR16: Flux for this spectra with linear interpolation and three-point moving average smoothing
    ### NONi: index for this subset in the original DR16Q catalog
    counts, bin_edges = np.histogram(RSNR, bins=int((max(RSNR)-min(RSNR))/0.5))
    Mcount = np.round(counts / np.sum(counts) * 100000).astype(int) 
    Mcount[:2] += 1
    counts, _ = np.histogram(ASNR[temp1], bins=bin_edges)
	
    Dsn = []
    Dflux = []
    Nindex = []
    for i in range(0, len(Mcount)): 
        if i == 60:
            num = np.sum(Mcount[i:])   ### number for each SNR bin
            temp = np.where((ASNR[temp1]>=bin_edges[i])&(ASNR[temp1]<bin_edges[-1]))[0]
        else:
            num = Mcount[i]
            temp = np.where((ASNR[temp1]>=bin_edges[i])&(ASNR[temp1]<bin_edges[i+1]))[0]

        if int(num) > 0  and len(temp) > 0:
            if int(num) <= len(temp):
                rannum = random.sample(range(0, len(temp)), int(num))
            else:
                rannum = [random.randint(0, len(temp)-1) for _ in range(int(num))]
        else:
            temp = np.where((ASNR[temp1]>=bin_edges[0])&(ASNR[temp1]<bin_edges[-1]))[0]
            rannum = [random.randint(0, len(temp)-1) for _ in range(int(num))]

        Dsn.append(ASNR[temp1][temp][rannum])
        Dflux.append(Flux_DR16[temp1][temp][rannum])
        Nindex.append(NONi[temp1][temp[rannum]])


    Allflux = np.concatenate(Dflux, axis=0)
    Allsn = np.concatenate(Dsn, axis=0)
    Nindex = np.concatenate(Nindex, axis=0)
    result = np.column_stack((Nindex, Allsn))

    # Generate randomly order
    indices = np.arange(len(Allflux)) 
    np.random.shuffle(indices) 
    Allflux1 = Allflux[indices]
    result1 = result[indices]

    Dpath = 'your_path'   ### for save the simulated spectra
    if cut == 0:
        ### Randomly select the non-BAL quasars, which are used to generate simulated BAL quasars
        SaveFits(result1, Dpath, 'NONBALcatalog_B')
        SaveFits(Allflux1, Dpath, 'NONBALs_B') 
    else:
        ### Randomly select the non-BAL quasars, which serve as the simulated non-BAL quasars
        SaveFits(result1, Dpath, 'NONBALcatalog_N')
        SaveFits(Allflux1, Dpath, 'NONBALs_N')
    
    return Allflux1, result1

def GenNorm(Flux9, random_numbers):
    flux = np.ones(len(V))    ### initial normalized flux is 1
    Ctemp = set()  # save the changed index to avoid overlap between BALs
    successful_indices = []  # List to store successful indices
    EValue = []
    Cz1165 = np.zeros(len(V))
    Cz387 = np.zeros(387)
    for value in random_numbers:  
        tough = Flux9[value]  
        Temp = np.where(tough != 1.)[0]  # Absorption indices  
        # Boundaries for moving indices
        low = (V[nn[0]] - V[Temp[0]]) / deltaV
        upper = (V[nn[-1]] - V[Temp[-1]]) / deltaV
        random_number = random.randint(int(low) -1, int(upper)-1)  
        Ind = Temp + random_number  
        foundunique = False
        if len(Ctemp) == 0 or not any(i in Ctemp for i in Ind):
            Ctemp.update(Ind)
            foundunique = True
        else:
            for _ in range(1000):
                random_number = random.randint(int(low) -1, int(upper)-1) 
                Ind = Temp + random_number
                if not any(i in Ctemp for i in Ind):
                    Ctemp.update(Ind)
                    foundunique = True
                    break
        if not foundunique:
            EValue.append(value)
            continue
        else:
            successful_indices.append(value)
            flux[Ind] = tough[Temp] 
            l = Ind[0]
            u = Ind[-1]
            ll = int(l * 387 / 1165)
            uu = int(u * 387 / 1165) + 1
            Cz1165[l:u] = 1
            Cz387[ll:uu] = 1
    return flux, Cz387, Cz1165, successful_indices

def ConstructBALQSO(nonBAL, nonIndex, BALs, Index_BALs):
    Nsnr = nonIndex[:,1]  ### this column is SNR for selected non-BALQSO
    Bsnr = Index_BALs[:,1]  ### this column is SNR for BAL pattern library 47,267
    ### Generate a randomly index
    indices = np.arange(len(Nsnr)) 
    np.random.shuffle(indices)   
    ### Number of C IV BAL troughs per spectrum for observed BALQSO
    NSv = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])  
    ### According to the same number distribution, convert to the 100,000 simulated BALQSO
    NS = np.array([37717, 34326, 17826, 6919, 2198, 759, 212, 30, 13]) 
    Current = np.zeros(len(NSv)).astype(int)  ## initial is 0
    for value in indices:
        Tag = []
        sn = Nsnr[value]
        # Limit the SNR diff with the given nonBALQSO and BAL troughs
        mask = np.where(np.abs(Bsnr - sn) <= 0.5)[0]
        k = 1
        while len(mask) <= 0:
            mask = np.where(np.abs(Bsnr - sn) <= (k+1)*0.5)[0]
            k += 1
        comparison = NS - Current
        
        invalid_indices = np.where(comparison > 0)[0]

        if len(invalid_indices) > 0: 
            temp = NS[invalid_indices] - Current[invalid_indices]
            probabilities = temp / temp.sum()
            Bnum = np.random.choice(NSv[invalid_indices], p=probabilities) 

            while Current[Bnum-1] >= NS[Bnum-1]:
                Bnum = np.random.choice(NSv[invalid_indices], p=probabilities)
            
            rannum = [random.randint(0, len(mask)-1) for _ in range(int(Bnum))]  
            flux, Cz387, Cz1165, sucind = GenNorm(BALs, mask[rannum])

            while len(sucind) < Bnum:
                rannum = [random.randint(0, len(mask)-1) for _ in range(int(Bnum))]  
                flux, Cz387, Cz1165, sucind = GenNorm(BALs, mask[rannum])
            Current[len(sucind)-1] += 1
            M_spec.append(flux * nonBAL[value])
            NON_spec.append(nonBAL[value])
            BALflux.append(flux)
            Mcata.append(Cz387)
            Mcata1165.append(Cz1165)
            Tag.append(int(1))
            Tag.extend(nonIndex[value])
            Tag.append(len(sucind))
            Tag.extend(sucind)
            Tag.extend([-1]*(10-len(sucind)-1))
            Index.append(Tag)

    BALflux = np.array(BALflux)
    M_spec = np.array(M_spec)
    NON_spec = np.array(NON_spec)
    Mcata = np.array(Mcata)
    Mcata1165 = np.array(Mcata1165)
    Index = np.array(Index)
    
    return BALflux, M_spec, NON_spec, Mcata, Mcata1165, Index

def readfits(fname):
    # open FITS file
    hdul = fits.open(fname)
    # hdul.info()
    data = hdul[0].data
    hdul.close()
    return data


def Read_catalog(path):
	hdulist = fits.open(path)
	BI_CIV = hdulist[1].data['BI_CIV']
	AI_CIV = hdulist[1].data['AI_CIV']
	hdulist.close()
	return BI_CIV, AI_CIV

if __name__ == '__main__':
    Dpath = 'your path'
    BI_CIV, AI_CIV = Read_catalog('DR16Q_v4.fits download from SDSS DR16 quasar catalog')
    
	#### The median SNR (1400-1600 \AA in restframe wavelength) observed BALQSO from DR16Q
    RSNR = readfits(Dpath + 'SNR_BALQSO.fits')
    
	#### The subset spectra from DR16Q with redshift range of 1.5 to 5.7
    Flux_DR16 = readfits(Dpath+'Flux_DR16.fits')  
    
	#### The corresponding information for the Flux_DR16.fits
    Index_DR16 = readfits(Dpath+'Index_DR16.fits')
    NONi = Index_DR16[:, 1].astype(int)   ### subset index in the origin DR16Q
    ASNR = Index_DR16[:, 3]  ### subset SNR
    temp1 = np.where((BI_CIV[NONi]==0)&(AI_CIV[NONi]==0))[0]  ## the index of non-BALQSO in the original DR16
    
	##### BAL pattern library obtained from the observed BALQSO
    BALs = readfits(Dpath + 'BALs.fits')   ### 47267*1165
    Index_BALs = readfits(Dpath + 'BALscatalog.fits')  ### (index in DR16, SNR, Number of BAL, AI, Vmax, Vmin)

    Allflux_B, result_B = RandomNONBAL(RSNR, ASNR, temp1, Flux_DR16, NONi, cut=0)
    BALflux, M_spec, NON_spec, Mcata, Mcata1165, MIndex = ConstructBALQSO(Allflux_B, result_B, BALs, Index_BALs)
    
    Allflux_N, result_N = RandomNONBAL(RSNR, ASNR, temp1, Flux_DR16, NONi, cut=1)
    BALflag = np.zeros(len(Allflux_N)).astype(int)
    BALflag = BALflag[:, np.newaxis]
    M_spec = np.concatenate((M_spec, Allflux_N), axis=0)
    NON_spec = np.concatenate((NON_spec, Allflux_N), axis=0)
    NormBALs = np.concatenate((BALflux, np.ones((len(Allflux_N), 1165))), axis=0)
    Mcata = np.concatenate((Mcata, np.zeros((len(Allflux_N), 387))), axis=0)
    Mcata1165 = np.concatenate((Mcata1165, np.zeros((len(Allflux_N), 1165))), axis=0)

    TIndex = np.hstack((result_N, np.ones((len(Allflux_N), 10))*(-1)))
    IndexN = np.hstack((BALflag, TIndex))
    Index = np.concatenate((MIndex, IndexN), axis=0)

    ## print(np.isnan(M_spec).sum())
    indices = np.arange(len(M_spec))  
    np.random.shuffle(indices)     
    np.random.shuffle(indices)      
    M_spec1 = M_spec[indices]
    NON_spec1 = NON_spec[indices]
    Mcata1 = Mcata[indices]
    Mcata11651 = Mcata1165[indices]
    Index11 = Index[indices]
    NormBALs1 = NormBALs[indices]

    ## 200,000 simulated data are randomly stored, M_SPEC is the spectra and Mcata is the corresponding label
    SaveFits(M_spec1, Dpath, 'M_SPEC')
    SaveFits(NON_spec1, Dpath, 'NON_SPEC')
    SaveFits(Mcata1, Dpath, 'Mcata')
    SaveFits(Mcata11651, Dpath, 'Mcata1165')
    SaveFits(Index11, Dpath, 'Index')
    SaveFits(NormBALs1, Dpath, 'BAL_flux')
