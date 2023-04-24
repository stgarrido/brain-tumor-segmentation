import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt

def tumor_segmentation(imageName, nSlice):
    # Se obtiene el slice de interés para trabajar
    img = nib.load(imageName)
    imgData = img.get_fdata()
    a = imgData.shape[2] - 1
    slice = imgData[:, :, a-nSlice]

    # Se normaliza el slice
    zeros = np.zeros((320,320))
    normalizedSlice = cv2.normalize(slice, zeros, 0, 255, cv2.NORM_MINMAX)

    # Se convierte el slice normalizado en un array float32
    arraySlice = normalizedSlice.reshape((-1,1))
    arraySlice = np.float32(arraySlice)

    # Se hace uso de k=3 para dividir la imagen en 3 grupos segun la intensidad de voxels
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(data=arraySlice,
                                    K=3,
                                    bestLabels=None,
                                    criteria=criteria,
                                    attempts=10,
                                    flags=cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    arrayKmeans = center[label.flatten()]
    kmeansSlice = arrayKmeans.reshape(slice.shape)

    # Generacion de mascara binaria para aislar el tumor del resto del slice
    umbral = center[1][0]
    _, mask = cv2.threshold(src=kmeansSlice,
                            thresh=umbral,
                            maxval=255,
                            type=cv2.THRESH_BINARY)
    # kernel1 = np.ones((3,3), np.uint8)
    # maskErosion1 = cv2.erode(src=mask,
    #                         kernel=kernel1,
    #                         iterations=4)
    edgeMask = cv2.Canny(mask, 10, 10)
    kernel = np.ones((9,9), np.uint8)
    maskErosion = cv2.erode(src=mask, kernel=kernel, iterations=1)
    maskDilation = cv2.dilate(src=maskErosion, kernel=kernel, iterations=1)
    edgeTumor = cv2.Canny(maskDilation, 10, 10)

    # Binarización de máscara del tumor
    maskDilation = np.float64(maskDilation/255)

    # Segmentación del tumor en slice seleccionado
    tumor = slice*maskDilation
    
    # Se obtienen bordes rojo para la mascara y el tumor
    cor = np.uint8(normalizedSlice)
    rgb = cv2.cvtColor(cor, cv2.COLOR_GRAY2BGR)
    bgr = cv2.cvtColor(cor, cv2.COLOR_GRAY2BGR)
    rgb[:,:,0] = edgeMask; rgb[:,:,1] = 0 ; rgb[:,:,2] = 0
    bgr[:,:,0] = edgeTumor; bgr[:,:,1] = 0 ; bgr[:,:,2] = 0

    # Aplica borde rojo de mascara al slice original
    edgeMaskInv = np.array(1-edgeMask/255)
    edgeSlice = slice * edgeMaskInv
    edgeSlice = cv2.convertScaleAbs(edgeSlice, alpha=(255/edgeSlice.max()))
    edgeSlice = cv2.cvtColor(edgeSlice, cv2.COLOR_GRAY2BGR)
    edgeSlice = edgeSlice | rgb

    # Aplica borde rojo de tumor al slice original
    edgeTumorInv = np.array(1-edgeTumor/255)
    edgeRTumor = slice * edgeTumorInv
    edgeRTumor = cv2.convertScaleAbs(edgeRTumor, alpha=(255/edgeRTumor.max()))
    edgeRTumor = cv2.cvtColor(edgeRTumor, cv2.COLOR_GRAY2BGR)
    edgeRTumor = edgeRTumor | bgr

    # Calculo de intensidad
    intensity = []
    for i in range(tumor.shape[0]):
        for j in range(tumor.shape[1]):
            if tumor[i,j] != 0:
                intensity.append(tumor[i,j])           
    minIntensity = min(intensity)

    # Segmentacion del tumor completo

    onlyTumor = imgData
    upperMask = lowerMask = maskDilation
    # Segmentacion superior del tumor
    n = nSlice
    while n < imgData.shape[2]:
        upperSlice = np.array(onlyTumor[:, :, a-n])
        upperSlice = upperSlice*upperMask
        onlyTumor[:, :, a-n] = upperSlice

        for i in range(upperSlice.shape[0]):
            for j in range(upperSlice.shape[1]):
                if upperSlice[i,j] <= minIntensity:
                    onlyTumor[i, j, a-n] = 0
        _, upperMask = cv2.threshold(np.uint8(onlyTumor[:, :, a-n]), 1, 255, cv2.THRESH_BINARY)
        upperMask = np.array(np.float64(upperMask/255))
        n += 1

    # Segmentacion inferior del tumor
    n = nSlice - 1
    while n >= 0:
        lowerSlice = np.array(onlyTumor[:, :, a-n])
        lowerSlice = lowerSlice*lowerMask
        onlyTumor[:, :, a-n] = lowerSlice

        for i in range(lowerSlice.shape[0]):
            for j in range(lowerSlice.shape[1]):
                if lowerSlice[i,j] <= minIntensity:
                    onlyTumor[i,j, a-n] = 0
        _, lowerMask = cv2.threshold(np.uint8(onlyTumor[:, :, a-n]), 1, 255, cv2.THRESH_BINARY)
        lowerMask = np.array(np.float64(lowerMask/255))
        n -= 1

    allTumor = nib.nifti1.Nifti1Image(onlyTumor, None, header=img.header)

    return edgeRTumor, allTumor

def main():
    sliceTumorBorder, tumor =  tumor_segmentation('data/case_014_2.nii.gz', 17)

    # Guarda slice del tumor delineado
    cv2.imwrite('results/Tumor_delineado_case_014_2.jpg', cv2.cvtColor(sliceTumorBorder, cv2.COLOR_RGB2BGR))

    # Guarda la extraccion del tumor completo
    nib.save(tumor, 'results/Tumor_completo_case_014_2.nii')

if __name__ == "__main__":
    main()