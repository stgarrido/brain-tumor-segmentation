import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt

def tumor_extraction(image_name, n):
    # Se obtiene el slice de interés para trabajar
    img = nib.load(image_name)
    imgData = img.get_fdata()
    a = imgData.shape[2] - 1
    slice = imgData[:, :, a-n]

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

    plt.figure(1)
    plt.subplot(221), plt.imshow(slice, cmap=plt.cm.gist_gray), plt.axis('off'), plt.title('Slice')
    plt.subplot(222), plt.imshow(normalizedSlice, cmap=plt.cm.gist_gray), plt.axis('off'), plt.title('Normalize')
    plt.subplot(223), plt.imshow(kmeansSlice, cmap=plt.cm.gist_gray), plt.axis('off'), plt.title('K-means = 3')
    plt.subplot(224), plt.imshow(mask, cmap=plt.cm.gist_gray), plt.axis('off'), plt.title('Mascara binaria')
    plt.show()
    plt.figure(2)
    plt.subplot(121), plt.imshow(edgeMask, cmap=plt.cm.gist_gray), plt.axis('off'), plt.title('Bordes mascara')
    plt.subplot(122), plt.imshow(edgeTumor, cmap=plt.cm.gist_gray), plt.axis('off'), plt.title('Bordes tumor')
    plt.show()
    plt.figure(3)
    plt.subplot(121), plt.imshow(rgb, cmap=plt.cm.gist_gray), plt.axis('off'), plt.title('Borde rojo mascara')
    plt.subplot(122), plt.imshow(bgr, cmap=plt.cm.gist_gray), plt.axis('off'), plt.title('Borde rojo tumor')
    plt.show()
    plt.figure(4)
    plt.subplot(121), plt.imshow(maskErosion, cmap=plt.cm.gist_gray), plt.axis('off'), plt.title('Erosion')
    plt.subplot(122), plt.imshow(maskDilation, cmap=plt.cm.gist_gray), plt.axis('off'), plt.title('Dilatacion')
    plt.show()
    plt.figure(5)
    plt.subplot(121), plt.imshow(edgeSlice, cmap=plt.cm.gist_gray), plt.axis('off'), plt.title('Slice delineado')
    plt.subplot(122), plt.imshow(edgeRTumor, cmap=plt.cm.gist_gray), plt.axis('off'), plt.title('Tumor delineado')
    plt.show()

    cv2.imwrite('results/Tumor_delineado_'+ image_name[5:-7] + '.jpg', cv2.cvtColor(edgeRTumor, cv2.COLOR_RGB2BGR))

    return tumor, edgeRTumor

def main():
    sliceTumor, edgeTumor =  tumor_extraction('data/case_014_2.nii.gz', 17)

    plt.imshow(sliceTumor, cmap=plt.cm.gist_gray), plt.axis('off'), plt.title('Slice tumor segmentado')
    plt.show()

if __name__ == "__main__":
    main()