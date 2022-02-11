import numpy as np

# Otsu (version du prof)


def otsu(img):
    meilleurSeuil = 0
    minimun = 10_000_000_000

    histogram = np.zeros(256, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            histogram[img[i][j]] += 1

    for i in range(len(histogram)):
        print(i, " : ", histogram[i])

    for seuil in range(256):
        w1 = 0
        w2 = 0
        mu1 = 0
        mu2 = 0
        for i in range(0, seuil):
            w1 += histogram[i]
            mu1 += i * histogram[i]
        for i in range(seuil, 256):
            w2 += histogram[i]
            mu2 += i * histogram[i]

        if w1 == 0 or w2 == 0:
            continue

        mu1 /= w1
        mu2 /= w2
        w1 /= (img.shape[0] * img.shape[1])
        w2 /= (img.shape[0] * img.shape[1])

        # Variance
        s1 = 0
        s2 = 0
        for i in range(0, seuil):
            s1 += ((i - mu1) ** 2) * histogram[i]
        for i in range(seuil, 256):
            s2 += ((i - mu2) ** 2) * histogram[i]

        intraClassVar = w1 * s1 + w2 * s2
        if intraClassVar < minimun:
            meilleurSeuil = seuil
            minimun = intraClassVar
            print(f"Nouveau meilleur seuil : {seuil}")

    print(f"Le meilleur seuil est {meilleurSeuil}")
