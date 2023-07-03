import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
import cv2


def get_interest_points(image, feature_width):
    '''
    Returns a set of interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max
        - skimage.measure.regionprops
    返回输入图像的一组兴趣点（请注意，我们建议最后实现此函数，并使用cheat_interest_points（）测试get_features（）和match_features的实现）
    首先实施Harris转角检测器（见Szeliski 4.1.1）。您不需要担心尺度不变性或关键点方向估计为你的哈里斯角探测器。
    您可以创建额外的兴趣点检测器功能（例如MSER）获得额外信贷。
    如果您在边界附近发现虚假（虚假/伪造）兴趣点检测，简单地抑制边缘附近的梯度/角是安全的图像。

    有用的功能：一个有效的解决方案不需要使用所有这些函数，但根据您的实现，您可能会发现一些有用的函数。请参考每个函数/库的文档，随时可以使用或在Piazza上发布任何问题

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here!

    # 归一化
    image = image / 255.0

    # 使用sobel滤波器获得x和y方向上的梯度
    sobel = True
    if sobel:
        # cv2.Sobel函数
        # src：输入图像，一般是灰度图像（单通道）或者彩色图像（多通道）。
        # ddepth：输出图像的深度（数据类型），通常使用cv2.CV_64F。
        # dx和dy：表示要在哪个方向上计算梯度。可以为0、1或- 1，分别表示在x轴方向、y轴方向或者两个轴方向同时计算。
        # ksize：Sobel核的大小。如果指定为 - 1，则使用3x3的Sobel核。一般情况下，使用3或5。
        # 横向梯度
        Ix = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=5)     # cv2.CV_8U是一个8位无符号整数数据类型的常量，表示每个像素的值范围在0到255之间。
        # 纵向梯度
        Iy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=5)
    else:
        # Scharr滤波器是一种类似于Sobel滤波器的边缘检测算子，但它对图像的梯度计算更加敏感，能够更好地捕捉边缘细节。
        Ix = np.abs(cv2.Scharr(image, cv2.CV_32F, 1, 0))
        Iy = np.abs(cv2.Scharr(image, cv2.CV_32F, 0, 1))

    # 对I_x和I_y应用高斯滤波器（高斯平滑）
    sigma = 0.3
    Ix = cv2.GaussianBlur(Ix, (3, 3), sigma)
    Iy = cv2.GaussianBlur(Iy, (3, 3), sigma)

    # 计算对应位置矩阵元素的值
    Ixx = Ix ** 2
    Ixy = Iy * Ix
    Iyy = Iy ** 2

    xs = []
    ys = []
    weight = 0.03
    yuzhi = 0.005
    stride = 2

    # 计算图像中的角点并将其存储在xs和ys列表中
    # 步幅来跳过一些像素，以加快处理速度
    # feature_width是用于计算角点响应的窗口或特征的宽度。
    for y in range(0, image.shape[0] - feature_width, stride):
        for x in range(0, image.shape[1] - feature_width, stride):
            # 计算了在给定特征窗口内的图像梯度的累积和
            Sxx = np.sum(Ixx[y:y + feature_width + 1, x:x + feature_width + 1])
            Syy = np.sum(Iyy[y:y + feature_width + 1, x:x + feature_width + 1])
            Sxy = np.sum(Ixy[y:y + feature_width + 1, x:x + feature_width + 1])
            # 计算了Hessian矩阵的行列式和迹并使用它们计算角点响应函数R
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            R = det - weight * (trace ** 2)
            # 如果角点响应R超过了预先设定的阈值，那么将该位置视为一个角点
            if abs(R) > yuzhi:
                xs.append(x + int(feature_width / 2 - 1))
                ys.append(y + int(feature_width / 2 - 1))

    return np.asarray(xs), np.asarray(ys)


def get_features(image, x, y, feature_width):
    '''
    Returns a set of feature descriptors for a given set of interest points.

    (Please note that we reccomend implementing this function after you have implemented
    match_features)

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each descriptor_window_image_width/4.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''

    # TODO: Your implementation here!
    xs = np.round(x).astype(int)
    ys = np.round(y).astype(int)

    # 归一化
    image = image / 255.0

    features = np.zeros((len(xs), 4, 4, 8))
    # 通过过滤器获得粗略的梯度和对应方向角
    sigma = 0.8
    filtered = cv2.GaussianBlur(image, (5, 5), sigma)
    dx = cv2.Scharr(filtered, cv2.CV_32F, 1, 0)
    dy = cv2.Scharr(filtered, cv2.CV_32F, 0, 1)
    gradient = np.sqrt(np.square(dx) + np.square(dy))
    angles = np.arctan2(dy, dx)
    angles[angles < 0] += 2 * np.pi

    for n, (x, y) in enumerate(zip(xs, ys)):
        # 特征窗口
        i1, i2, j1, j2 = get_location(x, y, image, feature_width)
        grad_window = gradient[i1:i2, j1:j2]
        angle_window = angles[i1:i2, j1:j2]
        # 遍历子特征窗口
        for i in range(int(feature_width / 4)):
            for j in range(int(feature_width / 4)):
                # 高斯衰减窗口函数
                grad = current_window(i, j, grad_window, feature_width).flatten()
                angle = current_window(i, j, angle_window, feature_width).flatten()
                features[n, i, j] = np.histogram(angle, bins=8, range=(0, 2 * np.pi), weights=grad)[0]

    features = features.reshape((len(xs), -1,))
    dividend = np.linalg.norm(features, axis=1).reshape(-1, 1)
    # 处理梯度全为零的情况
    dividend[dividend == 0] = 1     # 由于除以零会导致 np.nan，将其设为1
    features = features / dividend
    yuzhi = 0.3
    features[features >= yuzhi] = yuzhi
    features = features ** 0.8

    return features


# 获取特征窗口在图像上的位置
def get_location(y, x, image, feature_width):
    rows = (x - (feature_width / 2 - 1), x + feature_width / 2)
    # 如果特征窗口超出图像的上边界，调整窗口位置
    if rows[0] < 0:
        rows = (0, rows[1] - rows[0])
    # 如果特征窗口超出图像的下边界，调整窗口位置
    if rows[1] >= image.shape[0]:
        rows = (rows[0] + (image.shape[0] - 1 - rows[1]), image.shape[0] - 1)
    cols = (y - (feature_width / 2 - 1), y + feature_width / 2)
    # 如果特征窗口超出图像的左边界，调整窗口位置
    if cols[0] < 0:
        cols = (0, cols[1] - cols[0])
    # 如果特征窗口超出图像的右边界，调整窗口位置
    if cols[1] >= image.shape[1]:
        cols = (cols[0] - (cols[1] + 1 - image.shape[1]), image.shape[1] - 1)

    return int(rows[0]), int(rows[1]) + 1, int(cols[0]), int(cols[1]) + 1


# 获取当前子特征窗口在矩阵中的位置
def current_window(i, j, mat, feature_width):
    return mat[int(i*feature_width/4): int((i+1)*feature_width/4), int(j*feature_width/4): int((j+1)*feature_width/4)]


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - zip (python built in function)

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here!

    # 变量初始化
    matches = []
    confidences = []

    # 循环遍历第一张图像中的特征点
    for i in range(im1_features.shape[0]):
        # 计算第一张图中的特征向量与第二张图中的所有其他特征向量之间的欧氏距离
        distances = np.sqrt(((im1_features[i, :] - im2_features) ** 2).sum(axis=1))

        # 按升序对距离进行排序，同时保留该距离的索引
        ind_sorted = np.argsort(distances)
        # 如果两个最小距离之间的比率小于0.9，则将最小距离添加到最佳匹配
        if distances[ind_sorted[0]] < 0.9 * distances[ind_sorted[1]]:
            matches.append([i, ind_sorted[0]])
            confidences.append(1.0 - distances[ind_sorted[0]] / distances[ind_sorted[1]])

    confidences = np.asarray(confidences)
    confidences[np.isnan(confidences)] = np.min(confidences[~np.isnan(confidences)])

    return np.asarray(matches), confidences
