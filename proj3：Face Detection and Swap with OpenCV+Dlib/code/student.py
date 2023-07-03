import cv2
import numpy as np
import dlib
from scipy.spatial import Delaunay, tsearch


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


# 获取人脸标志（68个特征点）
def get_landmarks(detector, predictor, img):
    
    '''
    This function first use `detector` to localize face bbox and then use `predictor` to detect landmarks (68 points, dtype: np.array).
    该函数首先使用“检测器”定位人脸框，然后使用“预测器”检测地标（68点，数据类型：np.array）。
    Inputs: 
        detector: a dlib face detector
        predictor: a dlib landmark detector, require the input as face detected by detector
        img: input image
        
    Outputs:
        landmarks: 68 detected landmark points, dtype: np.array

    '''
    
    # TODO: Implement this function!
    # Your Code to detect faces
    dets = detector(img, 1)

    if len(dets) > 1:
        raise TooManyFaces("Too many faces detected.")
    if len(dets) == 0:
        raise NoFaces("No face detected.")
    
    # Your Code to detect landmarks
    shape = predictor(img, dets[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])

    return landmarks


# 根据给定的人脸特征点生成人脸掩码
def get_face_mask(img, landmarks):
    
    '''
    This function gets the face mask according to landmarks.
    
    Inputs: 
        img: input image
        landmarks: 68 detected landmark points, dtype: np.array
        
    Outputs:
        convexhull: face convexhull
        mask: face mask 

    '''
    
    # TODO: Implement this function!
    # 创建一个与输入图像大小相同的全黑图像，作为掩码
    mask = np.zeros_like(img)

    points = np.array(landmarks, np.int32)
    # 使用人脸特征点构建凸包
    convexhull = cv2.convexHull(points)

    # 在掩码上绘制凸包区域，并填充为白色
    cv2.fillConvexPoly(mask, convexhull, (255, 255, 255))

    return convexhull, mask


# 根据据给定的人脸特征点和凸包生成面部三角剖分
def get_delaunay_triangulation(landmarks, convexhull):

    '''
    This function gets the face mesh triangulation according to landmarks.

    Inputs:
        landmarks: 68 detected landmark points, dtype: np.array
        convexhull: face convexhull

    Outputs:
        triangles: face triangles
    '''
    
    # TODO: Implement this function!
    triangles = []
    convexhull = np.squeeze(convexhull)

    # 将凸包中的点与人脸特征点合并
    points = np.concatenate((landmarks, convexhull), axis=0)

    # 使用Delaunay算法进行三角剖分
    tri = Delaunay(points)

    # 循环绘制每个三角形
    for indices in tri.simplices:
        triangle = points[indices]
        triangles.append(triangle)

    return triangles


# 从目标面部到源面部的变换
def affine_transform(source, source_feature_points, target, target_feature_points):
    # 三角剖分
    source_tri = Delaunay(source_feature_points)
    target_tri = Delaunay(target_feature_points)

    # 记录来自哪个三角面片
    # 所有用到Cv2的地方行列都要反过来
    d = target.shape
    index = np.zeros((d[1], d[0])).astype(int)
    for i in range(d[1]):
        for j in range(d[0]):
            index[i, j] = tsearch(target_tri, [i, j])

    # 每个三角形由哪三个顶点组成矩阵
    target_tri_arr = target_tri.simplices

    # 存储变换结果
    ans = np.zeros(target.shape, dtype=np.uint8)

    # 用来存储每个三角面片的仿射变换矩阵
    affine_arr = dict()

    for i in range(d[0]):
        for j in range(d[1]):
            # 找到这个点对应的三个顶点，中间图像的顶点
            # 对应第几个三角形
            tri_num = index[j, i]
            if tri_num != -1:
                point = np.array([[i], [j], [1]])
                if tri_num in affine_arr:
                    # 取出仿射变换矩阵
                    M = affine_arr[tri_num]
                    # 找到在源图像中的像素坐标
                    index_x_s, index_y_s, unuse = M @ point
                    index_x_s = np.round(index_x_s).astype(int)
                    index_y_s = np.round(index_y_s).astype(int)
                    ans[i, j] = source[index_x_s, index_y_s]
                else:
                    # 三个顶点(特征点提取时的第几个点）
                    A, B, C = target_tri_arr[tri_num]
                    # 构造矩阵
                    Ay = target_feature_points[A, 0]
                    Ax = target_feature_points[A, 1]
                    By = target_feature_points[B, 0]
                    Bx = target_feature_points[B, 1]
                    Cy = target_feature_points[C, 0]
                    Cx = target_feature_points[C, 1]
                    # Arr = np.array([[Ax, Bx, Cx], [Ay, By, Cy], [1, 1, 1]])
                    # 得到重心坐标
                    # aph = np.linalg.pinv(Arr) @ point
                    # 通过重心坐标，转换到源图像和目标图像，得到两个图像的像素值
                    Ay_s = source_feature_points[A, 0]
                    Ax_s = source_feature_points[A, 1]
                    By_s = source_feature_points[B, 0]
                    Bx_s = source_feature_points[B, 1]
                    Cy_s = source_feature_points[C, 0]
                    Cx_s = source_feature_points[C, 1]
                    # Arr_s = np.array([[Ax_s, Bx_s, Cx_s], [Ay_s, By_s, Cy_s], [1, 1, 1]])
                    # in_s = Arr_s @ aph
                    pts1 = np.float32([[Ax, Ay], [Bx, By], [Cx, Cy]])
                    pts2 = np.float32([[Ax_s, Ay_s], [Bx_s, By_s], [Cx_s, Cy_s]])
                    M = cv2.getAffineTransform(pts1, pts2)
                    M = np.insert(M, 2, [1, 1, 1], axis=0)
                    # 把三角面对应的仿射变换矩阵加入字典
                    affine_arr[tri_num] = M
                    # 找到在源图像中的像素坐标
                    index_x_s ,index_y_s, unuse = M @ point
                    index_x_s = np.round(index_x_s).astype(int)
                    index_y_s = np.round(index_y_s).astype(int)
                    ans[i, j] = source[index_x_s, index_y_s]
            else:
                continue
    return ans


# 使用仿射变换对图像进行变形
def change_triangle(original_img, target_img, original_pts, target_pts):
    # 提取原图中的三角形顶点坐标
    src_pts = np.float32(original_pts)

    # 提取目标图中的三角形顶点坐标
    dst_pts = np.float32(target_pts)

    original_img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    src_mask = np.zeros_like(original_img_gray)
    src_points = np.array(src_pts, np.int32)
    convexhull_src = cv2.convexHull(src_points)
    cv2.fillConvexPoly(src_mask, convexhull_src, (255, 255, 255))
    masked_src_image = cv2.bitwise_and(original_img, original_img, mask=src_mask)
    # plt.subplot(1, 2, 1)
    # plt.imshow(masked_src_image)
    # plt.axis('off')

    # 计算仿射变换矩阵
    affine_matrix = cv2.getAffineTransform(src_pts, dst_pts)
    # 对原图进行仿射变换
    warped_img = cv2.warpAffine(masked_src_image, affine_matrix, (target_img.shape[1], target_img.shape[0]))

    # 在目标图上绘制变换后的图形
    result_img = target_img.copy()
    result_img_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    dst_mask = np.ones_like(result_img_gray)
    dst_points = np.array(dst_pts, np.int32)
    convexhull_dst = cv2.convexHull(dst_points)
    cv2.fillConvexPoly(dst_mask, convexhull_dst, (0, 0, 0))
    masked_dst_image = cv2.bitwise_and(result_img, result_img, mask=dst_mask)
    result_img = masked_dst_image + warped_img
    # plt.subplot(1, 2, 2)
    # plt.imshow(result_img)
    # plt.axis('off')
    return result_img


def transformation_from_landmarks(target_landmarks, source_landmarks):
    # 计算两组特征点的中心点
    target_center = np.mean(target_landmarks, axis=0)
    source_center = np.mean(source_landmarks, axis=0)

    # 将两组特征点的坐标中心化
    target_landmarks = target_landmarks - target_center
    source_landmarks = source_landmarks - source_center

    # 计算相似性变换的缩放因子s
    target_scale = np.mean(np.sum(target_landmarks ** 2, axis=1)) ** 0.5
    source_scale = np.mean(np.sum(source_landmarks ** 2, axis=1)) ** 0.5
    s = target_scale / source_scale

    # 计算旋转矩阵R
    cov = np.dot(source_landmarks.T, target_landmarks)
    U, S, Vt = np.linalg.svd(cov)
    R = np.dot(U, Vt)

    # 计算仿射变换矩阵M
    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = s * R
    M[:, 2] = target_center - np.dot(M[:, :2], source_center)

    return M


def warp_img(img, M, target_shape):
    # TODO: Implement this function!
    # 获取目标图像大小
    h, w = target_shape[:2]

    # 构建变换矩阵
    M = M[:2, :]
    warped_img = cv2.warpAffine(img, M, (w, h))

    return warped_img


def swap_face(src_img, src_landmarks, dst_img, dst_landmarks, dstmask):
    target_img = dst_img
    warped_img = affine_transform(src_img, src_landmarks, dst_img, dst_landmarks)
    # 在目标图上绘制变换后的图形
    result_img = target_img.copy()
    result_img_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    dst_mask = np.ones_like(result_img_gray)
    convexhull_dst = cv2.convexHull(dst_landmarks)
    cv2.fillConvexPoly(dst_mask, convexhull_dst, (0, 0, 0))
    masked_dst_image = cv2.bitwise_and(result_img, result_img, mask=dst_mask)
    result_img = masked_dst_image + warped_img
    image_mask_index = np.argwhere(dstmask > 0)
    min = np.min(image_mask_index, axis=0)
    max = np.max(image_mask_index, axis=0)
    center_point = ((max[1] + min[1]) // 2, (max[0] + min[0]) // 2)
    eroded_image = cv2.seamlessClone(warped_img, target_img, mask=dstmask, p=center_point, flags=cv2.NORMAL_CLONE)

    return result_img, eroded_image
