import numpy as np
 
def plane_xequal1(point):
    return (point[1], point[2], 1 - point[0])

def plane_yequal1(point):
    return (point[0], point[2], 1 - point[1])

def plane_zequal1(point):
    return (point[0], point[1], 1 - point[2])

def plane_xequal_neg1(point):
    return (point[1], point[2], abs(-1 - point[0]))

def plane_yequal_neg1(point):
    return (point[0], point[2], abs(-1 - point[1]))

def plane_zequal_neg1(point):
    return (point[0], point[1], abs(-1 - point[2]))

planes = [plane_xequal1, plane_yequal1, plane_zequal1, plane_xequal_neg1, plane_yequal_neg1, plane_zequal_neg1]

def rasterize(point_cloud, img_width, img_height, planes):
    channels = len(planes)
    projections = []
    for plane in planes:
        projection = [plane(point) for point in point_cloud]
        projections.append(projection)
    projections = np.array(projections)
    projections[:,:,0:2] = (projections[:,:,0:2] * (img_width/2) + (img_width/2)).astype(np.int16)
    projections[:,:,2] = projections[:,:,2]/2
    
    img = np.zeros((channels, img_width, img_height))
    for i in range(channels):
        projection = projections[i]
        rev_intensity = projection[projection[:,2].argsort()]
        rev_intensity = rev_intensity[::-1]
        for point in rev_intensity:
            img[i][int(point[0])][int(point[1])] = 1 - point[2]
    return img