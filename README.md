# opencv-
打开电脑中指定的视频文件：video = cv2.VideoCapture("/home/xujiahua/zhuang jia ban/zhuangjiaban1.mp4")

      通过循环来不断地处理视频中的每一帧：while True:（以下代码为嵌套关系）

目标检测
读取每一帧：
ret, img = video.read()   

分离bgr通道：
 blue, g, r = cv2.split(img)

使用阈值二值化处理这一帧图像，蓝色通道>220变为255（白），其余的为黑：
ret2, binary = cv2.threshold(blue, 220, 255, 0)

运用高斯均值滤波去噪点：
Gaussian = cv2.GaussianBlur(binary, (5, 5), 0) 

将前面处理过的图像复制后用于轮廓检测：
draw_img = Gaussian.copy()

获得这帧图片的高和宽；
whole_h, whole_w = binary.shape[:2]

查找轮廓（RETR_TREE获取层级关系，CHAIN_APPROX_NONE保存所有轮廓点）
    contours, hierarchy = cv2.findContours(
        image=draw_img, 
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_NONE
    )
    
将轮廓转换为列表并按面积降序排序
    contours = list(contours)
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    
存储符合条件的矩形参数
    width_array = []
    height_array = []
    point_array = []
    
处理前5大轮廓：
    for cont in contours[:5]:
  获取轮廓的外接矩形 (x,y=左上角坐标, w=宽, h=高)
        x, y, w, h = cv2.boundingRect(cont)
        
  try:
  # 筛选条件：
  1. 高宽比≥2（细长形状）
  2. 高度>图像高度的10%
  3. 高度>宽度（冗余条件）
  if h / w >= 2 and h / whole_h > 0.1 and h > w:
将符合条件的高宽坐标存起来：
  width_array.append(w)
  height_array.append(h)
  point_array.append([x, y])
  跳过异常（如除零错误）
  except:
  continue  
    
寻找面积最接近的两个矩形：
   存储最接近的两个矩形的索引 point_near = [0, 0] 
   最小面积差初始值 min_diff = 10000  # 最小面积差初始值
    
双重循环比较所有矩形对
    for i in range(len(width_array) - 1):
        for j in range(i + 1, len(width_array)):
计算面积绝对差：
    area_diff = abs(width_array[i] * height_array[i] - width_array[j] * height_array[j])
            
更新最小差值及对应索引：
   
           if area_diff < min_diff:
            
           min_diff = area_diff
                
           point_near[0] = i
                
           point_near[1] = j
                

获取选中的两个矩形的左上角坐标    
try:
        
        rect1 = point_array[point_near[0]]
        
        rect2 = point_array[point_near[1]]
        
计算四个关键点（矩形中心线的上下端点）：

        矩形1中心线上点 (x1 + w1/2, y1):
        
        point1 = [rect1[0] + width_array[point_near[0]] / 2, rect1[1]]
        
        矩形1中心线下点 (x1 + w1/2, y1 + h1):
        
        point2 = [rect1[0] + width_array[point_near[0]] / 2, rect1[1] + height_array[point_near[0]]]
        
  矩形2中心线上点 (x2 + w2/2, y2):
  
  
  point3 = [rect2[0] + width_array[point_near[1]] / 2, rect2[1]]
  
  
  矩形2中心线下点 (x2 + w2/2, y2 + h2):
  
  
  point4 = [rect2[0] + width_array[point_near[1]] / 2, rect2[1] + height_array[point_near[1]]]
        
打印坐标点（调试用）
        
        print(point1, point2, point3, point4)
        
构建四边形顶点：point1 -> point2 -> point4 -> point3
        
        pts = np.array([point1, point2, point4, point3], np.int32)

重塑为OpenCV所需格式 (n,1,2)
        
        polygon = pts.reshape((-1, 1, 2)).astype(np.int32)
        
在原始图像上绘制绿色四边形
        
        cv2.polylines(img, [polygon], True, (0, 255, 0), 2)
        
except Exception as e:
        异常处理（如索引错误）
        print(f"Error: {e}")
        continue
    
显示处理结果:
    
    cv2.imshow('name', img)
    
按'q'键退出:
    if cv2.waitKey(45) & 0xFF == ord('q'):
        break

# 释放资源
video.release()
cv2.destroyAllWindows()




 
