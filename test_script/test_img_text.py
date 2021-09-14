import cv2

# 读入图片
img = cv2.imread(
    "/Users/huangyin/Documents/project/base-dl/datasets/Market-1501-v15.09.15_reduce/bounding_box_test/1371_c5s3_072662_02.jpg"
)
img = cv2.resize(img, (128, 384))
cv2.putText(
    img,
    "avc",
    (0, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0, 0, 255),
    1,
)
cv2.imshow("text", img)
cv2.waitKey(2000)
cv2.destroyAllWindows()
