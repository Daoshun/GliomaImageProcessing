import os, shutil
# 数据集解压之后的目录
original_dataset_dir = '/home/vincent/data/GliomaImage'
# 存放小数据集的目录
base_dir = '/home/vincent/data/GliomaImageProcessing'
os.mkdir(base_dir)
# 建立训练集、验证集、测试集目录
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
# 将照片按照训练、验证、测试分类
train_T1CY_dir = os.path.join(train_dir, 'T1CY')
os.mkdir(train_T1CY_dir)
train_T1CN_dir = os.path.join(train_dir, 'T1CN')
os.mkdir(train_T1CN_dir)
validation_T1CY_dir = os.path.join(validation_dir, 'T1CY')
os.mkdir(validation_T1CY_dir)
validation_T1CN_dir = os.path.join(validation_dir, 'T1CN')
os.mkdir(validation_T1CN_dir)
test_T1CY_dir = os.path.join(test_dir, 'T1CY')
os.mkdir(test_T1CY_dir)
test_T1CN_dir = os.path.join(test_dir, 'T1CN')
os.mkdir(test_T1CN_dir)

# collect all t1cy dicom images
lstFilesT1CYDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(original_dataset_dir):  # 遍历base data
    for filename in fileList:
        if '.png' in filename.lower():
            if 'T1C.Y.' in filename.upper():
                lstFilesT1CYDCM.append(os.path.join(dirName, filename))
# print(lstFilesDCM)
for filename in lstFilesT1CYDCM:
    src = os.path.join(original_dataset_dir, filename)
    # dst = os.path.join(train_T1CY_dir)
    shutil.copy(src, train_T1CY_dir)  # copy到train
# add validation data
lstFilesT1CYDCM_validation = ['T1C.Y.{}.png'.format(i) for i in range(40, 57)]  # range(0,2)=0,1
for filename in lstFilesT1CYDCM_validation:
    src = os.path.join(train_T1CY_dir, filename)
    shutil.move(src, validation_T1CY_dir)  # move images
# collect all t1cn dicom images
lstFilesT1CNDCM = []
for dirName, subdirList, fileList in os.walk(original_dataset_dir):
    for filename in fileList:
        if '.png' in filename.lower():
            if 'T1C.N.' in filename.upper():
                lstFilesT1CNDCM.append(os.path.join(dirName, filename))
for filename in lstFilesT1CNDCM:
    src = os.path.join(original_dataset_dir, filename)
    shutil.copy(src, train_T1CN_dir)
# add validation data
lstFilesT1CNDCM_validation = ['T1C.N.{}.png'.format(i) for i in range(40, 48)]
for filename in lstFilesT1CNDCM_validation:
    src = os.path.join(train_T1CN_dir, filename)
    shutil.move(src, validation_T1CN_dir)

# # 切割数据集
# path = '/home/vincent/data/GliomaImage/'
# dst_dir = os.path.abspath(r'/home/vincent/data/GliomaImageProcessing/train/T1CY')
# images = os.listdir(path)
# images.sort()
#
# for dirName, subdirList, fileList in os.walk(path):
#     for filename in fileList:
#         if '.png' in filename.lower():
#             if 'T1C.Y.' in filename.upper():
#                 src_file = os.path.join(dirName, filename)
#                 shutil.copy(src_file, dst_dir)
