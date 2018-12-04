import os, shutil
# 数据集解压之后的目录
original_dataset_dir = '/home/vincent/data/GliomaImage'
# 存放小数据集的目录
base_dir = '/home/vincent/data/FlairGliomaImageProcessing'
os.mkdir(base_dir)
# 建立训练集、验证集、测试集目录
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
# 将照片按照训练、验证、测试分类
train_FlairY_dir = os.path.join(train_dir, 'FlairY')
os.mkdir(train_FlairY_dir)
train_FlairN_dir = os.path.join(train_dir, 'FlairN')
os.mkdir(train_FlairN_dir)
validation_FlairY_dir = os.path.join(validation_dir, 'FlairY')
os.mkdir(validation_FlairY_dir)
validation_FlairN_dir = os.path.join(validation_dir, 'FlairN')
os.mkdir(validation_FlairN_dir)
test_FlairY_dir = os.path.join(test_dir, 'FlairY')
os.mkdir(test_FlairY_dir)
test_FlairN_dir = os.path.join(test_dir, 'FlairN')
os.mkdir(test_FlairN_dir)

# collect all t1cy dicom images
lstFilesT1CYDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(original_dataset_dir):  # 遍历base data
    # print(fileList)
    for filename in fileList:
        # print(filename.upper())
        if '.png' in filename.lower():
            if 'FLAIR.Y.' in filename.upper():
                lstFilesT1CYDCM.append(os.path.join(dirName, filename))
# print(lstFilesT1CYDCM)
for filename in lstFilesT1CYDCM:
    src = os.path.join(original_dataset_dir, filename)
    # dst = os.path.join(train_T1CY_dir)
    shutil.copy(src, train_FlairY_dir)  # copy到train
# add validation data
lstFilesT1CYDCM_validation = ['Flair.Y.{}.png'.format(i) for i in range(40, 57)]  # range(0,2)=0,1
for filename in lstFilesT1CYDCM_validation:
    src = os.path.join(train_FlairY_dir, filename)
    shutil.move(src, validation_FlairY_dir)  # move images
# collect all t1cn dicom images
lstFilesT1CNDCM = []
for dirName, subdirList, fileList in os.walk(original_dataset_dir):
    for filename in fileList:
        if '.png' in filename.lower():
            if 'FLAIR.N.' in filename.upper():
                lstFilesT1CNDCM.append(os.path.join(dirName, filename))
for filename in lstFilesT1CNDCM:
    src = os.path.join(original_dataset_dir, filename)
    shutil.copy(src, train_FlairN_dir)
# add validation data
lstFilesT1CNDCM_validation = ['Flair.N.{}.png'.format(i) for i in range(40, 48)]
for filename in lstFilesT1CNDCM_validation:
    src = os.path.join(train_FlairN_dir, filename)
    shutil.move(src, validation_FlairN_dir)

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
