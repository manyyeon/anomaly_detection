import numpy as np
from matplotlib import pyplot as plt
import time
import torch
import sys

data_file = "data/1920001_L.dat"

print("Reading data file", data_file, "...")
ued_image = np.genfromtxt(data_file)

print("Shape:", ued_image.shape)

plt.imshow(np.log(ued_image))
plt.axis('off')  # Optional: removes axis for a cleaner image
plt.savefig("anomaly2.png", bbox_inches='tight', pad_inches=0)  # Saves the image
plt.show()

print("Done.")
#
# class data_preprocess:
#     '''''
#         Usage:
#             Loads the training and test data, breaks them in tiles and standarizes with respect to the training
#             data. The images are textfiles in format .dat (whatever it is)
#
#             data_preprocess(data_list,tile_size=80, overlap=44, threshold=6)
#                 Declares a data preprocessing object
#                 data_list: list of paths to images by default.
#                 tile_size, overlap, threshold: parameters of the tiling process.
#
#             load(datatype,num_data='all',datalist='none',randomize=True)
#                 Loads, tiles and standardizes trainng or test data
#                 datatype:  'train' or 'test' Allocates each data in different variables. In training
#                            the data is normailzed with its mean and std. In test, the train data stats are used.
#                 numdata:   Number of samples to be read from the datalist. If 'all', the whole list is used.
#                 datalist:  List of paths to images. If 'none', the default one declared before is used.
#                 randomize: If true, reads numdata samples at random from the list. Otherwise, it uses the
#                            first numdata samples.
#
#             Variables:
#                 train_tiles
#                 test_tiles
#                 test_image_indexes: indexes of the first one of the tiles corresponding to each image. For example,
#                                     the tiles of image 0 are located in test_tiles in postions test_image[0]
#                                     to test_image[1]-1. Useful to compute the reconstruction error of a given
#                                     test image
#
#             Other variables:
#                 list:               Used data list
#                 train_images
#                 test_images
#                 train_loaded:       Boolean
#                 test_loaded:        Boolean
#                 train_standardized: Boolean
#                 test_standardized:  Boolean
#                 tile_size
#                 overlap
#                 threshold
#                 mean:               standardization mean
#                 std:                standardization std
#
#             Example usage
#                 preprocess=data_preprocess('my_data_list')
#                 preprocess.load('train',num_data=100)  # Reads 100 data at random from 'my_data_list' and it stores it in
#                                                        # preprocess.train_tiles
#                 preprocess.load('test',num_data=100)   # Reads all the data from 'my_data_list', stores it in
#                                                        # preprocess.test_tiles and indexes it in
#                                                        # preprocess.test_image_indexes
#                 preprocess.load('test',datalist='my_image.dat') # Loads a single image in test_tiles.
#
#     '''''
#     def __init__(s,data_list,tile_size=80, overlap=44, threshold=6):
#         s.list=data_list
#         s.train_loaded=False
#         s.test_loaded=False
#         s.train_standardized=False
#         s.test_standardized=False
#         s.tile_size=tile_size
#         s.overlap=overlap
#         s.threshold=threshold
#
#     def standardize(s,datatype):
#         if datatype=='train':
#             T_train_data = torch.from_numpy(s.train_tiles).float()
#             s.mean =0* T_train_data.mean().numpy()
#             s.std = T_train_data.std().numpy()
#
#             s.train_tiles=(s.train_tiles-s.mean)/s.std
#             s.train_standardized=True
#
#         if datatype=='test':
#             if s.train_standardized:
#                 s.test_tiles=(s.test_tiles-s.mean)/s.std
#                 s.test_standardized=True
#
#     def train_standardize(s):
#         T_train_data = torch.from_numpy(s.train_tiles).float()
#         s.mean = T_train_data.mean().numpy()
#         s.std = T_train_data.std().numpy()
#         s.train_tiles=(s.train_tiles-s.mean)/s.std
#         s.train_standardized=True
#
#     def test_standardize(s):
#         if s.train_standardized:
#             s.test_tiles=(s.test_tiles-s.mean)/s.std
#             s.test_standardized=True
#
#     def tile_image(s,image):
#         height, width = image.shape[:2]
#         if (height-s.overlap)/(s.tile_size-s.overlap)!=np.ceil((height-s.overlap)/(s.tile_size-s.overlap)): # Checks that the parameters are correct. If not, shows a list of acceptable overlaps for the specified tile size.
#             print("The overlaps for a tile size of "+str(s.tile_size)+" can only be:")
#             for w in range(1,s.tile_size):
#                 if (height-w)/(s.tile_size-w)==np.ceil((height-w)/(s.tile_size-w)):
#                     print(w)
#             sys.exit("")
#         tiles = []
#         for y in range(0, height-s.overlap, s.tile_size - s.overlap):
#             for x in range(0, width-s.overlap, s.tile_size - s.overlap):
#                 tile = image[y:min(y + s.tile_size, height), x:min(x + s.tile_size, width)]
#                 I=tile.reshape(s.tile_size**2,1)
#                 z=np.zeros(I.shape)
#                 m=np.median(I)
#                 z[np.argwhere(I>3*m)]=1
#                 mean=np.sum(z)
#                 if mean >= s.threshold:
#                     tiles.append(tile)
#         return tiles
#
#     def tile_data(s,datatype):
#         if datatype == 'train':
#             images=s.train_images
#         else:
#             images=s.test_images
#             test_image_indexes=[0]
#
#         height, width = images.shape[:2]
#         N=images.shape[2]
#         processTime=AlgorithmTimer(N)
#         max_num_tiles=int(((height-s.overlap)/(s.tile_size - s.overlap))**2)
#         temp_tiles=np.zeros((max_num_tiles*N,s.tile_size,s.tile_size))
#         index=0
#         processTime.restart(N)
#         for i in range(N):
#             image = images[:,:,i]
#             tiles = s.tile_image(image)
#             x=np.array(tiles)
#             number_of_tiles=len(tiles)
#             temp_tiles[index:index+number_of_tiles,:,:]=x
#             index=index+number_of_tiles
#             if datatype=='test':
#                 test_image_indexes.append(index)
#             message="Tiling, "+str(i+1)+"/"+str(N)+'('+ str(number_of_tiles)+ ')'
#             processTime.show_time(message)
#         print("                                                                                                \r",end='',flush=True)
#         print("Number of "+datatype+" tiles: "+str(index))
#
#         if datatype == 'train':
#             s.train_tiles=temp_tiles[:index,:,:]
#         else:
#             s.test_tiles=temp_tiles[:index,:,:]
#             s.test_image_indexes=test_image_indexes
#
#     def load(s,datatype,num_data='all',datalist='none',randomize=True):
#         if datalist=='none':
#             t_list=s.list
#         else:
#             t_list=datalist
#         if num_data=='all':
#             num_data=len(t_list)
#
#         f = t_list[0]
#         image = np.genfromtxt(f)
#         size=image.shape[0]
#         if randomize==True:
#             Ind=np.random.choice(len(t_list), size=num_data, replace=False)
#         else:
#             Ind = range(num_data)
#         images=np.zeros((size,size,num_data))
#         processTime=AlgorithmTimer(num_data)
#         for i,ind in enumerate(Ind):
#             f = t_list[ind]
#             images[:,:,i] = np.genfromtxt(f)
#             processTime.show_time('Reading images. ')
#
#         if datatype=='train':
#             s.train_loaded=True
#             s.train_images=images
#         if datatype=='test':
#             s.test_loaded=True
#             s.test_images=images
#
#         s.tile_data(datatype)
#         #s.standardize(datatype)
#
# class Autoencoder2(nn.Module): #We call this autoencoder 2
#     def __init__(self, input_shape,dropout_rate=0.2):
#         super(Autoencoder2, self).__init__()
#         self.dropout_rate = dropout_rate
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(4),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(4),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(4),
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 256, kernel_size=5, stride=5, padding=0),
#             nn.ReLU(),
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4, padding=0),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(64, 1, kernel_size=3, padding=1),
#             #nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         xb = self.encoder(x)
#         x = self.decoder(xb)
#         return x, xb
#
# class AlgorithmTimer:
#     def __init__(self,Num_iterations):
#         self.start_time = None
#         self.end_time = None
#         self.elapsed_time = None
#         self.iteration = 0
#         self.total_time = 0
#         self.num_iterations=Num_iterations
#     sround = lambda x,p: float(f'%.{p-1}e'%x) # This is to print numbers with a given number od decimals
#     def start(self):
#         self.start_time = time.time()
#
#     def end(self):
#         self.end_time = time.time()
#         self.elapsed_time = self.end_time - self.start_time
#
#     def restart(self,Num_iterations):
#         self.start_time = None
#         self.end_time = None
#         self.elapsed_time = None
#         self.iteration = 0
#         self.total_time = 0
#         self.num_iterations=Num_iterations
#
#     def hms(self,t):
#         h=np.floor(t/3600)
#         t=t-h*3600
#         m=np.min([59,np.floor(t/60)])
#         s=np.min([59,t-m*60])
#         hstr=f'{h:.0f}'
#         mstr=f'{m:.0f}'
#         if np.floor(m)<=9:
#             mstr=f'0{np.floor(m):.0f}'
#         sstr=f'{s:.0f}'
#         if np.floor(s)<=9:
#             sstr=f'0{np.floor(s):.0f}'
#         timestr=hstr+':'+mstr+':'+sstr
#         return timestr
#
#     def show_time(self,Message=''):
#         if self.start_time is not None:
#             self.iteration=self.iteration+1
#             self.end()
#             self.total_time=self.total_time+self.elapsed_time
#             t=self.elapsed_time
#             T=self.num_iterations*self.total_time/self.iteration
#             totaltime=self.hms(T)
#             RT=T-self.total_time
#             remtime=self.hms(RT)
#             print(Message+f' Iteration time: {sround(t,2):.2f} s. Total time: ' +totaltime+'. Remaining time: '+remtime+'           \r',end='',flush=True)
#         self.start()
#
#     def total_Time(self):
#         if self.start_time is not None:
#             return self.total_time
#
#     def show_total_time(self):
#         if self.start_time is not None:
#             timestr=self.hms(self.total_time)
#             print("Total time: " +timestr+"                                    " )
#             return self.total_time
#
# # Parameters of the training
# tile_size=80
# #overlap=44
# threshold=10
# batch_size=32
# input_shape = (tile_size, tile_size, 1)
# model = Autoencoder2(input_shape)
# N_train=100 # This is the number of training data (we choose later the first N_train images)
# # With N_train = 146 we get 10,000 tiles of 80x80 and overlap 44
#
# # List of data
# data_list = []
# for filename in os.listdir(data_path):
#     if filename.endswith(".dat"):
#         data_list.append(data_path+filename)
#
#
# preprocess=data_preprocess(data_list,threshold=threshold)
# preprocess.load('train',randomize=True,num_data=N_train)
# traindata=preprocess.train_tiles
#
# preprocess.load('test',randomize=True,num_data=3)
# valdata=preprocess.test_tiles