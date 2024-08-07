import torch
import torch.nn as nn
import torch.nn.functional as F
from .select_backbone import select_resnet
import math

class Audio_RNN(nn.Module):
	def __init__(self, img_dim, network='resnet50', num_layers_in_fc_layers = 1024, dropout=0.5):
		super(Audio_RNN, self).__init__();

		self.__nFeatures__ = 24;
		self.__nChs__ = 32;
		self.__midChs__ = 32;

		self.netcnnaud = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(1,1), stride=(1,1)),

			nn.Conv2d(64, 192, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
			nn.BatchNorm2d(192),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(3,3), stride=(1,2)),

			nn.Conv2d(192, 384, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(384),
			nn.ReLU(inplace=True),

			nn.Conv2d(384, 256, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),

			nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),

			nn.Conv2d(256, 512, kernel_size=(5,4), padding=(0,0)),
			nn.BatchNorm2d(512),
			nn.ReLU(),
		);

		self.netfcaud = nn.Sequential(
			nn.Linear(512*21, 4096),
			nn.BatchNorm1d(4096),
			nn.ReLU(),
			nn.Linear(4096, num_layers_in_fc_layers),
		);

		self.netcnnlip, self.param = select_resnet(network, track_running_stats=True)
		self.last_duration = int(math.ceil(30 / 4))
		self.last_size = int(math.ceil(img_dim / 32))

		self.netfclip = nn.Sequential(
			nn.Linear(self.param['feature_size']*self.last_size*self.last_size, 4096),
			nn.BatchNorm1d(4096),
			nn.ReLU(),
			nn.Linear(4096, num_layers_in_fc_layers),
		);

		self.final_bn_lip = nn.BatchNorm1d(num_layers_in_fc_layers)
		self.final_bn_lip.weight.data.fill_(1)
		self.final_bn_lip.bias.data.zero_()

		self.final_fc_lip = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_layers_in_fc_layers, 2))
		self._initialize_weights(self.final_fc_lip)

		self.final_bn_aud = nn.BatchNorm1d(num_layers_in_fc_layers)
		self.final_bn_aud.weight.data.fill_(1)
		self.final_bn_aud.bias.data.zero_()

		self.final_fc_aud = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_layers_in_fc_layers, 2))

		# Add Final Classification
		self.final_classification = nn.Sequential(
			nn.Linear(1024 * 2, 512),  # assuming vid_out and aud_out are both 1024-dimensional
			nn.ReLU(inplace=True),
			nn.Linear(512, 2)  # binary classification
		)

		self._initialize_weights(self.final_fc_aud)
		self._initialize_weights(self.netcnnaud)
		self._initialize_weights(self.netfcaud)
		self._initialize_weights(self.netfclip)

	def forward_aud(self, x):
		(B, N, N, H, W) = x.shape
		x = x.view(B*N, N, H, W) #(4,1,13,99)
		mid = self.netcnnaud(x) #(4,512,1,21)
		mid = mid.view((mid.size()[0], -1)) #(4,10752)
		out = self.netfcaud(mid) # (4,1024)
		return out

	def forward_lip(self, x):
		(B, N, C, NF, H, W) = x.shape
		x = x.view(B*N, C, NF, H, W) # (4,3,30,224,224)
		feature = self.netcnnlip(x) # (4,256,8,7,7)
		feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1)) # # (4,256,1,7,7)
		feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # # (4,1,256,7,7)
		feature = feature.view((feature.size()[0], -1))  # (4,12544)
		out = self.netfclip(feature) #(4,1024)
		return out

	def final_classification_lip(self,feature):
		feature = self.final_bn_lip(feature)
		output = self.final_fc_lip(feature)
		return output

	def final_classification_aud(self,feature):
		feature = self.final_bn_aud(feature)
		output = self.final_fc_aud(feature)
		return output

	def forward_lipfeat(self, x):
		mid = self.netcnnlip(x)
		out = mid.view((mid.size()[0], -1))
		return out

	def forward(self, video, audio):
		vid_out_feat = self.forward_lip(video) # vid_out torch.Size([4, 1024])
		aud_out_feat = self.forward_aud(audio) # vid_out torch.Size([4, 1024])

		fused_features = torch.cat((vid_out_feat, aud_out_feat), dim=1)
		final_out = self.final_classification(fused_features) # torch.Size([4, 2])

		vid_class = self.final_classification_lip(vid_out_feat) # torch.Size([4, 2])
		aud_class = self.final_classification_aud(aud_out_feat) # torch.Size([4, 2])

		return final_out, vid_out_feat, aud_out_feat, vid_class, aud_class

	def _initialize_weights(self, module):
		for m in module:
			if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.ReLU) or isinstance(m,nn.MaxPool2d) or isinstance(m,nn.Dropout):
				pass
			else:
				m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
				if m.bias is not None: m.bias.data.zero_()