class Encoder(nn.Module):
 	def __init__(self, encoded_img_size):
 		super(Encoder, self).__init__()

 		self.densenet = torchvision.models.densenet121(pretrained = True) #Pretrained ImageNet DenseNet-121
 		self.model.classifier = nn.Linear(in_features = 1024, out_features = 1024)
 		self.embed = nn.Linear(in_features = 1024, out_features = encoded_img_size)
 		self.dropout = nn.Dropout(p = 0.25)
 		self.relu = nn.ReLU()

 	def forward(self, images):
 		"""
 		Forward Propagation
 		parameters - images : tensor of dimensions (batch_size, nc, image_size, image_size)
 		return - encoded information of images
 		"""
 		densenet_outputs = self.dropout(self.relu(self.densenet(images)))
 		embeddings = self.embed(densenet_outputs)
 		return embeddings

 	def fine_tune(self, fine_tune = True):
 		for p in self.resnet.parameters():
 			p.requires_grad = fine_tune
