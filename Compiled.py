try:
	import numpy as np
	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	import torch.optim as optim
	import torchvision.datasets as dset
	import torchvision.transforms as transforms
except:
	print("Error in importing packages")
	exit()

workers = 2
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
nd1 = 64
num_epochs = 5
lr = 0.0002
beta1 = 0.5
ngpu = 1

class Generator_one(nn.Module):
	def __init__(self):
		super(Generator_one,self).__init__()
		self.main = nn.Sequential(
			#input is z, going into convolution
			nn.ConvTranspose2d(nz, ngf * 8, 4, bias = False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
			#State size (ngf * 8) x 4 x 4
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, bias = False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True),
			#State size (ngf * 4) x 8 x 8 
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, bias = False),
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(True),
			#State size (ngf * 2) x 16 x 16
			nn.ConvTranspose2d(ngf * 2, ngf , 4, bias = False),
			nn.BatchNorm2d(ngf),
			nn.ReLU(True),
			#State size (ngf) x 32 x 32
			nn.ConvTranspose2d(ngf, nc, 4, bias = False),
			nn.Tanh()
			#State size nc x 64 x 64
		)

	def forward(self,input):
		return self.main(input)

class Discriminator_one(nn.Module):
	def __init__(self):
		super(Discriminator_one, self).__init__()
		self.main = nn.Sequential(
			#Input is nc x 64 x 64
			nn.Conv2d(nc, nd1, 4, bias = False),
			nn.LeakyReLU(0.2, inplace = True),
			#State size (nd1) x 32 x 32
			nn.Conv2d(nd1, nd1 * 2, 4 , bias = False),
			nn.BatchNorm2d(nd1 * 2),
			nn.LeakyReLU(0.2, inplace = True),
			#State size (nd1 * 2) x 16 x 16
			nn.Conv2d(nd1 * 2, nd1 * 4, 4 , bias = False),
			nn.BatchNorm2d(nd1 * 4),
			nn.LeakyReLU(0.2, inplace = True),
			#State size (nd1 * 4) x 8 x 8
			nn.Conv2d(nd1 * 4, nd1 * 8, 4 , bias = False),
			nn.BatchNorm2d(nd1 * 8),
			nn.LeakyReLU(0.2, inplace = True),
			#State size (nd1 * 8) x 4 x 4
			nn.Conv2d(nd1 * 8, 1, 4, 1, 0, bias = False),
			nn.Sigmoid()
		)
	def forward(self, input):
		return self.main(input)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
# class Encoder(nn.Module):
#  	def __init__(self, encoded_img_size):
#  		super(Encoder, self).__init__()

#  		self.densenet = torchvision.models.densenet121(pretrained = True) #Pretrained ImageNet DenseNet-121
#  		self.model.classifier = nn.Linear(in_features = 1024, out_features = 1024)
#  		self.embed = nn.Linear(in_features = 1024, out_features = encoded_img_size)
#  		self.dropout = nn.Dropout(p = 0.25)
#  		self.relu = nn.ReLU()

#  	def forward(self, images):
#  		"""
#  		Forward Propagation
#  		parameters - images : tensor of dimensions (batch_size, nc, image_size, image_size)
#  		return - encoded information of images
#  		"""
#  		densenet_outputs = self.dropout(self.relu(self.densenet(images)))
#  		embeddings = self.embed(densenet_outputs)
#  		return embeddings

#  	def fine_tune(self, fine_tune = True):
#  		for p in self.resnet.parameters():
#  			p.requires_grad = fine_tune

# class DecoderRNN(nn.Module):
# 	def __init__(self, embed_size, hidden_size, vocab_size, num_layers = 1):
# 		super(DecoderRNN, self).__init__()
# 		self.embed_size = embed_size
# 		self.hidden_size = hidden_size
# 		self.vocab_size = vocab_size

# 		#lstm cell
# 		self.lstm_cell = nn.LSTMCell(input_size = embed_size, hidden_size = hidden_size)
# 		#output fully connected layer
# 		self.fc_out = nn.Linear(in_features = hidden_size, out_features = vocab_size)
# 		#embeddding layer 
# 		self.embed = nn.Embedding(num_embedding = vocab_size, embedding_dim = embed_size)
# 		#activations
# 		self.softmax = nn.Softmax(dim = 1)

# 	def forward(self, features, captions):
# 		batch_size = features.size(0)
# 		#initialise the hidden and cell states to zeros
# 		hidden_state = torch.zeros((batch_size, self.hidden_size))
# 		cell_state = torch.zeros((batch_size, self.hidden_size))

# 		#define the output tensor placeholder
# 		outputs = torch.empty((batch_size, captions_size(1), self.vocab_size))

# 		#embed the captions 
# 		captions_embed = self.embed(captions)

# 		#pass the captions word by word
# 		for t in range(captions.size(1)):
# 			# for the first time step the input is the feature vector
# 			if t == 0:
# 				hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))

# 			else:
# 				hidden_state, cell_state = self.lstm_cell(captions_embed[:, t, :], (hidden_state, cell_state))

# 			#output of the attention mechanism
# 			out = self.fc_out(hidden_state)

# 			#build the output tensor
# 			outputs[:, t, :] = out

# 		return output

# class Attention(nn.Module):
# 	def __init__(self, encoder_dim, decoder_dim, attention_dim):
# 		super(Attention, self).__init__()
# 		self.encoder_dim = encoder_dim
# 		self.decoder_dim = decoder_dim
# 		self.attention_dim = attention_dim

# 	def forward(self, encoder_out, decoder_hidden):
# 		"""
# 		Forward Propagation
# 		parameters :encoder_out - encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
# 					decoder_hidden - previous decoder output, a tensor of dimension (batch_size, decoder_dim)

# 		return : Attention weighted encoding, weights
# 		"""

# 		att1 = nn.Linear(encoder_dim, attention_dim)(encoder_out) # (batch_size, num_pixels, attention_dim)
# 		att2 = nn.Linear(decoder_dim. attention_dim)(decoder_hidden)  # (batch_size, attention_dim)
# 		att = nn.Linear(attention_dim, 1)(nn.ReLU()(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
# 		alpha = nn.Softmax(dim = 1)(att) # (batch_size, num_pixels)
# 		attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim = 1)   # (batch_size, encoder_dim)
		
# 		return attention_weighted_encoding, alpha

# class DecoderWithAttention(nn.Module):
# 	"""
# 	DecoderWithAttention
# 	"""
# 	def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim = 2048, dropout = 0.25):
# 		"""
# 		parameters: attention_dim : size of attention network
# 					embed_dim : embedding size
# 					decoder_dim: size of decoder's RNN
# 					vocab_size: size of vocabulary
# 					encoder_dim: feature size of encoded images
# 					dropout: dropout
# 		"""
# 		super(DecoderWithAttention, self).__init__()
# 		self.attention_dim = attention_dim
# 		self.embed_dim = embed_dim
# 		self.decoder_dim = decoder_dim
# 		self.vocab_size = vocab_size
# 		self.encoder_dim = encoder_dim
# 		self.droput = droput

# 		self.attention = Attention(encoder_dim, decoder_dim, attention_dim) #attention network

# 	def init_weights(self):
# 		"""
# 		Initializes some parameters with values from the uniform distribution, for easier convergence.
# 		"""
# 		self.embedding.weights.data.uniform_(-0.1, 0.1)
# 		self.fc.bias.data.fill_(0)
# 		self.fc.weight.data.uniform_(-0.1, 0.1)

# 	def load_pretrained_embeddings(self, embeddings):
# 		"""
# 		Loads embedding layer with pre-trained embeddings.
# 		param embeddings: pre-trained embeddings
# 		"""
# 		self.embedding.weight = nn.Parameter(embeddings)

# 	def fine_tune_embeddings(self, fine_tune = True):
# 		for p in self.embedding.parameters():
# 			p.requires_grad = fine_tune

# 	def init_hidden_state(self, encoder_out):
# 		"""
# 		Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
# 		param : encoder_out : encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
# 		return : hidden state, cell state
# 		"""

# 		mean_encoder_out = encoder_out.mean(dim = 1)
# 		h = self.init_h(mean_encoder_out)
# 		c = self.init_c(mean_encoder_out)
# 		return h, c

# 	def forward(self, encoder_out, encoded_captions, caption_lengths):
# 		"""
# 		Forward propagation

# 		param encoder_out : encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
# 			encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
# 			caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
# 		return scores for vocabulary, sorted encoded captionsscores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices, decode lengths, weights, sort indices
# 		"""		
# 		batch_size = encoder_out.size(0)
# 		encoder_dim = encoder_out.size(-1)
# 		vocab_size = self.vocab_size
# 		#Flatten image
# 		encoder_out = encoder_out.view(batch_size, -1, encoder_dim)

# class Kriti():
# 	def __init__(self, arg):
# 		super(Kriti, self).__init__()
# 		self.arg = arg
		

# def data_loading():
	
path2data = "./data/val2017"
path2json = "./data/annotations/captions_val2017.json"
data = dset.CocoDetection(root = path2data, annFile = path2json)
# print('Number of samples: ', len(data))
# img, target = data[3]
# print(np.shape(img))
# print(target)
criterion = nn.BCELoss()
real_label = 1
fake_label = 0

netG = Generator_one()
netD = Discriminator_one()
# netLSTM = LSTM()
netG.apply(weights_init)
netD.apply(weights_init)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
#For each epoch

for epoch in range(num_epochs):
	for img, target in data:
		#(1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
		# Train with all-real batch
		netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        #Train with all-fake batch
        word_encoding = 
        fake = netG(word_encoding)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        # (2) Update G network: maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

